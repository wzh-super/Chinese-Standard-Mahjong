from multiprocessing.shared_memory import SharedMemory, ShareableList
from multiprocessing import resource_tracker
import _pickle as cPickle
import time


def _unregister_from_resource_tracker(name):
    """从 resource_tracker 中注销共享内存，防止进程退出时被自动清理"""
    try:
        resource_tracker.unregister(f'/{name}' if not name.startswith('/') else name, 'shared_memory')
    except Exception:
        pass

class ModelPoolServer:
    
    def __init__(self, capacity, name):
        self.capacity = capacity
        self.n = 0
        self.model_list = [None] * capacity
        # shared_model_list: N metadata {id, _addr} + n
        metadata_size = 1024
        self.shared_model_list = ShareableList([' ' * metadata_size] * capacity + [self.n], name = name)
        
    def push(self, state_dict, metadata = {}):
        n = self.n % self.capacity
        if self.model_list[n]:
            # FIFO: release shared memory of older model
            try:
                self.model_list[n]['memory'].close()
                self.model_list[n]['memory'].unlink()
            except FileNotFoundError:
                pass  # 共享内存可能已被其他进程的 resource_tracker 清理
        
        data = cPickle.dumps(state_dict) # model parameters serialized to bytes
        memory = SharedMemory(create = True, size = len(data))
        memory.buf[:] = data[:]
        # print('Created model', self.n, 'in shared memory', memory.name)
        
        metadata = metadata.copy()
        metadata['_addr'] = memory.name
        metadata['id'] = self.n
        self.model_list[n] = metadata
        self.shared_model_list[n] = cPickle.dumps(metadata)
        self.n += 1
        self.shared_model_list[-1] = self.n
        metadata['memory'] = memory

class ModelPoolClient:
    
    def __init__(self, name):
        while True:
            try:
                self.shared_model_list = ShareableList(name = name)
                # Client 只应连接共享内存，不应在退出时由 resource_tracker 自动 unlink（由 Server 统一管理）
                try:
                    _unregister_from_resource_tracker(self.shared_model_list.shm.name)
                except Exception:
                    pass
                n = self.shared_model_list[-1]
                break
            except:
                time.sleep(0.1)
        self.capacity = len(self.shared_model_list) - 1
        self.model_list = [None] * self.capacity
        self.n = 0
        self._update_model_list()
    
    def _update_model_list(self):
        n = self.shared_model_list[-1]
        if n > self.n:
            # new models available, update local list
            for i in range(max(self.n, n - self.capacity), n):
                self.model_list[i % self.capacity] = cPickle.loads(self.shared_model_list[i % self.capacity])
            self.n = n
    
    def get_model_list(self):
        self._update_model_list()
        model_list = []
        if self.n >= self.capacity:
            model_list.extend(self.model_list[self.n % self.capacity :])
        model_list.extend(self.model_list[: self.n % self.capacity])
        return model_list
    
    def get_latest_model(self):
        self._update_model_list()
        while self.n == 0:
            time.sleep(0.1)
            self._update_model_list()
        return self.model_list[(self.n + self.capacity - 1) % self.capacity]
        
    def load_model(self, metadata):
        self._update_model_list()
        n = metadata['id']
        if n < self.n - self.capacity: return None
        try:
            memory = SharedMemory(name = metadata['_addr'])
            state_dict = cPickle.loads(memory.buf)
            memory.close()
            # 从 resource_tracker 注销，防止进程退出时自动 unlink（由 Server 统一管理）
            _unregister_from_resource_tracker(metadata['_addr'])
            return state_dict
        except FileNotFoundError:
            # 共享内存已被 Server 释放（模型过旧）
            return None
