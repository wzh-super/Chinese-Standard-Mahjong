from multiprocessing import Queue
from collections import deque
from queue import Empty
import numpy as np
import random

class ReplayBuffer:

    def __init__(self, capacity, episode):
        self.queue = Queue(episode)
        self.capacity = capacity
        self.buffer = None

    def push(self, samples): # only called by actors
        self.queue.put(samples)

    def _flush(self):
        if self.buffer is None: # called first time by learner
            self.buffer = deque(maxlen = self.capacity)
            self.stats = {'sample_in': 0, 'sample_out': 0, 'episode_in': 0}
            self.reward_stats = {'sum': 0.0, 'count': 0, 'recent': deque(maxlen=1000)}
        # 用 get_nowait + Empty 异常，比 empty() 在多进程下更可靠
        while True:
            try:
                episode_data = self.queue.get_nowait()
            except Empty:
                break
            # 提取并统计episode_reward
            if 'episode_reward' in episode_data:
                ep_reward = episode_data.pop('episode_reward')
                self.reward_stats['sum'] += ep_reward
                self.reward_stats['count'] += 1
                self.reward_stats['recent'].append(ep_reward)
            unpacked_data = self._unpack(episode_data)
            self.buffer.extend(unpacked_data)
            self.stats['sample_in'] += len(unpacked_data)
            self.stats['episode_in'] += 1
    
    def sample(self, batch_size): # only called by learner
        self._flush()
        assert len(self.buffer) > 0, "Empty buffer!"
        self.stats['sample_out'] += batch_size
        if batch_size >= len(self.buffer):
            samples = self.buffer
        else:
            samples = random.sample(self.buffer, batch_size)
        batch = self._pack(samples)
        return batch

    def get_all(self): # only called by learner - 取出所有样本用于 on-policy PPO
        self._flush()
        assert len(self.buffer) > 0, "Empty buffer!"
        samples = list(self.buffer)
        self.stats['sample_out'] += len(samples)
        batch = self._pack(samples)
        return batch

    def get_all_and_clear(self): # only called by learner - 原子操作：取出所有样本并清空
        """原子操作：取出所有样本并清空 buffer，避免丢失期间 push 的数据"""
        self._flush()
        assert len(self.buffer) > 0, "Empty buffer!"
        samples = list(self.buffer)
        self.stats['sample_out'] += len(samples)
        self.buffer.clear()  # 立即清空，不再调用 _flush
        batch = self._pack(samples)
        return batch

    def size(self): # only called by learner
        self._flush()
        return len(self.buffer)

    def clear(self): # only called by learner - 只清空 buffer，不 flush 队列
        if self.buffer is not None:
            self.buffer.clear()
    
    def _unpack(self, data):
        # convert dict (of dict...) of list of (num/ndarray/list) to list of dict (of dict...)
        if type(data) == dict:
            res = []
            for key, value in data.items():
                values = self._unpack(value)
                if not res: res = [{} for i in range(len(values))]
                for i, v in enumerate(values):
                    res[i][key] = v
            return res
        else:
            return list(data)
            
    def _pack(self, data):
        # convert list of dict (of dict...) to dict (of dict...) of numpy array
        if type(data[0]) == dict:
            keys = data[0].keys()
            res = {}
            for key in keys:
                values = [x[key] for x in data]
                res[key] = self._pack(values)
            return res
        elif type(data[0]) == np.ndarray:
            return np.stack(data)
        else:
            return np.array(data)