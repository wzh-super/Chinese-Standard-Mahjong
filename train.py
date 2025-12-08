from replay_buffer import ReplayBuffer
from actor import Actor
from learner import Learner
import argparse

if __name__ == '__main__':
    # 命令行参数
    parser = argparse.ArgumentParser(description='RL training for Mahjong')
    parser.add_argument('--pretrain', type=str, default=None, help='Path to pretrained model from supervised learning')
    args = parser.parse_args()

    # 硬件配置: RTX 4090 (24GB) + 25核 CPU + 90GB 内存
    config = {
        # === 经验收集 ===
        'replay_buffer_size': 200000,     # 90GB内存，可以存更多经验
        'replay_buffer_episode': 1000,    # 队列容量
        'model_pool_size': 50,            # 增大缓冲，避免历史模型被过快释放
        'model_pool_name': 'model-pool',
        'num_actors': 24,                 # 25核留1核给learner
        'episodes_per_actor': 20000,      # 每个actor跑更多局（只收集1个玩家数据，需要4倍补偿）

        # === PPO 参数 ===
        'gamma': 0.99,                    # 折扣因子，稍微提高
        'lambda': 0.95,                   # GAE参数
        'min_sample': 5000,               # 开始训练前的最小样本数（提高初始多样性）
        'batch_size': 1024,               # 4090可以开大batch
        'epochs': 5,                      # 每批数据的PPO迭代次数
        'clip': 0.2,                      # PPO裁剪范围
        'lr': 3e-5,                       # 学习率（降低，保护预训练知识）
        'lr_min': 1e-5,                   # 学习率下限
        'lr_decay_steps': 5000,           # 每隔多少步衰减
        'lr_decay_rate': 0.8,             # 衰减系数
        'value_coeff': 0.5,               # 价值损失系数
        'entropy_coeff': 0.1,             # 熵正则系数（大幅增加，强制探索）

        # === 保存 ===
        'device': 'cuda',
        'ckpt_save_interval': 300,        # 5分钟保存一次
        'ckpt_save_path': './checkpoint/',

        # === 预训练模型 ===
        'pretrain_path': args.pretrain,
    }
    
    replay_buffer = ReplayBuffer(config['replay_buffer_size'], config['replay_buffer_episode'])
    
    actors = []
    for i in range(config['num_actors']):
        config['name'] = 'Actor-%d' % i
        actor = Actor(config, replay_buffer)
        actors.append(actor)
    learner = Learner(config, replay_buffer)
    
    for actor in actors: actor.start()
    learner.start()
    
    for actor in actors: actor.join()
    learner.terminate()