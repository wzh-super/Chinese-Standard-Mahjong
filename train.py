from replay_buffer import ReplayBuffer
from actor import Actor
from learner import Learner
import argparse
from datetime import datetime
import os
import glob

if __name__ == '__main__':
    # 命令行参数
    parser = argparse.ArgumentParser(description='RL training for Mahjong')
    parser.add_argument('--pretrain', type=str, default=None, help='Path to pretrained model from supervised learning')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint directory to resume training')
    args = parser.parse_args()

    # 确定实验名称和检查点路径
    if args.resume:
        # 继续训练：使用已有的 checkpoint 目录
        ckpt_save_path = args.resume.rstrip('/\\')
        exp_name = os.path.basename(ckpt_save_path)

        # 找到目录中最新的检查点
        ckpt_files = glob.glob(os.path.join(ckpt_save_path, '*.pt'))
        if ckpt_files:
            # 按修改时间排序，取最新的
            latest_ckpt = max(ckpt_files, key=os.path.getmtime)
            resume_model_path = latest_ckpt
            print(f"Resuming from checkpoint: {latest_ckpt}")
        else:
            print(f"Warning: No checkpoint files found in {ckpt_save_path}")
            resume_model_path = None
    else:
        # 新训练：生成新的实验名称
        exp_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        ckpt_save_path = os.path.join('./checkpoint/', exp_name)
        resume_model_path = None

    # 硬件配置: RTX 4090 (24GB) + 25核 CPU + 90GB 内存
    config = {
        # === 经验收集 ===
        'replay_buffer_size': 200000,     # 90GB内存，可以存更多经验
        'replay_buffer_episode': 1000,    # 队列容量
        'model_pool_size': 50,            # 增大缓冲，避免历史模型被过快释放
        'model_pool_name': 'model-pool',
        'num_actors': 24,                 # 25核留1核给learner
        'episodes_per_actor': 100000,    # 每个actor跑的局数（足够多，可手动停止）

        # === PPO 参数 ===
        'gamma': 0.99,                    # 折扣因子，稍微提高
        'lambda': 0.95,                   # GAE参数
        'min_sample': 5000,               # 开始训练前的最小样本数（提高初始多样性）
        'value_warmup_steps': 2000,       # Value预热步数（冻结Policy，只训练Value）
        'batch_size': 1024,               # 4090可以开大batch
        'epochs': 5,                      # 每批数据的PPO迭代次数
        'clip': 0.15,                     # PPO裁剪范围（稍降低）
        'lr': 5e-5,                       # 学习率（适中）
        'lr_min': 1e-5,                   # 学习率下限
        'lr_decay_steps': 5000,           # 每隔多少步衰减
        'lr_decay_rate': 0.8,             # 衰减系数
        'value_coeff': 0.5,               # 价值损失系数
        'entropy_coeff': 0.03,            # 熵正则系数（降低）
        'kl_coeff': 0.05,                 # KL约束（保留但减半）

        # === 保存 ===
        'device': 'cuda',
        'ckpt_save_interval': 300,        # 5分钟保存一次
        'ckpt_save_path': ckpt_save_path, # 本次运行的检查点目录
        'exp_name': exp_name,             # 实验名称

        # === 预训练模型 ===
        'pretrain_path': args.pretrain,
        'resume_model_path': resume_model_path,  # 继续训练时的检查点路径
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