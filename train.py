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
    parser.add_argument('--self-play', action='store_true', default=True, help='Enable self-play mode (default: enabled)')
    parser.add_argument('--no-self-play', action='store_false', dest='self_play', help='Disable self-play mode')
    parser.add_argument('--pretrain-prob', type=float, default=0.3, help='Probability of having one pretrain opponent per episode (only in self-play mode)')
    parser.add_argument('--kl-init', type=float, default=0.0, help='Initial KL coefficient')
    parser.add_argument('--kl-min', type=float, default=0.0, help='Minimum KL coefficient')
    parser.add_argument('--kl-decay-steps', type=int, default=10000, help='Steps to decay KL coefficient')
    parser.add_argument('--reward-mode', type=str, default='original', choices=['simple', 'sqrt', 'original'],
                        help='Reward mode: simple(统一惩罚), sqrt(压缩但保留差异), original(原始)')
    args = parser.parse_args()

    # 确定实验名称和检查点路径
    if args.resume:
        # 继续训练：使用已有的 checkpoint 目录
        ckpt_save_path = args.resume.rstrip('/\\')
        exp_name = os.path.basename(ckpt_save_path)

        # 找到目录中最新的 model 检查点，按文件名中的数字排序
        ckpt_files = glob.glob(os.path.join(ckpt_save_path, 'model_*.pt'))
        if ckpt_files:
            # 从文件名解析 iteration 数字，选最大的
            def get_iteration(path):
                basename = os.path.basename(path)
                # model_123.pt -> 123
                try:
                    return int(basename.replace('model_', '').replace('.pt', ''))
                except ValueError:
                    return -1

            latest_iter = max(get_iteration(f) for f in ckpt_files)
            resume_model_path = os.path.join(ckpt_save_path, f'model_{latest_iter}.pt')
            resume_critic_path = os.path.join(ckpt_save_path, f'critic_{latest_iter}.pt')
            resume_iteration = latest_iter + 1  # 从下一个 iteration 开始

            if os.path.exists(resume_model_path):
                print(f"Resuming actor from checkpoint: {resume_model_path}")
            else:
                print(f"Warning: {resume_model_path} not found")
                resume_model_path = None

            if os.path.exists(resume_critic_path):
                print(f"Resuming critic from checkpoint: {resume_critic_path}")
            else:
                print(f"Warning: {resume_critic_path} not found")
                resume_critic_path = None

            print(f"Resuming from iteration: {resume_iteration}")
        else:
            print(f"Warning: No model checkpoint files found in {ckpt_save_path}")
            resume_model_path = None
            resume_critic_path = None
            resume_iteration = 0
    else:
        # 新训练：生成新的实验名称
        exp_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        ckpt_save_path = os.path.join('./checkpoint/', exp_name)
        resume_model_path = None
        resume_critic_path = None
        resume_iteration = 0

    # 硬件配置: RTX 4090 (24GB) + 25核 CPU + 90GB 内存
    config = {
        # === 经验收集 ===
        'replay_buffer_size': 50000,      # On-policy: 每轮清空，不需要太大
        'replay_buffer_episode': 500,     # 队列容量
        'model_pool_size': 50,            # 增大缓冲，避免历史模型被过快释放
        'model_pool_name': 'model-pool',
        'num_actors': 24,                 # 25核留1核给learner
        'episodes_per_actor': 10000,    # 每个actor跑的局数（足够多，可手动停止）

        # === PPO 参数（On-Policy）===
        'samples_per_update': 20000,      # 每轮收集的样本数（约 200-400 局）
        'max_staleness': 3,               # 最大允许的策略版本差（过滤太旧的样本）
        'gamma': 0.99,                    # 折扣因子
        'lambda': 0.95,                   # GAE参数
        'min_sample': 5000,               # 开始训练前的最小样本数
        'value_warmup_steps': 2000,       # Value预热步数
        'batch_size': 1024,               # mini-batch 大小
        'epochs': 5,                      # 每批数据的PPO迭代次数
        'clip': 0.15,                     # PPO裁剪范围
        'lr': 5e-5,                       # 学习率
        'lr_min': 1e-5,                   # 学习率下限
        'lr_decay_steps': 5000,           # 每隔多少步衰减
        'lr_decay_rate': 0.8,             # 衰减系数
        'value_coeff': 0.5,               # 价值损失系数
        'entropy_coeff': 0.01,            # 熵正则系数

        # === KL约束（动态衰减） ===
        'kl_coeff_init': args.kl_init,    # 初始KL系数
        'kl_coeff_min': args.kl_min,      # 最小KL系数
        'kl_decay_steps': args.kl_decay_steps,  # 衰减步数

        # === 自博弈模式 ===
        'self_play_mode': args.self_play,  # 是否开启自博弈模式（默认开启）
        'pretrain_prob': args.pretrain_prob,  # 每局有预训练对手的概率

        # === 奖励模式 ===
        'reward_mode': args.reward_mode,  # simple/sqrt/original

        # === 保存 ===
        'device': 'cuda',
        'ckpt_save_interval': 300,        # 5分钟保存一次
        'ckpt_save_path': ckpt_save_path, # 本次运行的检查点目录
        'exp_name': exp_name,             # 实验名称

        # === 预训练模型 ===
        'pretrain_path': args.pretrain,
        'resume_model_path': resume_model_path,  # 继续训练时的 Actor 检查点路径
        'resume_critic_path': resume_critic_path,  # 继续训练时的 Critic 检查点路径
        'resume_iteration': resume_iteration,  # 继续训练时的起始 iteration
    }
    
    replay_buffer = ReplayBuffer(config['replay_buffer_size'], config['replay_buffer_episode'])

    # 打印训练模式信息
    print("=" * 50)
    print("Training Configuration:")
    print(f"  Self-play mode: {config['self_play_mode']}")
    if config['self_play_mode']:
        print(f"  Pretrain opponent probability: {config['pretrain_prob']}")
    print(f"  Reward mode: {config['reward_mode']}")
    print(f"  KL coefficient: {config['kl_coeff_init']} -> {config['kl_coeff_min']} over {config['kl_decay_steps']} steps")
    print(f"  Pretrain model: {config['pretrain_path']}")
    print(f"  Checkpoint path: {config['ckpt_save_path']}")
    print("=" * 50)

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