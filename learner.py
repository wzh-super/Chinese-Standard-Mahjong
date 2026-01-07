from multiprocessing import Process
import os
import time
from datetime import datetime
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from replay_buffer import ReplayBuffer
from model_pool import ModelPoolServer
from model import CNNModel, CentralizedCritic


def compute_gae_from_trajectories(batch, values, gamma, lam):
    """
    根据轨迹信息计算 GAE (Generalized Advantage Estimation)

    Args:
        batch: dict 包含 reward, done, traj_id, t
        values: numpy array, 每个样本的 V(s) 估计
        gamma: 折扣因子
        lam: GAE lambda 参数

    Returns:
        advantages: numpy array
        targets: numpy array (V-target for critic training)
    """
    n_samples = len(values)
    advantages = np.zeros(n_samples, dtype=np.float32)
    targets = np.zeros(n_samples, dtype=np.float32)

    rewards = batch['reward']
    dones = batch['done']
    traj_ids = batch['traj_id']
    t_steps = batch['t']

    # 按 traj_id 分组
    unique_trajs = np.unique(traj_ids)

    for traj_id in unique_trajs:
        # 找到该轨迹的所有样本
        traj_mask = traj_ids == traj_id
        traj_indices = np.where(traj_mask)[0]

        # 按时间步排序
        traj_t = t_steps[traj_indices]
        sort_order = np.argsort(traj_t)
        sorted_indices = traj_indices[sort_order]

        # 获取该轨迹的数据
        traj_rewards = rewards[sorted_indices]
        traj_dones = dones[sorted_indices]
        traj_values = values[sorted_indices]

        T = len(sorted_indices)
        traj_advs = np.zeros(T, dtype=np.float32)
        traj_targets = np.zeros(T, dtype=np.float32)

        # 从后往前计算 GAE
        gae = 0.0
        for i in reversed(range(T)):
            # 显式使用 (1 - done) 作为 terminal mask，更稳健
            not_done = 1.0 - float(traj_dones[i])
            next_value = traj_values[i + 1] if i + 1 < T else 0.0

            # delta = r + gamma * (1-done) * V_next - V
            delta = traj_rewards[i] + gamma * not_done * next_value - traj_values[i]
            # GAE: A_t = delta_t + gamma * lambda * (1-done) * A_{t+1}
            gae = delta + gamma * lam * not_done * gae

            traj_advs[i] = gae
            traj_targets[i] = traj_advs[i] + traj_values[i]  # target = adv + V

        # 写回原数组
        advantages[sorted_indices] = traj_advs
        targets[sorted_indices] = traj_targets

    return advantages, targets


class Learner(Process):
    
    def __init__(self, config, replay_buffer):
        super(Learner, self).__init__()
        self.replay_buffer = replay_buffer
        self.config = config
    
    def run(self):
        # 使用config中预先生成的实验名称和检查点路径
        exp_name = self.config.get('exp_name', datetime.now().strftime('%Y%m%d_%H%M%S'))
        ckpt_path = self.config.get('ckpt_save_path', './checkpoint/')

        # create checkpoint directory if not exists
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)

        # create tensorboard writer with timestamp
        writer = SummaryWriter(f'./runs/{exp_name}')

        # create model pool
        model_pool = ModelPoolServer(self.config['model_pool_size'], self.config['model_pool_name'])

        # initialize model params
        device = torch.device(self.config['device'])
        model = CNNModel()

        # 加载模型：优先使用 resume，其次使用 pretrain
        pretrain_model = None
        resume_path = self.config.get('resume_model_path')
        pretrain_path = self.config.get('pretrain_path')

        if resume_path and os.path.exists(resume_path):
            # 继续训练：从检查点恢复
            print(f"Resuming model from: {resume_path}")
            model.load_state_dict(torch.load(resume_path, map_location='cpu'))
            print("Model resumed successfully!")
            # 继续训练时，如果有 pretrain 则用于 KL 约束，否则不使用 KL 约束
            if pretrain_path and os.path.exists(pretrain_path):
                pretrain_model = CNNModel()
                pretrain_model.load_state_dict(torch.load(pretrain_path, map_location='cpu'))
                pretrain_model = pretrain_model.to(device)
                pretrain_model.eval()
                for param in pretrain_model.parameters():
                    param.requires_grad = False
        elif pretrain_path and os.path.exists(pretrain_path):
            # 新训练：使用预训练模型
            print(f"Loading pretrained model: {pretrain_path}")
            model.load_state_dict(torch.load(pretrain_path, map_location='cpu'))
            print("Pretrained model loaded successfully!")
            # 保存一份预训练模型用于KL约束
            pretrain_model = CNNModel()
            pretrain_model.load_state_dict(torch.load(pretrain_path, map_location='cpu'))
            pretrain_model = pretrain_model.to(device)
            pretrain_model.eval()
            for param in pretrain_model.parameters():
                param.requires_grad = False

        # send to model pool
        model_pool.push(model.state_dict()) # push cpu-only tensor to model_pool
        current_policy_id = 0  # 初始模型版本
        model = model.to(device)

        # CTDE: 创建独立的集中式 Critic
        critic = CentralizedCritic()

        # 加载 Critic checkpoint（如果有）
        resume_critic_path = self.config.get('resume_critic_path')
        if resume_critic_path and os.path.exists(resume_critic_path):
            print(f"Resuming critic from: {resume_critic_path}")
            critic.load_state_dict(torch.load(resume_critic_path, map_location='cpu'))
            print("Critic resumed successfully!")

        critic = critic.to(device)

        # training - 分别为 Actor 和 Critic 创建优化器
        actor_optimizer = torch.optim.Adam(model.parameters(), lr=self.config['lr'])
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=self.config['lr'])

        # 加载 optimizer state（如果有）
        resume_iteration = self.config.get('resume_iteration', 0)
        if resume_iteration > 0:
            # 尝试加载 optimizer state
            optimizer_path = os.path.join(ckpt_path, f'optimizer_{resume_iteration - 1}.pt')
            if os.path.exists(optimizer_path):
                print(f"Resuming optimizers from: {optimizer_path}")
                opt_state = torch.load(optimizer_path, map_location='cpu')
                actor_optimizer.load_state_dict(opt_state['actor_optimizer'])
                critic_optimizer.load_state_dict(opt_state['critic_optimizer'])
                print("Optimizers resumed successfully!")
            else:
                print(f"Warning: Optimizer state {optimizer_path} not found, using fresh optimizers")

        # 学习率调度：带最小值限制的指数衰减
        lr_min = self.config.get('lr_min', 1e-5)  # 学习率下限
        lr_decay_steps = self.config.get('lr_decay_steps', 5000)  # 每隔多少步衰减
        lr_decay_rate = self.config.get('lr_decay_rate', 0.8)  # 衰减系数

        # wait for initial samples
        while self.replay_buffer.size() < self.config['min_sample']:
            time.sleep(0.1)

        # CTDE 说明：Critic warmup 已移除
        # 由于 Actor 不再计算 target，warmup 需要先用 critic 前向算 V
        # 简化起见，让主训练循环直接训练 critic
        # 如果需要 warmup，可以在收集首批数据后做几轮纯 critic 更新

        cur_time = time.time()
        iterations = resume_iteration  # 从 resume_iteration 开始，避免覆盖旧 checkpoint
        samples_per_update = self.config.get('samples_per_update', 10000)  # 每轮收集的样本数
        max_staleness = self.config.get('max_staleness', 3)  # 最大允许的策略版本差

        while True:
            # ==================== On-Policy PPO: 等待收集足够样本 ====================
            while self.replay_buffer.size() < samples_per_update:
                time.sleep(0.1)

            # 原子操作：取出所有样本并清空 buffer，避免丢失期间 push 的数据
            batch = self.replay_buffer.get_all_and_clear()

            # ==================== Policy ID 过滤：丢弃太旧的样本 ====================
            policy_ids = batch['policy_id']
            fresh_mask = policy_ids >= (current_policy_id - max_staleness)
            n_fresh = fresh_mask.sum()
            n_stale = len(policy_ids) - n_fresh

            if n_fresh == 0:
                print(f'Warning: All {len(policy_ids)} samples are stale (policy_id < {current_policy_id - max_staleness}), skipping iteration')
                continue

            if n_stale > 0:
                print(f'Filtered out {n_stale} stale samples (policy_id < {current_policy_id - max_staleness})')
                # 过滤所有数据（新格式：reward, done, traj_id, t）
                batch = {
                    'state': {
                        'observation': batch['state']['observation'][fresh_mask],
                        'action_mask': batch['state']['action_mask'][fresh_mask],
                        'other_hands': batch['state']['other_hands'][fresh_mask]  # CTDE: 12×4×9
                    },
                    'action': batch['action'][fresh_mask],
                    'log_prob': batch['log_prob'][fresh_mask],
                    'reward': batch['reward'][fresh_mask],
                    'done': batch['done'][fresh_mask],
                    'traj_id': batch['traj_id'][fresh_mask],
                    't': batch['t'][fresh_mask],
                    'policy_id': batch['policy_id'][fresh_mask]
                }

            total_samples = len(batch['action'])

            # ==================== CTDE: 用 Critic 计算 V，然后算 GAE ====================
            obs = torch.tensor(batch['state']['observation']).to(device)
            mask = torch.tensor(batch['state']['action_mask']).to(device)
            other_hands = torch.tensor(batch['state']['other_hands']).to(device)  # CTDE: 12×4×9
            # 拼接成全局观测供 Critic 使用
            global_obs = torch.cat([obs, other_hands], dim=1)  # 147 + 12 = 159

            # 用 Critic 前向计算所有样本的 V(s)
            critic.eval()
            with torch.no_grad():
                values_tensor = critic(global_obs).squeeze(-1)
                values_np = values_tensor.cpu().numpy()

            # 用轨迹信息计算 GAE
            advantages_np, targets_np = compute_gae_from_trajectories(
                batch, values_np, self.config['gamma'], self.config['lambda']
            )

            # 转为 tensor
            states = {
                'observation': obs,
                'action_mask': mask
            }
            actions = torch.tensor(batch['action']).unsqueeze(-1).to(device)
            old_log_probs = torch.tensor(batch['log_prob']).unsqueeze(-1).to(device)
            advs = torch.tensor(advantages_np).to(device)
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)  # Advantage归一化
            targets = torch.tensor(targets_np).to(device)

            print('Iteration %d, samples collected %d, total episodes %d, unique trajectories %d' % (
                iterations, total_samples, self.replay_buffer.stats['episode_in'],
                len(np.unique(batch['traj_id']))))

            # ==================== PPO 训练：多个 epoch，每个 epoch 遍历所有 mini-batch ====================
            batch_size = self.config['batch_size']
            indices = np.arange(total_samples)

            for epoch in range(self.config['epochs']):
                np.random.shuffle(indices)  # 每个 epoch 打乱顺序

                for start in range(0, total_samples, batch_size):
                    end = min(start + batch_size, total_samples)
                    mb_indices = indices[start:end]

                    mb_states = {
                        'observation': obs[mb_indices],
                        'action_mask': mask[mb_indices]
                    }
                    mb_global_obs = global_obs[mb_indices]  # CTDE
                    mb_actions = actions[mb_indices]
                    mb_advs = advs[mb_indices]
                    mb_targets = targets[mb_indices]
                    mb_old_log_probs = old_log_probs[mb_indices]

                    # ========== Actor 前向传播 ==========
                    model.train(True)
                    logits, _ = model(mb_states)  # 不使用 model 的 value 输出
                    action_dist = torch.distributions.Categorical(logits=logits)
                    probs = F.softmax(logits, dim=1).gather(1, mb_actions)
                    log_probs = torch.log(probs + 1e-8)

                    # ========== CTDE Critic 前向传播 ==========
                    critic.train(True)
                    values = critic(mb_global_obs)

                    # PPO ratio 使用真正的 old_log_probs（actor 采样时的策略）
                    ratio = torch.exp(log_probs - mb_old_log_probs)
                    surr1 = ratio * mb_advs.unsqueeze(-1)
                    surr2 = torch.clamp(ratio, 1 - self.config['clip'], 1 + self.config['clip']) * mb_advs.unsqueeze(-1)
                    policy_loss = -torch.mean(torch.min(surr1, surr2))
                    value_loss = F.mse_loss(values.squeeze(-1), mb_targets)
                    entropy_loss = -torch.mean(action_dist.entropy())

                    # 动态调整 KL 约束
                    kl_coeff_init = self.config.get('kl_coeff_init', 0.02)
                    kl_coeff_min = self.config.get('kl_coeff_min', 0.0)
                    kl_decay_steps = self.config.get('kl_decay_steps', 10000)

                    if iterations < kl_decay_steps:
                        kl_coeff = kl_coeff_init - (kl_coeff_init - kl_coeff_min) * (iterations / kl_decay_steps)
                    else:
                        kl_coeff = kl_coeff_min

                    entropy_coeff = self.config.get('entropy_coeff', 0.01)

                    # KL散度约束
                    kl_loss = torch.tensor(0.0).to(device)
                    if pretrain_model is not None and kl_coeff > 0:
                        with torch.no_grad():
                            pretrain_logits, _ = pretrain_model(mb_states)
                        pretrain_probs = F.softmax(pretrain_logits, dim=1)
                        current_probs = F.softmax(logits, dim=1)
                        kl_loss = torch.mean(torch.sum(pretrain_probs * (torch.log(pretrain_probs + 1e-8) - torch.log(current_probs + 1e-8)), dim=1))

                    # ========== 分别更新 Actor 和 Critic ==========
                    # Actor loss: policy + entropy + KL
                    actor_loss = policy_loss + entropy_coeff * entropy_loss + kl_coeff * kl_loss
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    actor_optimizer.step()

                    # Critic loss: value
                    critic_loss = self.config['value_coeff'] * value_loss
                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
                    critic_optimizer.step()

            # push new model（每轮更新完 push 一次）
            model = model.to('cpu')
            model_pool.push(model.state_dict())
            current_policy_id += 1  # 更新当前策略版本
            model = model.to(device)

            # log to tensorboard
            current_lr = actor_optimizer.param_groups[0]['lr']
            total_episodes = self.replay_buffer.stats['episode_in']
            writer.add_scalar('Loss/policy', policy_loss.item(), iterations)
            writer.add_scalar('Loss/value', value_loss.item(), iterations)
            writer.add_scalar('Loss/entropy', entropy_loss.item(), iterations)
            writer.add_scalar('Loss/kl', kl_loss.item(), iterations)
            writer.add_scalar('Loss/actor_total', actor_loss.item(), iterations)
            writer.add_scalar('Loss/critic_total', critic_loss.item(), iterations)
            writer.add_scalar('Stats/advantage_mean', advs.mean().item(), iterations)
            writer.add_scalar('Stats/advantage_std', advs.std().item(), iterations)
            writer.add_scalar('Stats/value_mean', values.mean().item(), iterations)
            writer.add_scalar('Stats/samples_per_update', total_samples, iterations)
            writer.add_scalar('Stats/learning_rate', current_lr, iterations)
            writer.add_scalar('Params/kl_coeff', kl_coeff, iterations)
            writer.add_scalar('Params/entropy_coeff', entropy_coeff, iterations)
            writer.add_scalar('Stats/total_episodes', total_episodes, iterations)
            # 记录实际episode reward统计
            if hasattr(self.replay_buffer, 'reward_stats') and self.replay_buffer.reward_stats['count'] > 0:
                avg_reward = self.replay_buffer.reward_stats['sum'] / self.replay_buffer.reward_stats['count']
                writer.add_scalar('Stats/episode_reward_avg', avg_reward, iterations)
                if len(self.replay_buffer.reward_stats['recent']) > 0:
                    recent_avg = sum(self.replay_buffer.reward_stats['recent']) / len(self.replay_buffer.reward_stats['recent'])
                    writer.add_scalar('Stats/episode_reward_recent', recent_avg, iterations)

            # 学习率衰减（同时更新 Actor 和 Critic）
            if (iterations + 1) % lr_decay_steps == 0 and current_lr > lr_min:
                new_lr = max(current_lr * lr_decay_rate, lr_min)
                for param_group in actor_optimizer.param_groups:
                    param_group['lr'] = new_lr
                for param_group in critic_optimizer.param_groups:
                    param_group['lr'] = new_lr
                print(f'Learning rate decayed: {current_lr:.2e} -> {new_lr:.2e}')

            # save checkpoints（同时保存 Actor、Critic 和 Optimizer）
            t = time.time()
            if t - cur_time > self.config['ckpt_save_interval']:
                actor_path = os.path.join(ckpt_path, 'model_%d.pt' % iterations)
                critic_path = os.path.join(ckpt_path, 'critic_%d.pt' % iterations)
                optimizer_path = os.path.join(ckpt_path, 'optimizer_%d.pt' % iterations)
                torch.save(model.state_dict(), actor_path)
                torch.save(critic.state_dict(), critic_path)
                torch.save({
                    'actor_optimizer': actor_optimizer.state_dict(),
                    'critic_optimizer': critic_optimizer.state_dict(),
                }, optimizer_path)
                print(f'Checkpoint saved: iteration {iterations}')
                cur_time = t
            iterations += 1