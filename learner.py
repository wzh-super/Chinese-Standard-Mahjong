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
from model import CNNModel

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
        model = model.to(device)
        
        # training
        optimizer = torch.optim.Adam(model.parameters(), lr = self.config['lr'])

        # 学习率调度：带最小值限制的指数衰减
        lr_min = self.config.get('lr_min', 1e-5)  # 学习率下限
        lr_decay_steps = self.config.get('lr_decay_steps', 5000)  # 每隔多少步衰减
        lr_decay_rate = self.config.get('lr_decay_rate', 0.8)  # 衰减系数
        
        # wait for initial samples
        while self.replay_buffer.size() < self.config['min_sample']:
            time.sleep(0.1)

        # === Value预热阶段：冻结Policy，只训练Value ===
        # 继续训练时跳过 warmup
        value_warmup_steps = self.config.get('value_warmup_steps', 1000)
        if resume_path:
            value_warmup_steps = 0
            print("Skipping value warmup (resuming from checkpoint)")
        if value_warmup_steps > 0:
            print(f"Starting value warmup for {value_warmup_steps} steps...")
            # 冻结backbone和policy头
            for name, param in model.named_parameters():
                if '_value_branch' not in name:
                    param.requires_grad = False

            value_optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=self.config['lr'] * 10  # value学习率可以更高
            )

            for warmup_step in range(value_warmup_steps):
                batch = self.replay_buffer.sample(self.config['batch_size'])
                obs = torch.tensor(batch['state']['observation']).to(device)
                mask = torch.tensor(batch['state']['action_mask']).to(device)
                states = {'observation': obs, 'action_mask': mask}
                targets = torch.tensor(batch['target']).to(device)

                model.train(True)
                _, values = model(states)
                value_loss = F.mse_loss(values.squeeze(-1), targets)

                value_optimizer.zero_grad()
                value_loss.backward()
                value_optimizer.step()

                if (warmup_step + 1) % 100 == 0:
                    print(f"Value warmup step {warmup_step + 1}/{value_warmup_steps}, loss: {value_loss.item():.4f}")

            # 解冻所有参数
            for param in model.parameters():
                param.requires_grad = True
            print("Value warmup completed!")

        cur_time = time.time()
        iterations = 0
        while True:
            # sample batch
            batch = self.replay_buffer.sample(self.config['batch_size'])
            obs = torch.tensor(batch['state']['observation']).to(device)
            mask = torch.tensor(batch['state']['action_mask']).to(device)
            states = {
                'observation': obs,
                'action_mask': mask
            }
            actions = torch.tensor(batch['action']).unsqueeze(-1).to(device)
            advs = torch.tensor(batch['adv']).to(device)
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)  # Advantage归一化，稳定训练
            targets = torch.tensor(batch['target']).to(device)
            
            print('Iteration %d, replay buffer in %d out %d' % (iterations, self.replay_buffer.stats['sample_in'], self.replay_buffer.stats['sample_out']))
            
            # calculate PPO loss
            model.train(True) # Batch Norm training mode
            old_logits, _ = model(states)
            old_probs = F.softmax(old_logits, dim = 1).gather(1, actions)
            old_log_probs = torch.log(old_probs + 1e-8).detach()
            for _ in range(self.config['epochs']):
                logits, values = model(states)
                action_dist = torch.distributions.Categorical(logits = logits)
                probs = F.softmax(logits, dim = 1).gather(1, actions)
                log_probs = torch.log(probs + 1e-8)
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advs
                surr2 = torch.clamp(ratio, 1 - self.config['clip'], 1 + self.config['clip']) * advs
                policy_loss = -torch.mean(torch.min(surr1, surr2))
                value_loss = torch.mean(F.mse_loss(values.squeeze(-1), targets))
                entropy_loss = -torch.mean(action_dist.entropy())

                # 动态调整 KL 约束（平滑衰减）
                # kl_coeff 从 kl_coeff_init 开始，经过 kl_decay_steps 步衰减到 kl_coeff_min
                kl_coeff_init = self.config.get('kl_coeff_init', 0.02)  # 初始KL系数
                kl_coeff_min = self.config.get('kl_coeff_min', 0.0)     # 最小KL系数
                kl_decay_steps = self.config.get('kl_decay_steps', 10000)  # 衰减步数

                if iterations < kl_decay_steps:
                    # 线性衰减
                    kl_coeff = kl_coeff_init - (kl_coeff_init - kl_coeff_min) * (iterations / kl_decay_steps)
                else:
                    kl_coeff = kl_coeff_min

                # 熵系数（可选：也可以动态调整）
                entropy_coeff = self.config.get('entropy_coeff', 0.01)

                # KL散度约束：限制策略不能偏离预训练太远
                kl_loss = torch.tensor(0.0).to(device)
                if pretrain_model is not None and kl_coeff > 0:
                    with torch.no_grad():
                        pretrain_logits, _ = pretrain_model(states)
                    pretrain_probs = F.softmax(pretrain_logits, dim=1)
                    current_probs = F.softmax(logits, dim=1)
                    # KL(pretrain || current)
                    kl_loss = torch.mean(torch.sum(pretrain_probs * (torch.log(pretrain_probs + 1e-8) - torch.log(current_probs + 1e-8)), dim=1))

                loss = policy_loss + self.config['value_coeff'] * value_loss + entropy_coeff * entropy_loss + kl_coeff * kl_loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # 梯度裁剪，防止梯度爆炸
                optimizer.step()

            # push new model
            model = model.to('cpu')
            model_pool.push(model.state_dict()) # push cpu-only tensor to model_pool
            model = model.to(device)

            # log to tensorboard
            current_lr = optimizer.param_groups[0]['lr']
            total_episodes = self.replay_buffer.stats['episode_in']
            writer.add_scalar('Loss/policy', policy_loss.item(), iterations)
            writer.add_scalar('Loss/value', value_loss.item(), iterations)
            writer.add_scalar('Loss/entropy', entropy_loss.item(), iterations)
            writer.add_scalar('Loss/kl', kl_loss.item(), iterations)
            writer.add_scalar('Loss/total', loss.item(), iterations)
            writer.add_scalar('Stats/advantage_mean', advs.mean().item(), iterations)
            writer.add_scalar('Stats/advantage_std', advs.std().item(), iterations)
            writer.add_scalar('Stats/value_mean', values.mean().item(), iterations)
            writer.add_scalar('Stats/buffer_size', self.replay_buffer.stats['sample_in'], iterations)
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

            # 学习率衰减（带下限）
            if (iterations + 1) % lr_decay_steps == 0 and current_lr > lr_min:
                new_lr = max(current_lr * lr_decay_rate, lr_min)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                print(f'Learning rate decayed: {current_lr:.2e} -> {new_lr:.2e}')

            # save checkpoints
            t = time.time()
            if t - cur_time > self.config['ckpt_save_interval']:
                path = os.path.join(ckpt_path, 'model_%d.pt' % iterations)
                torch.save(model.state_dict(), path)
                cur_time = t
            iterations += 1