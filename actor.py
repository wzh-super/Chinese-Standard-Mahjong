from multiprocessing import Process
import numpy as np
import torch
import random
import os
import glob

from replay_buffer import ReplayBuffer
from model_pool import ModelPoolClient
from env import MahjongGBEnv
from feature_v2 import FeatureAgentV2
from model import CNNModel


# 对手类型常量
OPPONENT_LATEST = 'latest'         # 最新模型
OPPONENT_PRETRAIN = 'pretrain'     # 预训练模型
OPPONENT_CHECKPOINT = 'checkpoint' # 历史检查点
OPPONENT_RANDOM = 'random'         # 随机策略


class Actor(Process):

    def __init__(self, config, replay_buffer):
        super(Actor, self).__init__()
        self.replay_buffer = replay_buffer
        self.config = config
        self.name = config.get('name', 'Actor-?')

    def _select_opponent_type(self, episode):
        """
        根据训练进度动态调整对手比例（课程学习）
        核心目标：打赢预训练模型

        阶段1 (0-2万局)：专注预训练
        阶段2 (2-6万局)：预训练为主，引入检查点
        阶段3 (6万+)：保持预训练，增加检查点
        """
        if episode < 20000:
            # 预训练85%, 自对弈15%
            r = random.random()
            if r < 0.85:
                return OPPONENT_PRETRAIN
            else:
                return OPPONENT_LATEST
        elif episode < 60000:
            # 预训练60%, 自对弈25%, 检查点15%
            r = random.random()
            if r < 0.60:
                return OPPONENT_PRETRAIN
            elif r < 0.85:
                return OPPONENT_LATEST
            else:
                return OPPONENT_CHECKPOINT
        else:
            # 预训练40%, 自对弈25%, 检查点35%
            r = random.random()
            if r < 0.40:
                return OPPONENT_PRETRAIN
            elif r < 0.65:
                return OPPONENT_LATEST
            else:
                return OPPONENT_CHECKPOINT

    def _load_checkpoint_model(self, model):
        """从本次运行的检查点目录随机加载一个模型"""
        ckpt_path = self.config.get('ckpt_save_path', './checkpoint/')

        # 扫描本次运行的检查点文件
        pattern = os.path.join(ckpt_path, '*.pt')
        ckpt_files = glob.glob(pattern)

        if not ckpt_files:
            return False

        # 随机选择一个
        ckpt_file = random.choice(ckpt_files)
        try:
            state_dict = torch.load(ckpt_file, map_location='cpu')
            model.load_state_dict(state_dict)
            return True
        except Exception as e:
            return False

    def run(self):
        torch.set_num_threads(1)

        # connect to model pool
        model_pool = ModelPoolClient(self.config['model_pool_name'])

        # create network models: 1 main + 3 opponents
        main_model = CNNModel()
        opponent_models = [CNNModel() for _ in range(3)]

        # 记录每个对手的类型（用于决定采样方式）
        opponent_types = [OPPONENT_LATEST] * 3

        # load pretrain model (for opponent use)
        pretrain_model = None
        pretrain_path = self.config.get('pretrain_path')
        if pretrain_path and os.path.exists(pretrain_path):
            pretrain_model = CNNModel()
            pretrain_model.load_state_dict(torch.load(pretrain_path, map_location='cpu'))
            pretrain_model.eval()

        # load initial model
        version = model_pool.get_latest_model()
        state_dict = model_pool.load_model(version)
        main_model.load_state_dict(state_dict)
        for opp in opponent_models:
            opp.load_state_dict(state_dict)

        # collect data
        env = MahjongGBEnv(config = {'agent_clz': FeatureAgentV2})

        for episode in range(self.config['episodes_per_actor']):
            # update main model to latest
            latest = model_pool.get_latest_model()
            if latest['id'] > version['id']:
                state_dict = model_pool.load_model(latest)
                if state_dict is not None:
                    main_model.load_state_dict(state_dict)
                    version = latest

            # 为每个对手随机选择类型和加载对应模型
            for i, opp in enumerate(opponent_models):
                opp_type = self._select_opponent_type(episode)
                opponent_types[i] = opp_type

                if opp_type == OPPONENT_LATEST:
                    # 使用最新模型
                    try:
                        opp_state = model_pool.load_model(latest)
                        if opp_state:
                            opp.load_state_dict(opp_state)
                    except Exception:
                        pass  # SharedMemory可能已释放，保持原模型

                elif opp_type == OPPONENT_PRETRAIN:
                    # 使用预训练模型
                    if pretrain_model is not None:
                        opp.load_state_dict(pretrain_model.state_dict())
                    else:
                        # 没有预训练模型，fallback到最新
                        opponent_types[i] = OPPONENT_LATEST
                        try:
                            opp_state = model_pool.load_model(latest)
                            if opp_state:
                                opp.load_state_dict(opp_state)
                        except Exception:
                            pass

                elif opp_type == OPPONENT_CHECKPOINT:
                    # 使用历史检查点
                    if not self._load_checkpoint_model(opp):
                        # 加载失败，fallback到最新
                        opponent_types[i] = OPPONENT_LATEST
                        try:
                            opp_state = model_pool.load_model(latest)
                            if opp_state:
                                opp.load_state_dict(opp_state)
                        except Exception:
                            pass

                elif opp_type == OPPONENT_RANDOM:
                    # 随机策略不需要加载模型，在采样时处理
                    pass

            # randomly choose main player position (for diverse seat wind)
            main_player_idx = random.randint(0, 3)
            main_player_name = f'player_{main_player_idx + 1}'

            # assign models and types: main player uses main_model, others use opponent_models
            opp_idx = 0
            models = {}
            player_types = {}  # 记录每个玩家的对手类型
            for i, agent_name in enumerate(env.agent_names):
                if i == main_player_idx:
                    models[agent_name] = main_model
                    player_types[agent_name] = 'main'
                else:
                    models[agent_name] = opponent_models[opp_idx]
                    player_types[agent_name] = opponent_types[opp_idx]
                    opp_idx += 1

            # run one episode and collect data
            obs = env.reset()
            episode_data = {agent_name: {
                'state' : {
                    'observation': [],
                    'action_mask': []
                },
                'action' : [],
                'reward' : [],
                'value' : []
            } for agent_name in env.agent_names}
            done = False
            while not done:
                # each player take action
                actions = {}
                values = {}
                for agent_name in obs:
                    agent_data = episode_data[agent_name]
                    state = obs[agent_name]
                    agent_data['state']['observation'].append(state['observation'])
                    agent_data['state']['action_mask'].append(state['action_mask'])

                    # 根据玩家类型选择动作
                    if player_types[agent_name] == OPPONENT_RANDOM:
                        # 随机策略：从合法动作中随机选
                        valid_actions = np.where(state['action_mask'] > 0)[0]
                        action = np.random.choice(valid_actions)
                        value = 0.0  # 随机策略没有value估计
                    else:
                        # 使用模型采样
                        state_tensor = {
                            'observation': torch.tensor(state['observation'], dtype=torch.float).unsqueeze(0),
                            'action_mask': torch.tensor(state['action_mask'], dtype=torch.float).unsqueeze(0)
                        }
                        current_model = models[agent_name]
                        current_model.train(False)
                        with torch.no_grad():
                            logits, value_tensor = current_model(state_tensor)
                            action_dist = torch.distributions.Categorical(logits=logits)
                            action = action_dist.sample().item()
                            value = value_tensor.item()

                    actions[agent_name] = action
                    values[agent_name] = value
                    agent_data['action'].append(action)
                    agent_data['value'].append(value)

                # interact with env
                next_obs, rewards, done = env.step(actions)
                for agent_name in rewards:
                    episode_data[agent_name]['reward'].append(rewards[agent_name])
                obs = next_obs

            # 打印对局信息
            opp_type_str = ','.join([player_types[f'player_{i+1}'] for i in range(4) if i != main_player_idx])
            print(f"{self.name} Ep {episode} Model {latest['id']} Main {main_player_name} Opp [{opp_type_str}] Reward {rewards[main_player_name]:.1f}")

            # postprocessing episode data - only for main player
            agent_data = episode_data[main_player_name]
            if len(agent_data['action']) < len(agent_data['reward']):
                agent_data['reward'].pop(0)
            obs_arr = np.stack(agent_data['state']['observation'])
            mask = np.stack(agent_data['state']['action_mask'])
            actions_arr = np.array(agent_data['action'], dtype=np.int64)
            rewards_arr = np.array(agent_data['reward'], dtype=np.float32)
            values_arr = np.array(agent_data['value'], dtype=np.float32)
            next_values = np.array(agent_data['value'][1:] + [0], dtype=np.float32)

            td_target = rewards_arr + next_values * self.config['gamma']
            td_delta = td_target - values_arr
            advs = []
            adv = 0
            for delta in td_delta[::-1]:
                adv = self.config['gamma'] * self.config['lambda'] * adv + delta
                advs.append(adv)
            advs.reverse()
            advantages = np.array(advs, dtype=np.float32)

            # send samples to replay_buffer (only main player)
            self.replay_buffer.push({
                'state': {
                    'observation': obs_arr,
                    'action_mask': mask
                },
                'action': actions_arr,
                'adv': advantages,
                'target': td_target
            })
