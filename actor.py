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
        # 从 name 中提取 actor index（如 'Actor-5' -> 5），用于生成唯一 traj_id
        try:
            self.actor_id = int(self.name.split('-')[1])
        except (IndexError, ValueError):
            self.actor_id = 0
        self.self_play_mode = config.get('self_play_mode', False)
        # 自博弈模式下，预训练对手的概率（每局最多1个预训练）
        self.pretrain_prob = config.get('pretrain_prob', 0.3)

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

        # 扫描本次运行的检查点文件（只匹配 model_*.pt，排除 critic_*.pt）
        pattern = os.path.join(ckpt_path, 'model_*.pt')
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
        env = MahjongGBEnv(config={
            'agent_clz': FeatureAgentV2,
            'reward_mode': self.config.get('reward_mode', 'simple')
        })

        for episode in range(self.config['episodes_per_actor']):
            # update main model to latest
            latest = model_pool.get_latest_model()
            if latest['id'] > version['id']:
                state_dict = model_pool.load_model(latest)
                if state_dict is not None:
                    main_model.load_state_dict(state_dict)
                    version = latest

            # ==================== 自博弈模式 ====================
            if self.self_play_mode:
                # 自博弈模式：4个玩家都用最新模型，最多1个用预训练
                # 决定是否有一个预训练对手
                has_pretrain = (random.random() < self.pretrain_prob) and (pretrain_model is not None)
                pretrain_player_idx = random.randint(0, 3) if has_pretrain else -1

                models = {}
                player_types = {}
                for i, agent_name in enumerate(env.agent_names):
                    if i == pretrain_player_idx:
                        models[agent_name] = pretrain_model
                        player_types[agent_name] = OPPONENT_PRETRAIN
                    else:
                        models[agent_name] = main_model
                        player_types[agent_name] = OPPONENT_LATEST

            # ==================== 原有模式 ====================
            else:
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
                    'action_mask': [],
                    'other_hands': []  # CTDE: 其他玩家手牌（12×4×9），在 learner 拼接
                },
                'action' : [],
                'reward' : [],  # 即时奖励，过程为0，终局写到最后
                'done': [],     # 是否终止，最后一步为True
                'log_prob': []  # 存储采样时的 log_prob，用于严格 on-policy PPO
            } for agent_name in env.agent_names}
            auto_pass_counts = {agent_name: 0 for agent_name in env.agent_names}  # debug统计
            done = False
            while not done:
                # 获取全局手牌信息（用于 CTDE）
                all_hands = env.get_global_hands()

                # each player take action
                actions = {}
                for agent_name in obs:
                    state = obs[agent_name]
                    action_mask = state['action_mask']

                    # ========== 自动过牌：只有 Pass 合法时直接过，不产数据 ==========
                    valid_actions = np.where(action_mask > 0)[0]
                    if len(valid_actions) == 1 and valid_actions[0] == 0:
                        actions[agent_name] = 0  # Pass
                        auto_pass_counts[agent_name] += 1
                        continue  # 跳过数据记录

                    # ========== 正常采样流程 ==========
                    agent_data = episode_data[agent_name]

                    # 获取对应的 FeatureAgentV2 实例，构造其他玩家手牌特征
                    player_idx = int(agent_name.split('_')[1]) - 1
                    feature_agent = env.agents[player_idx]
                    other_hands = feature_agent.build_other_hands_obs(all_hands)

                    # 记录 state（包括其他玩家手牌）
                    agent_data['state']['observation'].append(state['observation'])
                    agent_data['state']['action_mask'].append(state['action_mask'])
                    agent_data['state']['other_hands'].append(other_hands)

                    # 根据玩家类型选择动作
                    if player_types[agent_name] == OPPONENT_RANDOM:
                        # 随机策略：从合法动作中随机选
                        valid_actions = np.where(action_mask > 0)[0]
                        action = np.random.choice(valid_actions)
                        log_prob = 0.0  # 随机策略的 log_prob 不重要（不会被采样）
                    else:
                        # 使用模型采样（只需要 policy，不需要 value）
                        state_tensor = {
                            'observation': torch.tensor(state['observation'], dtype=torch.float).unsqueeze(0),
                            'action_mask': torch.tensor(state['action_mask'], dtype=torch.float).unsqueeze(0)
                        }
                        current_model = models[agent_name]
                        current_model.train(False)
                        with torch.no_grad():
                            logits, _ = current_model(state_tensor)  # 不再使用 value
                            action_dist = torch.distributions.Categorical(logits=logits)
                            sampled_action = action_dist.sample()
                            action = sampled_action.item()
                            log_prob = action_dist.log_prob(sampled_action).item()

                    actions[agent_name] = action
                    agent_data['action'].append(action)
                    agent_data['log_prob'].append(log_prob)
                    agent_data['reward'].append(0)  # 过程奖励为0
                    agent_data['done'].append(False)  # 非终止步

                # interact with env
                next_obs, rewards, done = env.step(actions)
                obs = next_obs

            # 游戏结束后，更新每个玩家最后一步的 reward 和 done
            for agent_name in env.agent_names:
                if episode_data[agent_name]['reward']:
                    episode_data[agent_name]['reward'][-1] = rewards[agent_name]
                    episode_data[agent_name]['done'][-1] = True  # 最后一步标记为终止

            # 打印对局信息
            total_auto_pass = sum(auto_pass_counts.values())
            if self.self_play_mode:
                pretrain_name = f'player_{pretrain_player_idx + 1}' if pretrain_player_idx >= 0 else 'None'
                # 输出每个玩家的reward
                reward_strs = [f"P{i+1}:{rewards[f'player_{i+1}']:.1f}" for i in range(4)]
                print(f"{self.name} Ep {episode} Model {latest['id']} Pretrain [{pretrain_name}] Rewards [{', '.join(reward_strs)}] AutoPass {total_auto_pass}")
            else:
                opp_type_str = ','.join([player_types[f'player_{i+1}'] for i in range(4) if player_types.get(f'player_{i+1}') != 'main'])
                main_player_name = [name for name, ptype in player_types.items() if ptype == 'main'][0]
                # 输出每个玩家的reward
                reward_strs = [f"P{i+1}:{rewards[f'player_{i+1}']:.1f}" for i in range(4)]
                print(f"{self.name} Ep {episode} Model {latest['id']} Main {main_player_name} Opp [{opp_type_str}] Rewards [{', '.join(reward_strs)}] AutoPass {total_auto_pass}")

            # ==================== 数据采样 ====================
            # CTDE: Actor 只存原始数据，不算 adv/target
            # adv/target 由 Learner 用 Centralized Critic 计算
            current_policy_id = version['id']

            # 生成唯一轨迹ID：actor_id(8bit) + episode(20bit) + player_idx(4bit)
            # 支持 256 个 actor，100万局，4 个玩家
            # traj_id 格式: [actor_id:8][episode:20][player_idx:4] = 32bit

            if self.self_play_mode:
                # 自博弈模式：采样所有用最新模型的玩家的数据
                for player_idx, (agent_name, agent_data) in enumerate(episode_data.items()):
                    # 跳过预训练对手的数据
                    if player_types[agent_name] == OPPONENT_PRETRAIN:
                        continue
                    # 跳过空数据（所有动作都是自动过牌）
                    if len(agent_data['action']) == 0:
                        continue

                    n_samples = len(agent_data['action'])
                    obs_arr = np.stack(agent_data['state']['observation'])
                    mask = np.stack(agent_data['state']['action_mask'])
                    other_hands_arr = np.stack(agent_data['state']['other_hands'])  # CTDE: 12×4×9
                    actions_arr = np.array(agent_data['action'], dtype=np.int64)
                    rewards_arr = np.array(agent_data['reward'], dtype=np.float32)
                    dones_arr = np.array(agent_data['done'], dtype=np.bool_)
                    log_probs_arr = np.array(agent_data['log_prob'], dtype=np.float32)

                    # 轨迹标识：用于 Learner 重建轨迹计算 GAE
                    # traj_id = actor_id(8bit) + episode(20bit) + player_idx(4bit)
                    traj_id = (self.actor_id << 24) | ((episode & 0xFFFFF) << 4) | player_idx
                    t_arr = np.arange(n_samples, dtype=np.int32)

                    episode_reward = rewards[agent_name]
                    self.replay_buffer.push({
                        'state': {
                            'observation': obs_arr,
                            'action_mask': mask,
                            'other_hands': other_hands_arr  # CTDE: 12×4×9
                        },
                        'action': actions_arr,
                        'log_prob': log_probs_arr,
                        'reward': rewards_arr,      # 原始奖励，Learner 算 GAE
                        'done': dones_arr,          # 终止标记
                        'traj_id': np.full(n_samples, traj_id, dtype=np.int64),  # 轨迹ID
                        't': t_arr,                 # 时间步
                        'policy_id': np.full(n_samples, current_policy_id, dtype=np.int64),
                        'episode_reward': episode_reward
                    })
            else:
                # 原有模式：只采样主玩家
                main_player_name = [name for name, ptype in player_types.items() if ptype == 'main'][0]
                main_player_idx = int(main_player_name.split('_')[1]) - 1
                agent_data = episode_data[main_player_name]
                # 跳过空数据（所有动作都是自动过牌）
                if len(agent_data['action']) == 0:
                    continue

                n_samples = len(agent_data['action'])
                obs_arr = np.stack(agent_data['state']['observation'])
                mask = np.stack(agent_data['state']['action_mask'])
                other_hands_arr = np.stack(agent_data['state']['other_hands'])  # CTDE: 12×4×9
                actions_arr = np.array(agent_data['action'], dtype=np.int64)
                rewards_arr = np.array(agent_data['reward'], dtype=np.float32)
                dones_arr = np.array(agent_data['done'], dtype=np.bool_)
                log_probs_arr = np.array(agent_data['log_prob'], dtype=np.float32)

                # 轨迹标识
                # traj_id = actor_id(8bit) + episode(20bit) + player_idx(4bit)
                traj_id = (self.actor_id << 24) | ((episode & 0xFFFFF) << 4) | main_player_idx
                t_arr = np.arange(n_samples, dtype=np.int32)

                episode_reward = rewards[main_player_name]
                self.replay_buffer.push({
                    'state': {
                        'observation': obs_arr,
                        'action_mask': mask,
                        'other_hands': other_hands_arr  # CTDE: 12×4×9
                    },
                    'action': actions_arr,
                    'log_prob': log_probs_arr,
                    'reward': rewards_arr,      # 原始奖励，Learner 算 GAE
                    'done': dones_arr,          # 终止标记
                    'traj_id': np.full(n_samples, traj_id, dtype=np.int64),  # 轨迹ID
                    't': t_arr,                 # 时间步
                    'policy_id': np.full(n_samples, current_policy_id, dtype=np.int64),
                    'episode_reward': episode_reward
                })
