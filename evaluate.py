"""
评估脚本：让当前模型和历史版本对战

用法：
    python evaluate.py --current model_1000.pt --baseline model_0.pt --games 100
    python evaluate.py --current model_1000.pt --baseline model_500.pt --games 50
"""

import argparse
import torch
import numpy as np
from collections import defaultdict

from env import MahjongGBEnv
from feature import FeatureAgent
from model import CNNModel


def load_model(path, device='cpu'):
    """加载模型"""
    model = CNNModel()
    if path:
        model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


def select_action(model, obs, device='cpu'):
    """根据观测选择动作"""
    with torch.no_grad():
        state = {
            'observation': torch.tensor(obs['observation'], dtype=torch.float).unsqueeze(0).to(device),
            'action_mask': torch.tensor(obs['action_mask'], dtype=torch.float).unsqueeze(0).to(device)
        }
        logits, _ = model(state)
        action = logits.argmax(dim=1).item()
    return action


def play_one_game(models, device='cpu'):
    """
    进行一局游戏

    Args:
        models: dict, {player_name: model} 每个玩家使用的模型

    Returns:
        rewards: dict, 每个玩家的最终奖励
        winner: str, 赢家名称 (如果有)
    """
    env = MahjongGBEnv(config={'agent_clz': FeatureAgent})
    obs = env.reset()
    done = False

    while not done:
        actions = {}
        for player_name in obs:
            model = models[player_name]
            actions[player_name] = select_action(model, obs[player_name], device)

        obs, rewards, done = env.step(actions)

    # 找出赢家
    winner = None
    max_reward = -float('inf')
    for player, reward in rewards.items():
        if reward > max_reward:
            max_reward = reward
            winner = player

    return rewards, winner


def evaluate(current_model_path, baseline_model_path, num_games=100, device='cpu'):
    """
    评估当前模型 vs 基准模型

    对战设置：
    - Player_0, Player_2: 当前模型 (对角位置)
    - Player_1, Player_3: 基准模型 (对角位置)

    这样更公平，因为麻将座位有优势差异
    """
    print(f"加载模型...")
    print(f"  当前模型: {current_model_path}")
    print(f"  基准模型: {baseline_model_path}")

    current_model = load_model(current_model_path, device)
    baseline_model = load_model(baseline_model_path, device)

    # 设置对战：当前模型 vs 基准模型 (对角位置)
    models = {
        'player_1': current_model,   # 当前
        'player_2': baseline_model,  # 基准
        'player_3': current_model,   # 当前
        'player_4': baseline_model,  # 基准
    }

    # 统计
    stats = {
        'current_wins': 0,
        'baseline_wins': 0,
        'draws': 0,
        'current_total_reward': 0,
        'baseline_total_reward': 0,
    }

    current_players = {'player_1', 'player_3'}
    baseline_players = {'player_2', 'player_4'}

    print(f"\n开始对战 {num_games} 局...")
    print("-" * 50)

    for game_idx in range(num_games):
        rewards, winner = play_one_game(models, device)

        # 统计奖励
        for player, reward in rewards.items():
            if player in current_players:
                stats['current_total_reward'] += reward
            else:
                stats['baseline_total_reward'] += reward

        # 统计胜负
        if winner in current_players:
            stats['current_wins'] += 1
        elif winner in baseline_players:
            stats['baseline_wins'] += 1
        else:
            stats['draws'] += 1

        # 进度显示
        if (game_idx + 1) % 10 == 0:
            win_rate = stats['current_wins'] / (game_idx + 1) * 100
            print(f"进度: {game_idx + 1}/{num_games} | 当前模型胜率: {win_rate:.1f}%")

    # 最终结果
    print("\n" + "=" * 50)
    print("评估结果")
    print("=" * 50)

    total_games = num_games
    current_win_rate = stats['current_wins'] / total_games * 100
    baseline_win_rate = stats['baseline_wins'] / total_games * 100
    draw_rate = stats['draws'] / total_games * 100

    # 注意：每局有2个当前模型玩家和2个基准模型玩家
    current_avg_reward = stats['current_total_reward'] / (total_games * 2)
    baseline_avg_reward = stats['baseline_total_reward'] / (total_games * 2)

    print(f"\n对局数: {total_games}")
    print(f"\n胜场统计:")
    print(f"  当前模型: {stats['current_wins']} 胜 ({current_win_rate:.1f}%)")
    print(f"  基准模型: {stats['baseline_wins']} 胜 ({baseline_win_rate:.1f}%)")
    print(f"  流局:     {stats['draws']} 局 ({draw_rate:.1f}%)")

    print(f"\n平均奖励:")
    print(f"  当前模型: {current_avg_reward:.2f}")
    print(f"  基准模型: {baseline_avg_reward:.2f}")

    # 判断结果
    print(f"\n结论:")
    if current_win_rate > baseline_win_rate + 5:
        print(f"  ✅ 当前模型更强 (胜率 +{current_win_rate - baseline_win_rate:.1f}%)")
    elif baseline_win_rate > current_win_rate + 5:
        print(f"  ❌ 基准模型更强 (胜率 -{baseline_win_rate - current_win_rate:.1f}%)")
    else:
        print(f"  ⚖️  两个模型实力相当")

    return stats


def evaluate_multiple(model_dir, model_list, num_games=50, device='cpu'):
    """
    评估多个模型版本，生成 Elo 风格的排名

    Args:
        model_dir: 模型目录
        model_list: 模型文件名列表，如 ['model_0.pt', 'model_100.pt', ...]
        num_games: 每对模型对战的局数
    """
    import os
    from itertools import combinations

    print(f"评估 {len(model_list)} 个模型版本")
    print("=" * 50)

    # 加载所有模型
    models = {}
    for model_name in model_list:
        path = os.path.join(model_dir, model_name)
        if os.path.exists(path):
            models[model_name] = load_model(path, device)
            print(f"  已加载: {model_name}")
        else:
            print(f"  未找到: {model_name}")

    # 胜负记录
    win_counts = defaultdict(int)
    total_counts = defaultdict(int)

    # 两两对战
    model_names = list(models.keys())
    for m1, m2 in combinations(model_names, 2):
        print(f"\n{m1} vs {m2}...")

        model_assignment = {
            'player_1': models[m1],
            'player_2': models[m2],
            'player_3': models[m1],
            'player_4': models[m2],
        }

        m1_wins = 0
        m2_wins = 0

        for _ in range(num_games):
            rewards, winner = play_one_game(model_assignment, device)
            if winner in {'player_1', 'player_3'}:
                m1_wins += 1
            elif winner in {'player_2', 'player_4'}:
                m2_wins += 1

        win_counts[m1] += m1_wins
        win_counts[m2] += m2_wins
        total_counts[m1] += num_games
        total_counts[m2] += num_games

        print(f"  {m1}: {m1_wins} 胜 | {m2}: {m2_wins} 胜")

    # 排名
    print("\n" + "=" * 50)
    print("总排名 (按胜率)")
    print("=" * 50)

    rankings = []
    for name in model_names:
        if total_counts[name] > 0:
            win_rate = win_counts[name] / total_counts[name] * 100
            rankings.append((name, win_rate, win_counts[name], total_counts[name]))

    rankings.sort(key=lambda x: x[1], reverse=True)

    for rank, (name, win_rate, wins, total) in enumerate(rankings, 1):
        print(f"  {rank}. {name}: {win_rate:.1f}% ({wins}/{total})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='评估麻将AI模型')
    parser.add_argument('--current', type=str, required=True, help='当前模型路径')
    parser.add_argument('--baseline', type=str, required=True, help='基准模型路径')
    parser.add_argument('--games', type=int, default=100, help='对战局数')
    parser.add_argument('--device', type=str, default='cpu', help='设备 (cpu/cuda)')

    args = parser.parse_args()

    evaluate(args.current, args.baseline, args.games, args.device)
