# Chinese Standard Mahjong (国标麻将) 强化学习课程大作业

本项目是北京大学 2025 秋《强化学习》课程大作业：基于国标麻将环境，使用 PPO 训练对局策略，并提供监督学习预训练、对战评估与 Botzone 提交脚本。

## 功能概览

- 国标麻将环境：`env.py`（基于 `PyMahjongGB` 计算番型/番数）
- 特征与动作：`feature_v2.py`（147×4×9 观测；235 维动作空间）
- 监督学习预训练：`preprocess.py` + `supervised.py`
- 强化学习训练（PPO, Actor-Learner 多进程）：`train.py` / `actor.py` / `learner.py`
- 模型对战评估：`evaluate.py`
- Botzone 单文件提交：`__main__.py`

## 环境要求

- Python 3.10+（本仓库开发环境可用 `Python 3.12`）
- PyTorch（训练默认使用 CUDA；如需 CPU 训练请自行改动脚本里的 `device`/`.to('cuda')`）
- 其余依赖见 `requirements.txt`（包含 `PyMahjongGB`）

## 安装

建议使用 Conda 环境（Windows PowerShell 示例）：

```powershell
conda create -n mahjong-rl python=3.12 -y
conda activate mahjong-rl
python -m pip install -U pip
pip install -r requirements.txt
```

提示：`requirements.txt` 中 `torch/torchvision` 未锁版本；如果你需要特定 CUDA 版本，请先按 PyTorch 官方指引安装对应版本，再安装其余依赖。

## 数据准备（监督学习用）

原始对局日志在 `data/data.txt`（约 9.8 万局，文件较大不建议直接用编辑器打开），格式说明见：`data/README.txt`。

1) 预处理生成训练样本（会在 `data/` 下生成大量 `*.npz` 以及 `data/count.json`，耗时/占用磁盘较大）：

```bash
python preprocess.py
```

## 监督学习预训练

在完成“数据准备”后运行：

```bash
python supervised.py
```

- checkpoint 输出到 `supervised_checkpoint/<时间戳>/epoch_*.pt`（以及 `final.pt`）
- TensorBoard 日志输出到 `supervised_runs/<时间戳>/`

恢复训练：

```bash
python supervised.py --resume supervised_checkpoint/<时间戳>/epoch_3.pt
```

查看训练曲线：

```bash
tensorboard --logdir supervised_runs
```

## 强化学习训练（PPO）

从头训练：

```bash
python train.py
```

使用监督学习模型作为初始化（推荐）：

```bash
python train.py --pretrain supervised_checkpoint/<时间戳>/final.pt
```

继续训练（传入 checkpoint 目录，而不是单个文件）：

```bash
python train.py --resume checkpoint/<时间戳>
```

常用参数：

- `--num-actors`：actor 进程数（默认 23；机器资源不足可调小）
- `--episodes-per-actor`：每个 actor 跑多少局（`<=0` 表示一直跑，Ctrl+C 结束）
- `--no-self-play`：关闭自博弈（默认开启）
- `--reward-mode {simple,sqrt,original}`：奖励模式

训练产物：

- checkpoints：`checkpoint/<时间戳>/model_*.pt`（以及 `critic_latest.pt`）
- TensorBoard：`runs/<时间戳>/`

## 模型评估（自对局）

让两个模型在环境中对战并统计胜率/平均奖励：

```bash
python evaluate.py --current checkpoint/<exp>/model_1000.pt --baseline checkpoint/<exp>/model_500.pt --games 100 --device cuda
```

没有 GPU 可用：把 `--device` 改为 `cpu`。

## Botzone 提交/运行

`__main__.py` 是为 Botzone 提交准备的“单文件版本”（合并了依赖）。

- 本地运行：`python __main__.py`（需按 Botzone 交互格式提供输入）
- 模型路径：脚本里默认从 `data_dir = '/data/*.pt'` 读取模型；用于 Botzone 时通常需要把模型文件放到其指定的 `/data` 目录下并匹配路径/文件名

## 参考

- 依赖库：`PyMahjongGB`（提供 `MahjongFanCalculator` 计番）
- 报告：`report.pdf` 
