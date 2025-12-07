import torch
from torch import nn
from torch.nn import functional as F


class ResBlock(nn.Module):
    """残差块：两层卷积 + 跳跃连接"""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


class CNNModel(nn.Module):
    """
    深度ResNet风格的Actor-Critic网络

    结构：
    - 输入卷积: in_channels → 128 通道
    - 8个残差块 (共16层卷积)
    - 策略头: 128*4*9 → 512 → 235
    - 价值头: 128*4*9 → 512 → 1

    输入通道数:
    - 旧版 (FeatureAgent): 6 通道
    - 新版 (FeatureAgentV2): 147 通道
    """

    def __init__(self, num_res_blocks=8, channels=128, in_channels=147):
        super().__init__()

        self.num_res_blocks = num_res_blocks
        self.channels = channels
        self.in_channels = in_channels

        # 输入卷积层
        self.input_conv = nn.Conv2d(in_channels, channels, 3, 1, 1, bias=False)
        self.input_bn = nn.BatchNorm2d(channels)

        # 残差块
        self.res_blocks = nn.ModuleList([
            ResBlock(channels) for _ in range(num_res_blocks)
        ])

        # 策略头
        self._logits = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 4 * 9, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 235)
        )

        # 价值头
        self._value_branch = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 4 * 9, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 1)
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input_dict):
        obs = input_dict["observation"].float()

        # 输入层
        x = F.relu(self.input_bn(self.input_conv(obs)))

        # 残差块
        for res_block in self.res_blocks:
            x = res_block(x)

        # 策略和价值输出
        logits = self._logits(x)
        value = self._value_branch(x)

        # 动作掩码
        mask = input_dict["action_mask"].float()
        inf_mask = torch.clamp(torch.log(mask), -1e38, 1e38)
        masked_logits = logits + inf_mask

        return masked_logits, value


# 保留旧版本用于加载旧checkpoint（可选）
class CNNModelLegacy(nn.Module):
    """原始浅层网络，用于兼容旧模型"""

    def __init__(self):
        nn.Module.__init__(self)
        self._tower = nn.Sequential(
            nn.Conv2d(6, 64, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(64, 32, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Flatten()
        )
        self._logits = nn.Sequential(
            nn.Linear(32 * 4 * 9, 256),
            nn.ReLU(True),
            nn.Linear(256, 235)
        )
        self._value_branch = nn.Sequential(
            nn.Linear(32 * 4 * 9, 256),
            nn.ReLU(True),
            nn.Linear(256, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input_dict):
        obs = input_dict["observation"].float()
        hidden = self._tower(obs)
        logits = self._logits(hidden)
        mask = input_dict["action_mask"].float()
        inf_mask = torch.clamp(torch.log(mask), -1e38, 1e38)
        masked_logits = logits + inf_mask
        value = self._value_branch(hidden)
        return masked_logits, value
