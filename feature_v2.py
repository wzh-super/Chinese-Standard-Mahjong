"""
FeatureAgentV2: 147通道特征提取

通道分布 (共147通道):
- [0]: 门风 (Seat Wind)
- [1]: 场风 (Prevalent Wind)
- [2-5]: 己方手牌 (Own Hand, 4 channels)
- [6-21]: 各方吃牌 (Chi for 4 players × 4 chi each = 16 channels)
    - [6-9]: Player 0 (self)
    - [10-13]: Player 1
    - [14-17]: Player 2
    - [18-21]: Player 3
- [22-25]: 各方碰牌 (Peng for 4 players × 1 = 4 channels)
- [26-29]: 各方杠牌 (Gang for 4 players × 1 = 4 channels)
- [30]: 己方暗杠 (Own AnGang, 1 channel)
- [31-142]: 各方弃牌历史 (Discard for 4 players × 28 = 112 channels)
    - [31-58]: Player 0 discards
    - [59-86]: Player 1 discards
    - [87-114]: Player 2 discards
    - [115-142]: Player 3 discards
- [143-146]: 牌山 (Tile Wall, 4 channels)

观测形状: (147, 4, 9)
- 4行: 万/条/饼/字
- 9列: 1-9 (字牌只用前7列)
"""

from agent import MahjongGBAgent
from collections import defaultdict
import numpy as np

try:
    from MahjongGB import MahjongFanCalculator
except:
    print('MahjongGB library required! Please visit https://github.com/ailab-pku/PyMahjongGB for more information.')
    raise


class FeatureAgentV2(MahjongGBAgent):
    """
    147通道特征提取Agent

    observation: 147 × 4 × 9
    action_mask: 235

    全局观测 (用于 CTDE Critic): 159 × 4 × 9
    - 147 channels: 原有观测
    - 12 channels: 其他3个玩家的手牌 (每人4 channels)
    """

    OBS_SIZE = 147
    GLOBAL_OBS_SIZE = 159  # 147 + 12 (其他3人手牌)
    ACT_SIZE = 235

    # 用于环境接口
    observation_space = (OBS_SIZE, 4, 9)
    global_observation_space = (GLOBAL_OBS_SIZE, 4, 9)
    action_space = ACT_SIZE

    # 观测通道偏移
    OFFSET_OBS = {
        'SEAT_WIND': 0,           # 1 channel
        'PREVALENT_WIND': 1,      # 1 channel
        'HAND': 2,                # 4 channels [2-5]
        'CHI': 6,                 # 16 channels [6-21], 4 players × 4 chi each
        'PENG': 22,               # 4 channels [22-25], 4 players × 1
        'GANG': 26,               # 4 channels [26-29], 4 players × 1
        'ANGANG': 30,             # 1 channel [30], only own
        'DISCARD': 31,            # 112 channels [31-142], 4 players × 28
        'WALL': 143,              # 4 channels [143-146]
    }

    # 动作偏移 (与原版相同)
    OFFSET_ACT = {
        'Pass': 0,
        'Hu': 1,
        'Play': 2,
        'Chi': 36,
        'Peng': 99,
        'Gang': 133,
        'AnGang': 167,
        'BuGang': 201
    }

    # 牌列表
    TILE_LIST = [
        *('W%d' % (i + 1) for i in range(9)),  # 万 0-8
        *('T%d' % (i + 1) for i in range(9)),  # 条 9-17
        *('B%d' % (i + 1) for i in range(9)),  # 饼 18-26
        *('F%d' % (i + 1) for i in range(4)),  # 风 27-30 (东南西北)
        *('J%d' % (i + 1) for i in range(3))   # 箭 31-33 (中发白)
    ]
    OFFSET_TILE = {c: i for i, c in enumerate(TILE_LIST)}

    def __init__(self, seatWind):
        self.seatWind = seatWind
        self.packs = [[] for _ in range(4)]  # 各玩家的副露
        self.history = [[] for _ in range(4)]  # 各玩家的弃牌历史
        self.tileWall = [21] * 4  # 各玩家的牌墙剩余
        self.shownTiles = defaultdict(int)  # 已显示的牌计数
        self.wallLast = False
        self.isAboutKong = False

        # 吃牌计数 (每个玩家最多4次吃)
        self.chi_count = [0, 0, 0, 0]
        # 弃牌计数 (每个玩家最多28张)
        self.discard_count = [0, 0, 0, 0]

        # 初始化观测张量 (147 × 36)
        self.obs = np.zeros((self.OBS_SIZE, 36), dtype=np.float32)

        # 设置门风
        self.obs[self.OFFSET_OBS['SEAT_WIND']][self.OFFSET_TILE['F%d' % (self.seatWind + 1)]] = 1

        # 初始化牌山 (所有牌各4张都在牌山中)
        self.obs[self.OFFSET_OBS['WALL']:self.OFFSET_OBS['WALL'] + 4, :] = 1

    def _tile_to_pos(self, tile):
        """将牌名转换为 (row, col) 位置"""
        idx = self.OFFSET_TILE[tile]
        return idx // 9, idx % 9

    def _update_wall(self, tile, delta=-1):
        """更新牌山特征

        Args:
            tile: 牌名
            delta: -1 表示移除一张牌, +1 表示恢复
        """
        # row, col = self._tile_to_pos(tile)
        index = self.OFFSET_TILE[tile]
        base = self.OFFSET_OBS['WALL']

        if delta < 0:
            # 移除牌: 从最高的1开始置0
            for i in range(3, -1, -1):
                if self.obs[base + i, index] == 1:
                    self.obs[base + i, index] = 0
                    break
        else:
            # 恢复牌: 从最低的0开始置1
            for i in range(4):
                if self.obs[base + i, index] == 0:
                    self.obs[base + i, index] = 1
                    break

    def _update_hand(self):
        """更新己方手牌特征"""
        base = self.OFFSET_OBS['HAND']
        self.obs[base:base + 4, :] = 0

        d = defaultdict(int)
        for tile in self.hand:
            d[tile] += 1

        for tile, count in d.items():
            idx = self.OFFSET_TILE[tile]
            for i in range(count):
                self.obs[base + i, idx] = 1

    def _update_chi(self, player, chi_tile):
        """更新吃牌特征

        Args:
            player: 相对玩家编号 (0=自己)
            chi_tile: 吃牌的中间牌
        """
        if self.chi_count[player] >= 4:
            return  # 最多记录4次吃

        base = self.OFFSET_OBS['CHI'] + player * 4 + self.chi_count[player]
        color = chi_tile[0]
        num = int(chi_tile[1])

        # 吃牌涉及三张牌
        for i in range(-1, 2):
            tile = color + str(num + i)
            idx = self.OFFSET_TILE[tile]
            self.obs[base, idx] = 1

        self.chi_count[player] += 1

    def _update_peng(self, player, tile):
        """更新碰牌特征"""
        base = self.OFFSET_OBS['PENG'] + player
        idx = self.OFFSET_TILE[tile]
        self.obs[base, idx] = 1

    def _remove_peng(self, player, tile):
        """移除碰牌特征 (用于补杠)"""
        base = self.OFFSET_OBS['PENG'] + player
        idx = self.OFFSET_TILE[tile]
        self.obs[base, idx] = 0

    def _update_gang(self, player, tile):
        """更新杠牌特征"""
        base = self.OFFSET_OBS['GANG'] + player
        idx = self.OFFSET_TILE[tile]
        self.obs[base, idx] = 1

    def _update_angang(self, tile):
        """更新己方暗杠特征"""
        base = self.OFFSET_OBS['ANGANG']
        idx = self.OFFSET_TILE[tile]
        self.obs[base, idx] = 1

    def _update_discard(self, player, tile):
        """更新弃牌历史特征"""
        if self.discard_count[player] >= 28:
            return  # 最多记录28张弃牌

        base = self.OFFSET_OBS['DISCARD'] + player * 28 + self.discard_count[player]
        idx = self.OFFSET_TILE[tile]
        self.obs[base, idx] = 1

        self.discard_count[player] += 1

    def request2obs(self, request):
        """处理游戏请求，更新观测"""
        t = request.split()

        if t[0] == 'Wind':
            self.prevalentWind = int(t[1])
            self.obs[self.OFFSET_OBS['PREVALENT_WIND']][self.OFFSET_TILE['F%d' % (self.prevalentWind + 1)]] = 1
            return

        if t[0] == 'Deal':
            self.hand = t[1:]
            self._update_hand()
            # 手牌从牌山移除
            for tile in self.hand:
                self._update_wall(tile, -1)
            return

        if t[0] == 'Huang':
            self.valid = []
            return self._obs()

        if t[0] == 'Draw':
            # 自己摸牌
            self.tileWall[0] -= 1
            self.wallLast = self.tileWall[1] == 0
            tile = t[1]
            self.valid = []

            if self._check_mahjong(tile, isSelfDrawn=True, isAboutKong=self.isAboutKong):
                self.valid.append(self.OFFSET_ACT['Hu'])

            self.isAboutKong = False
            self.hand.append(tile)
            self._update_hand()
            self._update_wall(tile, -1)  # 从牌山移除

            for tile in set(self.hand):
                self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                if self.hand.count(tile) == 4 and not self.wallLast and self.tileWall[0] > 0:
                    self.valid.append(self.OFFSET_ACT['AnGang'] + self.OFFSET_TILE[tile])

            if not self.wallLast and self.tileWall[0] > 0:
                for packType, tile, offer in self.packs[0]:
                    if packType == 'PENG' and tile in self.hand:
                        self.valid.append(self.OFFSET_ACT['BuGang'] + self.OFFSET_TILE[tile])

            return self._obs()

        # Player N 动作
        p = (int(t[1]) + 4 - self.seatWind) % 4  # 相对玩家编号

        if t[2] == 'Draw':
            self.tileWall[p] -= 1
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            return

        if t[2] == 'Invalid':
            self.valid = []
            return self._obs()

        if t[2] == 'Hu':
            self.valid = []
            return self._obs()

        if t[2] == 'Play':
            self.tileFrom = p
            self.curTile = t[3]
            self.shownTiles[self.curTile] += 1
            self.history[p].append(self.curTile)

            # 更新弃牌历史特征
            self._update_discard(p, self.curTile)
            # 更新牌山 (弃牌不再是未知牌)
            if p != 0:
                self._update_wall(self.curTile, -1)

            if p == 0:
                self.hand.remove(self.curTile)
                self._update_hand()
                return
            else:
                # 可选: Hu/Gang/Peng/Chi/Pass
                self.valid = []
                if self._check_mahjong(self.curTile):
                    self.valid.append(self.OFFSET_ACT['Hu'])

                if not self.wallLast:
                    if self.hand.count(self.curTile) >= 2:
                        self.valid.append(self.OFFSET_ACT['Peng'] + self.OFFSET_TILE[self.curTile])
                        if self.hand.count(self.curTile) == 3 and self.tileWall[0]:
                            self.valid.append(self.OFFSET_ACT['Gang'] + self.OFFSET_TILE[self.curTile])

                    color = self.curTile[0]
                    if p == 3 and color in 'WTB':
                        num = int(self.curTile[1])
                        tmp = []
                        for i in range(-2, 3):
                            tmp.append(color + str(num + i))
                        if tmp[0] in self.hand and tmp[1] in self.hand:
                            self.valid.append(self.OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 3) * 3 + 2)
                        if tmp[1] in self.hand and tmp[3] in self.hand:
                            self.valid.append(self.OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 2) * 3 + 1)
                        if tmp[3] in self.hand and tmp[4] in self.hand:
                            self.valid.append(self.OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 1) * 3)

                self.valid.append(self.OFFSET_ACT['Pass'])
                return self._obs()

        if t[2] == 'Chi':
            tile = t[3]  # 中间牌
            color = tile[0]
            num = int(tile[1])
            self.packs[p].append(('CHI', tile, int(self.curTile[1]) - num + 2))
            self.shownTiles[self.curTile] -= 1

            for i in range(-1, 2):
                self.shownTiles[color + str(num + i)] += 1

            self.wallLast = self.tileWall[(p + 1) % 4] == 0

            # 更新吃牌特征
            self._update_chi(p, tile)

            if p == 0:
                # 自己吃牌
                self.valid = []
                self.hand.append(self.curTile)
                for i in range(-1, 2):
                    self.hand.remove(color + str(num + i))
                self._update_hand()

                for tile in set(self.hand):
                    self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                return self._obs()
            else:
                # 别人吃牌：他们亮出的两张手牌变成已知，从牌山移除
                for i in range(-1, 2):
                    t_tile = color + str(num + i)
                    if t_tile != self.curTile:  # curTile 已在 Play 时移除
                        self._update_wall(t_tile, -1)
                return

        if t[2] == 'UnChi':
            tile = t[3]
            color = tile[0]
            num = int(tile[1])
            self.packs[p].pop()
            self.shownTiles[self.curTile] += 1
            for i in range(-1, 2):
                self.shownTiles[color + str(num + i)] -= 1
            if p == 0:
                for i in range(-1, 2):
                    self.hand.append(color + str(num + i))
                self.hand.remove(self.curTile)
                self._update_hand()
            # 撤销吃牌特征
            if self.chi_count[p] > 0:
                self.chi_count[p] -= 1
                base = self.OFFSET_OBS['CHI'] + p * 4 + self.chi_count[p]
                self.obs[base, :] = 0  # 清除该通道
            return

        if t[2] == 'Peng':
            self.packs[p].append(('PENG', self.curTile, (4 + p - self.tileFrom) % 4))
            self.shownTiles[self.curTile] += 2
            self.wallLast = self.tileWall[(p + 1) % 4] == 0

            # 更新碰牌特征
            self._update_peng(p, self.curTile)

            if p == 0:
                # 自己碰牌
                self.valid = []
                for i in range(2):
                    self.hand.remove(self.curTile)
                self._update_hand()

                for tile in set(self.hand):
                    self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                return self._obs()
            else:
                # 别人碰牌：他们亮出的两张手牌变成已知，从牌山移除
                self._update_wall(self.curTile, -1)
                self._update_wall(self.curTile, -1)
                return

        if t[2] == 'UnPeng':
            self.packs[p].pop()
            self.shownTiles[self.curTile] -= 2
            if p == 0:
                for i in range(2):
                    self.hand.append(self.curTile)
                self._update_hand()
            # 撤销碰牌特征
            base = self.OFFSET_OBS['PENG'] + p
            idx = self.OFFSET_TILE[self.curTile]
            self.obs[base, idx] = 0
            return

        if t[2] == 'Gang':
            self.packs[p].append(('GANG', self.curTile, (4 + p - self.tileFrom) % 4))
            self.shownTiles[self.curTile] += 3

            # 更新杠牌特征
            self._update_gang(p, self.curTile)

            if p == 0:
                for i in range(3):
                    self.hand.remove(self.curTile)
                self._update_hand()
                self.isAboutKong = True
            else:
                # 别人明杠：他们亮出的三张手牌变成已知，从牌山移除
                for _ in range(3):
                    self._update_wall(self.curTile, -1)
            return

        if t[2] == 'AnGang':
            tile = 'CONCEALED' if p else t[3]
            self.packs[p].append(('GANG', tile, 0))

            if p == 0:
                self.isAboutKong = True
                for i in range(4):
                    self.hand.remove(tile)
                self._update_hand()
                # 更新己方暗杠特征
                self._update_angang(tile)
            else:
                self.isAboutKong = False
            return

        if t[2] == 'BuGang':
            tile = t[3]
            for i in range(len(self.packs[p])):
                if tile == self.packs[p][i][1]:
                    self.packs[p][i] = ('GANG', tile, self.packs[p][i][2])
                    break

            self.shownTiles[tile] += 1

            # 补杠: 移除碰牌特征，添加杠牌特征
            # 注意: 不需要更新牌山，因为第4张牌在发牌/摸牌时已经移除
            self._remove_peng(p, tile)
            self._update_gang(p, tile)

            if p == 0:
                self.hand.remove(tile)
                self._update_hand()
                self.isAboutKong = True
                return
            else:
                # 可选: Hu/Pass (抢杠胡)
                self._update_wall(tile, -1)  # 从牌山移除
                self.valid = []
                if self._check_mahjong(tile, isSelfDrawn=False, isAboutKong=True):
                    self.valid.append(self.OFFSET_ACT['Hu'])
                self.valid.append(self.OFFSET_ACT['Pass'])
                return self._obs()

        raise NotImplementedError('Unknown request %s!' % request)

    def action2response(self, action):
        """将动作索引转换为响应字符串"""
        if action < self.OFFSET_ACT['Hu']:
            return 'Pass'
        if action < self.OFFSET_ACT['Play']:
            return 'Hu'
        if action < self.OFFSET_ACT['Chi']:
            return 'Play ' + self.TILE_LIST[action - self.OFFSET_ACT['Play']]
        if action < self.OFFSET_ACT['Peng']:
            t = (action - self.OFFSET_ACT['Chi']) // 3
            return 'Chi ' + 'WTB'[t // 7] + str(t % 7 + 2)
        if action < self.OFFSET_ACT['Gang']:
            return 'Peng'
        if action < self.OFFSET_ACT['AnGang']:
            return 'Gang'
        if action < self.OFFSET_ACT['BuGang']:
            return 'Gang ' + self.TILE_LIST[action - self.OFFSET_ACT['AnGang']]
        return 'BuGang ' + self.TILE_LIST[action - self.OFFSET_ACT['BuGang']]

    def response2action(self, response):
        """将响应字符串转换为动作索引"""
        t = response.split()
        if t[0] == 'Pass':
            return self.OFFSET_ACT['Pass']
        if t[0] == 'Hu':
            return self.OFFSET_ACT['Hu']
        if t[0] == 'Play':
            return self.OFFSET_ACT['Play'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'Chi':
            return self.OFFSET_ACT['Chi'] + 'WTB'.index(t[1][0]) * 7 * 3 + (int(t[2][1]) - 2) * 3 + int(t[1][1]) - int(t[2][1]) + 1
        if t[0] == 'Peng':
            return self.OFFSET_ACT['Peng'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'Gang':
            return self.OFFSET_ACT['Gang'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'AnGang':
            return self.OFFSET_ACT['AnGang'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'BuGang':
            return self.OFFSET_ACT['BuGang'] + self.OFFSET_TILE[t[1]]
        return self.OFFSET_ACT['Pass']

    def _obs(self):
        """返回当前观测"""
        mask = np.zeros(self.ACT_SIZE, dtype=np.float32)
        for a in self.valid:
            mask[a] = 1

        return {
            'observation': self.obs.reshape((self.OBS_SIZE, 4, 9)).copy(),
            'action_mask': mask
        }

    def _check_mahjong(self, winTile, isSelfDrawn=False, isAboutKong=False):
        """检查是否可以和牌"""
        try:
            fans = MahjongFanCalculator(
                pack=tuple(self.packs[0]),
                hand=tuple(self.hand),
                winTile=winTile,
                flowerCount=0,
                isSelfDrawn=isSelfDrawn,
                is4thTile=(self.shownTiles[winTile] + isSelfDrawn) == 4,
                isAboutKong=isAboutKong,
                isWallLast=self.wallLast,
                seatWind=self.seatWind,
                prevalentWind=self.prevalentWind,
                verbose=True
            )
            fanCnt = 0
            for fanPoint, cnt, fanName, fanNameEn in fans:
                fanCnt += fanPoint * cnt
            if fanCnt < 8:
                raise Exception('Not Enough Fans')
        except:
            return False
        return True

    def build_global_obs(self, all_hands):
        """构建全局观测（用于 CTDE Critic）

        Args:
            all_hands: dict {player_id: list of tiles}，所有玩家的手牌
                       player_id 是绝对编号 (0-3)

        Returns:
            global_obs: numpy array of shape (159, 4, 9)
                - [0-146]: 原有观测
                - [147-150]: 下家手牌 (4 channels)
                - [151-154]: 对家手牌 (4 channels)
                - [155-158]: 上家手牌 (4 channels)
        """
        # 创建全局观测，先复制原有观测
        global_obs = np.zeros((self.GLOBAL_OBS_SIZE, 36), dtype=np.float32)
        global_obs[:self.OBS_SIZE, :] = self.obs

        # 添加其他3个玩家的手牌
        # 相对编号：1=下家, 2=对家, 3=上家
        for rel_player in range(1, 4):
            abs_player = (self.seatWind + rel_player) % 4
            hand = all_hands.get(abs_player, [])

            # 统计手牌
            tile_count = defaultdict(int)
            for tile in hand:
                tile_count[tile] += 1

            # 填充特征 (4 channels per player)
            base = self.OBS_SIZE + (rel_player - 1) * 4  # 147, 151, 155
            for tile, count in tile_count.items():
                idx = self.OFFSET_TILE[tile]
                for i in range(count):
                    global_obs[base + i, idx] = 1

        return global_obs.reshape((self.GLOBAL_OBS_SIZE, 4, 9))

    def build_other_hands_obs(self, all_hands):
        """构建其他玩家手牌特征（用于 CTDE，减少存储）

        Args:
            all_hands: dict {player_id: list of tiles}，所有玩家的手牌
                       player_id 是绝对编号 (0-3)

        Returns:
            other_hands_obs: numpy array of shape (12, 4, 9)
                - [0-3]: 下家手牌 (4 channels)
                - [4-7]: 对家手牌 (4 channels)
                - [8-11]: 上家手牌 (4 channels)
        """
        OTHER_HANDS_SIZE = 12  # 3 players × 4 channels
        other_hands_obs = np.zeros((OTHER_HANDS_SIZE, 36), dtype=np.float32)

        # 添加其他3个玩家的手牌
        # 相对编号：1=下家, 2=对家, 3=上家
        for rel_player in range(1, 4):
            abs_player = (self.seatWind + rel_player) % 4
            hand = all_hands.get(abs_player, [])

            # 统计手牌
            tile_count = defaultdict(int)
            for tile in hand:
                tile_count[tile] += 1

            # 填充特征 (4 channels per player)
            base = (rel_player - 1) * 4  # 0, 4, 8
            for tile, count in tile_count.items():
                idx = self.OFFSET_TILE[tile]
                for i in range(count):
                    other_hands_obs[base + i, idx] = 1

        return other_hands_obs.reshape((OTHER_HANDS_SIZE, 4, 9))
