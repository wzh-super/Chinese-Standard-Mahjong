"""
Botzone 提交用单文件，合并了所有依赖
"""

import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from collections import defaultdict

try:
    from MahjongGB import MahjongFanCalculator
except:
    print('MahjongGB library required!')
    raise


# ==================== Model ====================

class ResBlock(nn.Module):
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
    def __init__(self, num_res_blocks=8, channels=128, in_channels=147):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, channels, 3, 1, 1, bias=False)
        self.input_bn = nn.BatchNorm2d(channels)
        self.res_blocks = nn.ModuleList([ResBlock(channels) for _ in range(num_res_blocks)])
        self._logits = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 4 * 9, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 235)
        )
        self._value_branch = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * 4 * 9, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 1)
        )

    def forward(self, input_dict):
        obs = input_dict["observation"].float()
        x = F.relu(self.input_bn(self.input_conv(obs)))
        for res_block in self.res_blocks:
            x = res_block(x)
        logits = self._logits(x)
        value = self._value_branch(x)
        mask = input_dict["action_mask"].float()
        inf_mask = torch.clamp(torch.log(mask), -1e38, 1e38)
        masked_logits = logits + inf_mask
        return masked_logits, value


# ==================== Agent ====================

class FeatureAgentV2:
    OBS_SIZE = 147
    ACT_SIZE = 235
    observation_space = (OBS_SIZE, 4, 9)
    action_space = ACT_SIZE

    OFFSET_OBS = {
        'SEAT_WIND': 0, 'PREVALENT_WIND': 1, 'HAND': 2, 'CHI': 6,
        'PENG': 22, 'GANG': 26, 'ANGANG': 30, 'DISCARD': 31, 'WALL': 143,
    }

    OFFSET_ACT = {
        'Pass': 0, 'Hu': 1, 'Play': 2, 'Chi': 36,
        'Peng': 99, 'Gang': 133, 'AnGang': 167, 'BuGang': 201
    }

    TILE_LIST = [
        *('W%d' % (i + 1) for i in range(9)),
        *('T%d' % (i + 1) for i in range(9)),
        *('B%d' % (i + 1) for i in range(9)),
        *('F%d' % (i + 1) for i in range(4)),
        *('J%d' % (i + 1) for i in range(3))
    ]
    OFFSET_TILE = {c: i for i, c in enumerate(TILE_LIST)}

    def __init__(self, seatWind):
        self.seatWind = seatWind
        self.packs = [[] for _ in range(4)]
        self.history = [[] for _ in range(4)]
        self.tileWall = [21] * 4
        self.shownTiles = defaultdict(int)
        self.wallLast = False
        self.isAboutKong = False
        self.chi_count = [0, 0, 0, 0]
        self.discard_count = [0, 0, 0, 0]
        self.obs = np.zeros((self.OBS_SIZE, 36), dtype=np.float32)
        self.obs[self.OFFSET_OBS['SEAT_WIND']][self.OFFSET_TILE['F%d' % (self.seatWind + 1)]] = 1
        self.obs[self.OFFSET_OBS['WALL']:self.OFFSET_OBS['WALL'] + 4, :] = 1

    def _update_wall(self, tile, delta=-1):
        index = self.OFFSET_TILE[tile]
        base = self.OFFSET_OBS['WALL']
        if delta < 0:
            for i in range(3, -1, -1):
                if self.obs[base + i, index] == 1:
                    self.obs[base + i, index] = 0
                    break
        else:
            for i in range(4):
                if self.obs[base + i, index] == 0:
                    self.obs[base + i, index] = 1
                    break

    def _update_hand(self):
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
        if self.chi_count[player] >= 4:
            return
        base = self.OFFSET_OBS['CHI'] + player * 4 + self.chi_count[player]
        color = chi_tile[0]
        num = int(chi_tile[1])
        for i in range(-1, 2):
            tile = color + str(num + i)
            idx = self.OFFSET_TILE[tile]
            self.obs[base, idx] = 1
        self.chi_count[player] += 1

    def _update_peng(self, player, tile):
        base = self.OFFSET_OBS['PENG'] + player
        idx = self.OFFSET_TILE[tile]
        self.obs[base, idx] = 1

    def _remove_peng(self, player, tile):
        base = self.OFFSET_OBS['PENG'] + player
        idx = self.OFFSET_TILE[tile]
        self.obs[base, idx] = 0

    def _update_gang(self, player, tile):
        base = self.OFFSET_OBS['GANG'] + player
        idx = self.OFFSET_TILE[tile]
        self.obs[base, idx] = 1

    def _update_angang(self, tile):
        base = self.OFFSET_OBS['ANGANG']
        idx = self.OFFSET_TILE[tile]
        self.obs[base, idx] = 1

    def _update_discard(self, player, tile):
        if self.discard_count[player] >= 28:
            return
        base = self.OFFSET_OBS['DISCARD'] + player * 28 + self.discard_count[player]
        idx = self.OFFSET_TILE[tile]
        self.obs[base, idx] = 1
        self.discard_count[player] += 1

    def request2obs(self, request):
        t = request.split()

        if t[0] == 'Wind':
            self.prevalentWind = int(t[1])
            self.obs[self.OFFSET_OBS['PREVALENT_WIND']][self.OFFSET_TILE['F%d' % (self.prevalentWind + 1)]] = 1
            return

        if t[0] == 'Deal':
            self.hand = t[1:]
            self._update_hand()
            for tile in self.hand:
                self._update_wall(tile, -1)
            return

        if t[0] == 'Huang':
            self.valid = []
            return self._obs()

        if t[0] == 'Draw':
            self.tileWall[0] -= 1
            self.wallLast = self.tileWall[1] == 0
            tile = t[1]
            self.valid = []
            if self._check_mahjong(tile, isSelfDrawn=True, isAboutKong=self.isAboutKong):
                self.valid.append(self.OFFSET_ACT['Hu'])
            self.isAboutKong = False
            self.hand.append(tile)
            self._update_hand()
            self._update_wall(tile, -1)
            for tile in set(self.hand):
                self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                if self.hand.count(tile) == 4 and not self.wallLast and self.tileWall[0] > 0:
                    self.valid.append(self.OFFSET_ACT['AnGang'] + self.OFFSET_TILE[tile])
            if not self.wallLast and self.tileWall[0] > 0:
                for packType, tile, offer in self.packs[0]:
                    if packType == 'PENG' and tile in self.hand:
                        self.valid.append(self.OFFSET_ACT['BuGang'] + self.OFFSET_TILE[tile])
            return self._obs()

        p = (int(t[1]) + 4 - self.seatWind) % 4

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
            self._update_discard(p, self.curTile)
            if p != 0:
                self._update_wall(self.curTile, -1)
            if p == 0:
                self.hand.remove(self.curTile)
                self._update_hand()
                return
            else:
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
            tile = t[3]
            color = tile[0]
            num = int(tile[1])
            self.packs[p].append(('CHI', tile, int(self.curTile[1]) - num + 2))
            self.shownTiles[self.curTile] -= 1
            for i in range(-1, 2):
                self.shownTiles[color + str(num + i)] += 1
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            self._update_chi(p, tile)
            if p == 0:
                self.valid = []
                self.hand.append(self.curTile)
                for i in range(-1, 2):
                    self.hand.remove(color + str(num + i))
                self._update_hand()
                for tile in set(self.hand):
                    self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                return self._obs()
            else:
                for i in range(-1, 2):
                    t_tile = color + str(num + i)
                    if t_tile != self.curTile:
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
            if self.chi_count[p] > 0:
                self.chi_count[p] -= 1
                base = self.OFFSET_OBS['CHI'] + p * 4 + self.chi_count[p]
                self.obs[base, :] = 0
            return

        if t[2] == 'Peng':
            self.packs[p].append(('PENG', self.curTile, (4 + p - self.tileFrom) % 4))
            self.shownTiles[self.curTile] += 2
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            self._update_peng(p, self.curTile)
            if p == 0:
                self.valid = []
                for i in range(2):
                    self.hand.remove(self.curTile)
                self._update_hand()
                for tile in set(self.hand):
                    self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                return self._obs()
            else:
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
            base = self.OFFSET_OBS['PENG'] + p
            idx = self.OFFSET_TILE[self.curTile]
            self.obs[base, idx] = 0
            return

        if t[2] == 'Gang':
            self.packs[p].append(('GANG', self.curTile, (4 + p - self.tileFrom) % 4))
            self.shownTiles[self.curTile] += 3
            self._update_gang(p, self.curTile)
            if p == 0:
                for i in range(3):
                    self.hand.remove(self.curTile)
                self._update_hand()
                self.isAboutKong = True
            else:
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
            self._remove_peng(p, tile)
            self._update_gang(p, tile)
            if p == 0:
                self.hand.remove(tile)
                self._update_hand()
                self.isAboutKong = True
                return
            else:
                self._update_wall(tile, -1)
                self.valid = []
                if self._check_mahjong(tile, isSelfDrawn=False, isAboutKong=True):
                    self.valid.append(self.OFFSET_ACT['Hu'])
                self.valid.append(self.OFFSET_ACT['Pass'])
                return self._obs()

        raise NotImplementedError('Unknown request %s!' % request)

    def action2response(self, action):
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

    def _obs(self):
        mask = np.zeros(self.ACT_SIZE, dtype=np.float32)
        for a in self.valid:
            mask[a] = 1
        return {
            'observation': self.obs.reshape((self.OBS_SIZE, 4, 9)).copy(),
            'action_mask': mask
        }

    def _check_mahjong(self, winTile, isSelfDrawn=False, isAboutKong=False):
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


# ==================== Main ====================

def obs2response(model, obs):
    logits, value = model({
        'observation': torch.from_numpy(np.expand_dims(obs['observation'], 0)),
        'action_mask': torch.from_numpy(np.expand_dims(obs['action_mask'], 0))
    })
    action = logits.detach().numpy().flatten().argmax()
    response = agent.action2response(action)
    return response


if __name__ == '__main__':
    model = CNNModel()
    data_dir = '/data/epoch_8.pkl'
    model.load_state_dict(torch.load(data_dir, map_location=torch.device('cpu')))
    model.train(False)

    # # 初始化变量，避免未定义错误
    # angang = None
    # zimo = False

    input()  # 1
    while True:
        request = input()
        while not request.strip():
            request = input()
        request = request.split()

        if request[0] == '0':
            seatWind = int(request[1])
            agent = FeatureAgentV2(seatWind)
            agent.request2obs('Wind %s' % request[2])
            print('PASS')

        elif request[0] == '1':
            agent.request2obs(' '.join(['Deal', *request[5:]]))
            print('PASS')

        elif request[0] == '2':
            obs = agent.request2obs('Draw %s' % request[1])
            response = obs2response(model, obs)
            response = response.split()
            if response[0] == 'Hu':
                print('HU')
            elif response[0] == 'Play':
                print('PLAY %s' % response[1])
            elif response[0] == 'Gang':
                print('GANG %s' % response[1])
                angang = response[1]
            elif response[0] == 'BuGang':
                print('BUGANG %s' % response[1])

        elif request[0] == '3':
            p = int(request[1])
            if request[2] == 'DRAW':
                agent.request2obs('Player %d Draw' % p)
                zimo = True
                print('PASS')
            elif request[2] == 'GANG':
                if p == seatWind and angang:
                    agent.request2obs('Player %d AnGang %s' % (p, angang))
                elif zimo:
                    agent.request2obs('Player %d AnGang' % p)
                else:
                    agent.request2obs('Player %d Gang' % p)
                print('PASS')
            elif request[2] == 'BUGANG':
                obs = agent.request2obs('Player %d BuGang %s' % (p, request[3]))
                if p == seatWind:
                    print('PASS')
                else:
                    response = obs2response(model, obs)
                    if response == 'Hu':
                        print('HU')
                    else:
                        print('PASS')
            else:
                zimo = False
                if request[2] == 'CHI':
                    agent.request2obs('Player %d Chi %s' % (p, request[3]))
                elif request[2] == 'PENG':
                    agent.request2obs('Player %d Peng' % p)
                obs = agent.request2obs('Player %d Play %s' % (p, request[-1]))
                if p == seatWind:
                    print('PASS')
                else:
                    response = obs2response(model, obs)
                    response = response.split()
                    if response[0] == 'Hu':
                        print('HU')
                    elif response[0] == 'Pass':
                        print('PASS')
                    elif response[0] == 'Gang':
                        print('GANG')
                        angang = None
                    elif response[0] in ('Peng', 'Chi'):
                        obs = agent.request2obs('Player %d ' % seatWind + ' '.join(response))
                        response2 = obs2response(model, obs)
                        print(' '.join([response[0].upper(), *response[1:], response2.split()[-1]]))
                        agent.request2obs('Player %d Un' % seatWind + ' '.join(response))

        print('>>>BOTZONE_REQUEST_KEEP_RUNNING<<<')
        sys.stdout.flush()
