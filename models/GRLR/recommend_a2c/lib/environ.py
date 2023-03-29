import gym
import gym.spaces
from gym.envs.registration import EnvSpec

import numpy as np
import pandas as pd
import math
import logging
import scipy.sparse as sp
import random

HAS_SELECT_NUMBER = None    #已经选择的项目的个数
ITEM_NUMBER = 1682          #项目的个数
USER_NUMBER = 943          #用户数
FEATURE_CATEGORY = 27     #节点特征种类数
NODE_NUMBER = 2625          #图的节点数

NOISE_RANGE = 500       # 每500轮加入一次噪声
IDX = 0

RANGE = 10

FEATURE_PATH = '../data/net_data/data_set.csv'


logging.basicConfig(level=logging.INFO,
                    filename='../log/all_reward.txt',
                    filemode='w',
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

# 动作空间
class Actions:
    def __init__(self):
        self.n = ITEM_NUMBER
        self.action_space = {}
        self.update_action_space()
    def update_action_space(self):
        for i in range(ITEM_NUMBER):
            self.action_space[i] = i + USER_NUMBER

class State:
    def __init__(self):
        features_pd = pd.read_csv(FEATURE_PATH)
        self.features = np.asarray(features_pd, dtype=np.float32)
        self.is_select = []     # 记录下哪些节点最开始被选择了
        self.action_list = []

    def reset(self):
        '''选择的节点清零'''
        self.action_list = [1340, 962, 1207, 1423, 1169]
        '''获得整体节点特征'''
        features_pd = pd.read_csv(FEATURE_PATH)
        self.features = np.asarray(features_pd, dtype=np.float32)
        '''随机选取一个用户，并修改特征'''
        # self.user_index = random.randint(0, 9)
        self.user_index = 0

        '''记录下当前节点的标签'''
        ua_test = np.asarray(pd.read_csv('../data/net_data/ua_test.csv'))
        self.user_select_lable = []
        for i in range(len(ua_test)):
            if ua_test[i][0] == self.user_index:
                self.user_select_lable.append(ua_test[i][1])
            if ua_test[i][0] > self.user_index:
                break

        # 首先修改当前用户为目标用户的标志，改为1
        self.features[self.user_index][3] = 1

        # 然后修改目标用户选到的节点，置为1，并将评分一并修改
        ua_base = np.asarray(pd.read_csv('../data/net_data/ua_base.csv'))
        for i in range(len(ua_base)):
            if ua_base[i][0] == self.user_index:
                index = ua_base[i][1]
                # 被选标志
                self.features[index][4] = 1
                # 记录下哪些节点被选择了
                self.is_select.append(index)
                # 评分
                self.features[index][5] = ua_base[i][2]
            if ua_base[i][0] > self.user_index:
                break

        # 最后修改所分配的资源
        item_source = np.asarray(pd.read_csv('../data/net_data/all_item_source.csv'))

        for i in range(len(item_source[self.user_index])):
            self.features[i+USER_NUMBER][26] = item_source[self.user_index][i]

        '''暂时先把评分屏蔽了'''
        for i in range(NODE_NUMBER):
            self.features[i][5] = 0

        '''将选中节点的特征值值为零'''
        be_select = self.is_select + self.action_list
        for i in be_select:
            for j in range(FEATURE_CATEGORY):
                self.features[i][j] = 0
            self.features[i][4] = 1
            self.features[i][26] = 0

        return self.features

    def encode(self):
        global IDX
        IDX += 1
        '''每500轮加入噪声'''
        if IDX % NOISE_RANGE == 0:
            '''添加噪声'''
            noise = np.random.normal(loc=0.0, scale=1, size=self.features.shape)
            self.features += noise
            print("add noise")
            # if IDX == NOISE_RANGE:
            #     print("end add noise")
            #     logging.info("end add noise{}".format(IDX))

        for i in self.action_list:
            for j in range(FEATURE_CATEGORY):
                self.features[i][j] = -1 * len(self.action_list)
            self.features[i][4] = 1
            self.features[i][26] = -1

        return normalize(self.features)

    @property
    def shape(self):
        return (NODE_NUMBER, FEATURE_CATEGORY)

def normalize(mx):
    # mx = np.asarray(mx)
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))  #  矩阵行求和
    r_inv = np.power(rowsum, -1).flatten()  # 求和的-1次方
    r_inv[np.isinf(r_inv)] = 0.   # 如果是inf，转换成0
    r_mat_inv = sp.diags(r_inv)  # 构造对角戏矩阵
    mx = r_mat_inv.dot(mx)  # 构造D-1*A，非对称方式，简化方式
    return mx

class RecommendEnv(gym.Env):
    spec = EnvSpec("RecommendEnv-v0")

    def __init__(self):
        self.state = State()
        self.action = Actions()
        self.action_space = gym.spaces.Discrete(n=self.action.n)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=self.state.shape, dtype=np.float32)
        self.seed()

    def step(self, action_idx):

        action = self.action.action_space[action_idx]
        # self.state.is_select.append(action)
        self.state.action_list.append(action)


        '''判断是否结束'''
        if len(self.state.action_list) == RANGE:
            is_done = True
        else:
            is_done = False

        '''更新状态空间'''
        new_state = self.state.encode()

        '''info'''
        info = ""

        '''计算奖励'''
        reward = 0

        if is_done:

            be_selected = set(self.state.action_list).intersection(set(self.state.user_select_lable))
            reward = len(be_selected) / RANGE
            pre = len(be_selected) / RANGE
            if len(self.state.action_list) > len(set(self.state.action_list)):
                reward -= (len(self.state.action_list) - len(set(self.state.action_list))) * 0.2

            info = {
                'action': self.state.action_list,
                'reward': reward,
            }
            logging.info('user:{}, reward:{}, action:{}, be_selected:{}, pre@k;{}'.format(self.state.user_index ,reward, self.state.action_list, be_selected, pre))

        return new_state, reward, is_done, info

    def reset(self):
        '''初始化状态， 返回节点的特征作为初始观察矩阵'''
        observation = self.state.reset()

        # '''将选中节点的特征值值为零'''
        # for i in self.state.is_select:
        #     for j in range(FEATURE_CATEGORY):
        #         observation[i][j] = 0
        #     observation[i][4] = 1

        return normalize(observation)


    def render(self, mode='human'):
        pass

    def close(self):
        super().close()

    def seed(self, seed=None):
        return super().seed(seed)



