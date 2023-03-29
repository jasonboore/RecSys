import gym
import ptan
import numpy as np
import argparse
import pandas as pd
import datetime
import logging

from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GATConv, GCNConv

import environ

import common

SOURCE_TARGET_PATH = '../data/net_data/ua_base_edges.csv'

data_set = pd.read_csv(SOURCE_TARGET_PATH)
edges = np.asarray(data_set, np.int64)
edges = edges.T



GAMMA = 1.0
LEARNING_RATE = 0.003
ENTROPY_BETA = 0.01
BATCH_SIZE = 64
NUM_ENVS = 70

REWARD_STEPS = 5
CLIP_GRAD = 0.1

class SocialA2C(nn.Module):
    def __init__(self, n_features, n_actions, hidden=32):
        super(SocialA2C, self).__init__()
        self.device = torch.device('cuda')
        self.edges = torch.LongTensor(edges)
        self.edges_gpu = self.edges.to(self.device)
        self.gat1 = GCNConv(n_features[1], hidden)
        self.gat2 = GCNConv(hidden, hidden)

        # self.conv = nn.Sequential(
        #     GATConv(n_features[1], hidden),
        #     nn.ReLU(),
        #     GATConv(hidden, hidden),
        #     nn.ReLU()
        # )


        conv_out_size = self._get_conv_out(n_features)

        self.policy = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        zeros = torch.zeros(*shape)
        features = self.gat1(zeros, self.edges)
        features = F.relu(features)
        features = self.gat2(features, self.edges)
        o = F.relu(features)
        # o = self.conv(zeros, self.edges)
        return int(np.prod(o.size()))

    def forward(self, features):
        # features = features.squeeze(dim=0)
        # features = torch.tensor(features, dtype=torch.double)
        # features = torch.double()
        features = self.gat1(features, self.edges_gpu)
        features = F.relu(features)
        features = self.gat2(features, self.edges_gpu)
        features = F.relu(features)
        # conv_out = features.view(1, -1)
        conv_out = features.view(features.size()[0], -1)
        # conv_out = self.conv(features, self.edges).view(features.size()[0], -1)
        policy = self.policy(conv_out)
        # '''加入高斯噪声'''
        # if IDX <= 1000:
        #     if IDX == 1000:
        #         print("End Add noise")
        #     noise = np.random.normal(loc=0.0, scale=0.1, size=policy.shape)
        #     noise = torch.tensor(noise).to(policy.device)
        #     policy += noise
        value = self.value(conv_out)
        return policy, value


def unpack_batch(batch, net, device='cpu'):
    """
    Convert batch into training tensors
    :param batch:
    :param net:
    :return: states variable, actions tensor, reference values variable
    """
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []
    for idx, exp in enumerate(batch):
        states.append(np.array(exp.state, copy=False))
        actions.append(int(exp.action))
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(np.array(exp.last_state, copy=False))

    states_v = torch.FloatTensor(
        np.array(states, copy=False)).to(device)
    actions_t = torch.LongTensor(actions).to(device)

    # handle rewards
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_states_v = torch.FloatTensor(np.array(last_states, copy=False)).to(device)
        last_vals_v = net(last_states_v)[1]
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        last_vals_np *= GAMMA ** REWARD_STEPS
        rewards_np[not_done_idx] += last_vals_np

    ref_vals_v = torch.FloatTensor(rewards_np).to(device)

    return states_v, actions_t, ref_vals_v

if __name__ == "__main__":
    VALUE_THRESHOLD = 0.4

    device = torch.device("cuda")

    # env = environ.SocialEnv()
    # env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)

    envs = []
    for i in range(NUM_ENVS):
        envs.append(environ.RecommendEnv())

    # make_env = lambda: ptan.common.wrappers.wrap_dqn(env)
    # envs = [make_env() for _ in range(NUM_ENVS)]

    writer = SummaryWriter(comment='-RecommendNetwork')

    net = SocialA2C(envs[0].observation_space.shape, envs[0].action_space.n, hidden=32).to(device)
    # '''加载模型的方式'''
    # net = torch.load('../model/a2c.pth')
    print(net)

    agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], apply_softmax=True, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma=GAMMA, steps_count=REWARD_STEPS)
    # exp_source = ptan.experience.ExperienceSourceFirstLast(envs[0], agent, gamma=GAMMA, steps_count=REWARD_STEPS)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)

    batch = []

    with common.RewardTracker(writer, stop_reward=0.4) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
            for step_idx, exp in enumerate(exp_source):
                batch.append(exp)

                # 处理新奖励
                new_rewards = exp_source.pop_total_rewards()
                if new_rewards:
                    if tracker.reward(new_rewards[0], step_idx):
                        '''保存模型'''
                        torch.save(net, '../model/a2c.pth')
                        break

                if len(batch) < BATCH_SIZE:
                    # print(len(batch))
                    continue

                states_v, actions_t, vals_ref_v = unpack_batch(batch, net, device=device)
                batch.clear()

                if vals_ref_v[-1] > VALUE_THRESHOLD:
                    VALUE_THRESHOLD += 0.2
                    time_str = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                    model_name = '../model/' + time_str + '.pth'
                    torch.save(net, model_name)

                optimizer.zero_grad()
                logits_v, value_v = net(states_v)
                loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)

                log_prob_v = F.log_softmax(logits_v, dim=1)
                adv_v = vals_ref_v - value_v.detach()
                log_prob_actions_v = adv_v * log_prob_v[range(BATCH_SIZE), actions_t]
                loss_policy_v = -log_prob_actions_v.mean()

                prob_v = F.softmax(logits_v, dim=1)
                entropy_loss_v = ENTROPY_BETA * (prob_v * log_prob_v).sum(dim=1).mean()

                # calculate policy gradients only
                loss_policy_v.backward(retain_graph=True)
                grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                        for p in net.parameters()
                                        if p.grad is not None])

                # apply entropy and value gradients
                loss_v = entropy_loss_v + loss_value_v
                loss_v.backward()
                nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
                optimizer.step()
                # get full loss
                loss_v += loss_policy_v

                tb_tracker.track("advantage", adv_v, step_idx)
                tb_tracker.track("values", value_v, step_idx)
                tb_tracker.track("batch_rewards", vals_ref_v, step_idx)
                tb_tracker.track("loss_entropy", entropy_loss_v, step_idx)
                tb_tracker.track("loss_policy", loss_policy_v, step_idx)
                tb_tracker.track("loss_value", loss_value_v, step_idx)
                tb_tracker.track("loss_total", loss_v, step_idx)
                tb_tracker.track("grad_l2", np.sqrt(np.mean(np.square(grads))), step_idx)
                tb_tracker.track("grad_max", np.max(np.abs(grads)), step_idx)
                tb_tracker.track("grad_var", np.var(grads), step_idx)












