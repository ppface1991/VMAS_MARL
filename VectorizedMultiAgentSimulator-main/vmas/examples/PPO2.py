
import argparse
import pickle
from collections import namedtuple

import os
import numpy as np
import matplotlib.pyplot as plt

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import time
from typing import Type

import torch

from vmas import make_env
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy, RandomPolicy
from vmas.simulator.utils import save_video

# Parameters
parser = argparse.ArgumentParser(description='Solve the Wheel with CPPO')
parser.add_argument(
    '--gamma', type=float, default=0.9, metavar='G', help='discount factor (default: 0.9)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', default=True, help='render the environment')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    metavar='N',
    help='interval between training status logs (default: 10)')
args = parser.parse_args()

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'a_log_prob', 'next_state'])
TrainRecord = namedtuple('TrainRecord', ['episode', 'reward'])
global num_state
global num_action
global HETIppo_reward
global cppo_reward
global mappo_reward
global heuristic_reward
HETIppo_reward=[]
cppo_reward=[]
mappo_reward=[]
heuristic_reward=[]

def run_HETIppo(
    scenario_name: str = "transport",
    heuristic: Type[BaseHeuristicPolicy] = RandomPolicy,
    n_steps: int = 200,
    n_envs: int = 32,
    env_kwargs: dict = {},
    render: bool = False,
    save_render: bool = False,
    device: str = "cpu",
):
    env = make_env(
        scenario_name=scenario_name,
        num_envs=n_envs,
        device=device,
        continuous_actions=True,
        wrapper=None,
        # Environment specific variables
        **env_kwargs,
    )
    global num_state
    global num_action
    num_state = env.observation_space[0].shape[0]
    num_action = env.action_space[0].shape[0]
    torch.manual_seed(args.seed)
    env.seed(args.seed)
    agent_num = 2
    agent_all=[]
    for i in range(agent_num):
      agent = HETIPPO()
      agent_all.append(agent)

    training_records = []
    running_reward = -1000
    trans_all=[]
    reward_all=[]
    next_state_all=[]
    for i_epoch in range(400):
        init_time = time.time()
        score = 0
        state = env.reset()
        # if args.render: env.render()
        for t in range(n_steps):
            action, action_log_prob = agent_all[i].select_action(state)
            next_state, reward, done, info = env.step(action)
            for j in range(agent_num):
                trans = Transition(state[j], action[j], reward[j], action_log_prob[j], next_state[j])
                if agent_all[j].store_transition(trans):
                    agent_all[j].update()
            score += sum(reward)/len(reward)
            state = next_state

        running_reward = running_reward * 0.9 + score * 0.1
        HETIppo_reward.append(running_reward.item())
        total_time = time.time() - init_time
        print(
            f"It took: {total_time}s for {n_steps} steps of {n_envs} parallel environments on device {device}\n"
        f"The average total reward was {running_reward}"
        )
        # training_records.append(TrainingRecord(i_epoch, running_reward))

class Policy(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, num_outputs):
        super(Policy, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.ReLU(True),
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.ReLU(True),
            nn.Linear(n_hidden_2, num_outputs)
        )

class Critic(nn.Module):
    def __init__(self, in_dim, n_hidden_1,n_hidden_2):
        super(Critic, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.ReLU(True),
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.ReLU(True),
            nn.Linear(n_hidden_2, 1)
        )

class HETIPPO():
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_epoch = 10
    buffer_capacity = 1000
    batch_size = 2000

    def __init__(self):
        super(HETIPPO, self).__init__()
        self.policy_net = Policy(num_state,64,64,num_action).float()
        self.critic_net = Critic(num_state,64,64).float()
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(num_action)))

        self.actor_optimizer = optim.Adam(self.policy_net.parameters(), 1e-3)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), 4e-3)
        if not os.path.exists('../param'):
            os.makedirs('../param/net_param')
            os.makedirs('../param/img')

    def select_action(self, state):
        action_all=[]
        action_log_prob_all=[]
        for i in range(2):
            state_n = state[i]
            action_mean = self.policy_net.layer(state_n)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            probs = Normal(action_mean, action_std)
            action = probs.sample()
            action = action.clamp(-1,1)
            f_prob=probs.log_prob(action).sum(1), probs.entropy().sum(1)
            action_log_prob_all.append(f_prob)
            action_all.append(action)
        return action_all, action_log_prob_all


    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    # def save_param(self):
    #     torch.save(self.actor_net.state_dict(), '../param/net_param/actor_net'+str(time.time())[:10],+'.pkl')
    #     torch.save(self.critic_net.state_dict(), '../param/net_param/critic_net'+str(time.time())[:10],+'.pkl')

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter+=1
        return self.counter % self.buffer_capacity == 0

    def update(self):
        self.training_step +=1
        state = torch.tensor([t.state.detach().numpy() for t in self.buffer ], dtype=torch.float)
        action = torch.tensor([t.action.detach().numpy() for t in self.buffer], dtype=torch.float).view(-1, 1)
        reward = torch.tensor([t.reward.detach().numpy() for t in self.buffer], dtype=torch.float).view(-1, 1)
        next_state = torch.tensor([t.next_state.detach().numpy() for t in self.buffer], dtype=torch.float)
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1)

        reward = (reward - reward.mean())/(reward.std() + 1e-10)
        with torch.no_grad():
            target_v = reward + args.gamma * self.critic_net.layer(next_state)

        advantage = (target_v - self.critic_net.layer(state)).detach()
        for _ in range(self.ppo_epoch): # iteration ppo_epoch 
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, True):
                # epoch iteration, PPO core!!!
                action_mean = self.policy_net.layer(state[index])
                action_logstd = self.actor_logstd.expand_as(action_mean)
                action_std = torch.exp(action_logstd)
                probs = Normal(action_mean, action_std)
                action_log_prob=probs.log_prob(action[index]).sum(1), probs.entropy().sum(1)
                action_log_prob=torch.tensor([item.cpu().detach().numpy() for item in action_log_prob], dtype=torch.float).view(-1, 1)
                ratio = torch.exp(action_log_prob - old_action_log_prob)
                
                L1 = ratio * advantage[index]
                L2 = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param) * advantage[index]
                action_loss = -torch.min(L1, L2).mean() # MAX->MIN desent
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                value_loss = F.smooth_l1_loss(self.critic_net(state[index]), target_v[index])
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()

        del self.buffer[:]

global mappo_Critic
global mappo_critic_net_optimizer
def run_mappo(
        scenario_name: str = "transport",
        heuristic: Type[BaseHeuristicPolicy] = RandomPolicy,
        n_steps: int = 200,
        n_envs: int = 32,
        env_kwargs: dict = {},
        render: bool = False,
        save_render: bool = False,
        device: str = "cpu",
):
    env = make_env(
        scenario_name=scenario_name,
        num_envs=n_envs,
        device=device,
        continuous_actions=True,
        wrapper=None,
        # Environment specific variables
        **env_kwargs,
    )
    global num_state
    global num_action
    num_state = env.observation_space[0].shape[0]
    num_action = env.action_space[0].shape[0]
    torch.manual_seed(args.seed)
    env.seed(args.seed)
    agent_num = 2
    agent_all = []
    for i in range(agent_num):
        agent = mappo()
        agent_all.append(agent)
    global mappo_Critic
    global mappo_critic_net_optimizer
    mappo_Critic=Critic(num_state, 64, 64).float()
    mappo_critic_net_optimizer = optim.Adam(mappo_Critic.parameters(), 4e-3)
    training_records = []
    running_reward = -1000
    trans_all = []
    reward_all = []
    next_state_all = []
    for i_epoch in range(400):
        init_time = time.time()
        score = 0
        state = env.reset()
        # if args.render: env.render()
        for t in range(n_steps):
            action, action_log_prob = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            for j in range(agent_num):
                trans = Transition(state[j], action[j], reward[j], action_log_prob[j], next_state[j])
                if agent_all[j].store_transition(trans):
                    agent_all[j].update()
            score += sum(reward) / len(reward)
            state = next_state

        running_reward = running_reward * 0.9 + score * 0.1
        mappo_reward.append(running_reward.item())
        total_time = time.time() - init_time
        print(
            f"It took: {total_time}s for {n_steps} steps of {n_envs} parallel environments on device {device}\n"
        f"The average total reward was {running_reward}"
        )
        # training_records.append(TrainingRecord(i_epoch, running_reward))

class mappo():
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_epoch = 10
    buffer_capacity = 1000
    batch_size = 2000

    def __init__(self):
        super(mappo, self).__init__()
        self.policy_net = Policy(num_state, 64, 64, num_action).float()

        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(num_action)))

        self.actor_optimizer = optim.Adam(self.policy_net.parameters(), 1e-3)
        if not os.path.exists('../param'):
            os.makedirs('../param/net_param')
            os.makedirs('../param/img')

    def select_action(self, state):
        action_all = []
        action_log_prob_all = []
        for i in range(2):
            state_n = state[i]
            action_mean = self.policy_net.layer(state_n)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            probs = Normal(action_mean, action_std)
            action = probs.sample()
            action = action.clamp(-1, 1)
            f_prob = probs.log_prob(action).sum(1), probs.entropy().sum(1)
            action_log_prob_all.append(f_prob)
            action_all.append(action)
        return action_all, action_log_prob_all

    def get_value(self, state):
        global mappo_Critic
        global mappo_critic_net_optimizer
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = mappo_Critic(state)
        return value.item()

    # def save_param(self):
    #     torch.save(self.actor_net.state_dict(), '../param/net_param/actor_net'+str(time.time())[:10],+'.pkl')
    #     torch.save(self.critic_net.state_dict(), '../param/net_param/critic_net'+str(time.time())[:10],+'.pkl')

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1
        return self.counter % self.buffer_capacity == 0

    def update(self):
        global mappo_Critic
        global mappo_critic_net_optimizer
        self.training_step += 1
        state = torch.tensor([t.state.detach().numpy() for t in self.buffer], dtype=torch.float)
        action = torch.tensor([t.action.detach().numpy() for t in self.buffer], dtype=torch.float).view(-1,
                                                                                                        1)
        reward = torch.tensor([t.reward.detach().numpy() for t in self.buffer], dtype=torch.float).view(-1,
                                                                                                        1)
        next_state = torch.tensor([t.next_state.detach().numpy() for t in self.buffer], dtype=torch.float)
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1,
                                                                                                        1)

        reward = (reward - reward.mean()) / (reward.std() + 1e-10)
        with torch.no_grad():
            target_v = reward + args.gamma * mappo_Critic.layer(next_state)

        advantage = (target_v - mappo_Critic.layer(state)).detach()
        for _ in range(self.ppo_epoch):  # iteration ppo_epoch
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size,
                                      True):
                # epoch iteration, PPO core!!!
                action_mean = self.policy_net.layer(state[index])
                action_logstd = self.actor_logstd.expand_as(action_mean)
                action_std = torch.exp(action_logstd)
                probs = Normal(action_mean, action_std)
                action_log_prob = probs.log_prob(action[index]).sum(1), probs.entropy().sum(1)
                action_log_prob = torch.tensor([item.cpu().detach().numpy() for item in action_log_prob],
                                               dtype=torch.float).view(-1, 1)
                ratio = torch.exp(action_log_prob - old_action_log_prob)

                L1 = ratio * advantage[index]
                L2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage[index]
                action_loss = -torch.min(L1, L2).mean()  # MAX->MIN desent
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                value_loss = F.smooth_l1_loss(mappo_Critic(state[index]), target_v[index])
                mappo_critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(mappo_Critic.parameters(), self.max_grad_norm)
                mappo_critic_net_optimizer.step()

        del self.buffer[:]

def run_CPPO(
        scenario_name: str = "transport",
        heuristic: Type[BaseHeuristicPolicy] = RandomPolicy,
        n_steps: int = 200,
        n_envs: int = 32,
        env_kwargs: dict = {},
        render: bool = False,
        save_render: bool = False,
        device: str = "cpu",
):
    env = make_env(
        scenario_name=scenario_name,
        num_envs=n_envs,
        device=device,
        continuous_actions=True,
        wrapper=None,
        # Environment specific variables
        **env_kwargs,
    )
    global num_state
    global num_action
    num_state = env.observation_space[0].shape[0]
    num_action = env.action_space[0].shape[0]
    torch.manual_seed(args.seed)
    env.seed(args.seed)
    agent_num = 2
    agent=CPPO()
    training_records = []
    running_reward = -1000
    trans_all = []
    reward_all = []
    next_state_all = []
    for i_epoch in range(400):
        init_time = time.time()
        score = 0
        state = env.reset()
        # if args.render: env.render()
        for t in range(n_steps):
            action, action_log_prob = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            for j in range(agent_num):
                trans = Transition(state[j], action[j], reward[j], action_log_prob[j], next_state[j])
                if agent.store_transition(trans):
                    agent.update()
            score += sum(reward) / len(reward)
            state = next_state

        running_reward = running_reward * 0.9 + score * 0.1
        cppo_reward.append(running_reward.item())
        total_time = time.time() - init_time
        print(
            f"It took: {total_time}s for {n_steps} steps of {n_envs} parallel environments on device {device}\n"
        f"The average total reward was {running_reward}"
        )
        # training_records.append(TrainingRecord(i_epoch, running_reward))

class CPPO():
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_epoch = 10
    buffer_capacity = 1000
    batch_size = 2000

    def __init__(self):
        super(CPPO, self).__init__()
        self.policy_net = Policy(num_state, 64, 64, num_action).float()
        self.critic_net = Critic(num_state, 64, 64).float()
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(num_action)))

        self.actor_optimizer = optim.Adam(self.policy_net.parameters(), 1e-3)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), 4e-3)
        if not os.path.exists('../param'):
            os.makedirs('../param/net_param')
            os.makedirs('../param/img')

    def select_action(self, state):
        action_all = []
        action_log_prob_all = []
        for i in range(2):
            state_n = state[i]
            action_mean = self.policy_net.layer(state_n)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            probs = Normal(action_mean, action_std)
            action = probs.sample()
            action = action.clamp(-1, 1)
            f_prob = probs.log_prob(action).sum(1), probs.entropy().sum(1)
            action_log_prob_all.append(f_prob)
            action_all.append(action)
        return action_all, action_log_prob_all

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    # def save_param(self):
    #     torch.save(self.actor_net.state_dict(), '../param/net_param/actor_net'+str(time.time())[:10],+'.pkl')
    #     torch.save(self.critic_net.state_dict(), '../param/net_param/critic_net'+str(time.time())[:10],+'.pkl')

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1
        return self.counter % self.buffer_capacity == 0

    def update(self):
        self.training_step += 1
        state = torch.tensor([t.state.detach().numpy() for t in self.buffer], dtype=torch.float)
        action = torch.tensor([t.action.detach().numpy() for t in self.buffer], dtype=torch.float).view(-1,
                                                                                                        1)
        reward = torch.tensor([t.reward.detach().numpy() for t in self.buffer], dtype=torch.float).view(-1,
                                                                                                        1)
        next_state = torch.tensor([t.next_state.detach().numpy() for t in self.buffer], dtype=torch.float)
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1,
                                                                                                        1)

        reward = (reward - reward.mean()) / (reward.std() + 1e-10)
        with torch.no_grad():
            target_v = reward + args.gamma * self.critic_net.layer(next_state)

        advantage = (target_v - self.critic_net.layer(state)).detach()
        for _ in range(self.ppo_epoch):  # iteration ppo_epoch
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size,
                                      True):
                # epoch iteration, PPO core!!!
                action_mean = self.policy_net.layer(state[index])
                action_logstd = self.actor_logstd.expand_as(action_mean)
                action_std = torch.exp(action_logstd)
                probs = Normal(action_mean, action_std)
                action_log_prob = probs.log_prob(action[index]).sum(1), probs.entropy().sum(1)
                action_log_prob = torch.tensor([item.cpu().detach().numpy() for item in action_log_prob],
                                               dtype=torch.float).view(-1, 1)
                ratio = torch.exp(action_log_prob - old_action_log_prob)

                L1 = ratio * advantage[index]
                L2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage[index]
                action_loss = -torch.min(L1, L2).mean()  # MAX->MIN desent
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                value_loss = F.smooth_l1_loss(self.critic_net(state[index]), target_v[index])
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()

        del self.buffer[:]

def run_heuristic(
        scenario_name: str = "transport",
        heuristic: Type[BaseHeuristicPolicy] = RandomPolicy,
        n_steps: int = 200,
        n_envs: int = 32,
        env_kwargs: dict = {},
        render: bool = False,
        save_render: bool = False,
        device: str = "cpu",
):

    assert not (save_render and not render), "To save the video you have to render it"

    # Scenario specific variables
    policy = heuristic(continuous_action=True)

    env = make_env(
        scenario_name=scenario_name,
        num_envs=n_envs,
        device=device,
        continuous_actions=True,
        wrapper=None,
        # Environment specific variables
        **env_kwargs,
    )

    frame_list = []  # For creating a gif
    init_time = time.time()
    step = 0
    obs = env.reset()
    total_reward = 0
    for i_epoch in range(400):
        for s in range(n_steps):
            step += 1
            actions = [None] * len(obs)
            for i in range(len(obs)):
                actions[i] = policy.compute_action(obs[i], u_range=env.agents[i].u_range)
            obs, rews, dones, info = env.step(actions)
            rewards = torch.stack(rews, dim=1)
            global_reward = rewards.mean(dim=1)
            mean_global_reward = global_reward.mean(dim=0)
            total_reward += mean_global_reward
        heuristic_reward.append(total_reward)
    total_time = time.time() - init_time
    if render and save_render:
        save_video(scenario_name, frame_list, 1 / env.scenario.world.dt)

    print(
        f"It took: {total_time}s for {n_steps} steps of {n_envs} parallel environments on device {device}\n"
    f"The average total reward was {total_reward}"
    )
def plot_all_figure():
    plt.figure(figsize=(50, 50), dpi=100)
    x_axis_data = np.linspace(0, 400, 400)
    plt.plot(x_axis_data, cppo_reward, alpha=0.8, linewidth=1, label='cppo')
    plt.plot(x_axis_data, mappo_reward, alpha=0.8, linewidth=1, label='mappo')
    plt.plot(x_axis_data, HETIppo_reward, alpha=0.8, linewidth=1, label='HETIppo')
    plt.plot(x_axis_data, heuristic_reward, alpha=0.8, linewidth=1, label='Heuristic')
    plt.legend(loc="upper right")
    plt.xlabel('Training iteration')
    plt.ylabel('Episode reward mean')
    plt.show()

if __name__ == '__main__':
    from vmas.scenarios.transport import HeuristicPolicy as TransportHeuristic
    run_CPPO(
        scenario_name="transport",
        heuristic=1,
        n_envs=1,
        n_steps=200,
        render=True,
        save_render=False,
    )
    run_HETIppo(
        scenario_name="transport",
        heuristic=1,
        n_envs=1,
        n_steps=200,
        render=True,
        save_render=False,
    )
    run_heuristic(
        scenario_name="transport",
        heuristic=TransportHeuristic,
        n_envs=1,
        n_steps=200,
        render=True,
        save_render=False,
    )
    run_mappo(
        scenario_name="transport",
        heuristic=1,
        n_envs=1,
        n_steps=200,
        render=True,
        save_render=False,
    )
    plot_all_figure()