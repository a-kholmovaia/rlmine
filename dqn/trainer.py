from gym import Env
import torch
import torch.optim as optim
from dqn import DQN
from replay_memory import ReplayMemory
import random 
import math 
from torch import nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from itertools import count
from preprocess_observations import process_state, parse_action_ind2dict
import pandas as pd
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
class Trainer:
    BATCH_SIZE = 64
    GAMMA = 0.99
    TAU = 0.005
    LR = 1e-4
    def __init__(self, env: Env, path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Get number of actions from gym action space
        self.n_actions = len(env.action_space.noop().keys())
        print(env.action_space.noop())
        print(f'Number of actions: {self.n_actions}')
        self.env = env

        state = process_state(self.env.reset())
        print(f'Shape of observations: {state[0].shape}')
        self.policy_net = DQN(env, list(state[0].shape), self.n_actions).to(self.device)
        self.target_net = DQN(env, list(state[0].shape), self.n_actions).to(self.device)
        if path is not None:
            for target_param, param in zip(self.policy_net.parameters(), self.target_net.parameters()):
                target_param.data.copy_(param)
        #self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.LR)
        self.memory = ReplayMemory(10_000)

        self.steps_done = 0
        

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(self.BATCH_SIZE)
    
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.stack(rewards).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        actions = actions.view(actions.size(0))
        dones = dones.view(dones.size(0))

        curr_Q = self.policy_net.forward(states.squeeze()).gather(1, actions.view(actions.size(0), 1))
        next_Q = self.target_net.forward(next_states.squeeze())
        max_next_Q = torch.max(next_Q, 1)[0]
        max_next_Q = max_next_Q.view(max_next_Q.size(0), 1)
        expected_Q = rewards + (1 - dones) * self.GAMMA * max_next_Q

        loss = F.mse_loss(curr_Q, expected_Q.detach())
        print(f'Loss: {loss.item()}')
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
    
    
    def train(self, num_episodes=5000):
        cum_rewards = []
        episode_durations = []
        for i_episode in range(num_episodes):
            reward_episode = 0
            # Initialize the environment and get it's state
            state = self.env.reset()
            state = process_state(state).to(self.device)
            for t in count():
                action = self.policy_net.get_action(state, self.steps_done)
                self.steps_done += 1

                observation, reward, done, _ = self.env.step(
                                                                parse_action_ind2dict(
                                                                    self.env, action.item()
                                                                    )
                                                            )
                reward_episode += reward
                reward = torch.tensor([reward], device=self.device)
                state = torch.tensor(state, device=self.device)

                next_state = process_state(observation).to(self.device)
                done = torch.tensor([done], device=self.device)

                # Store  in memory
                self.memory.push(state, action, reward, next_state, done)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
                self.target_net.load_state_dict(target_net_state_dict)
                print(f'Episode: {i_episode}, Step: {t}, Reward: {reward_episode}')
                if done or t > 10_000:
                    cum_rewards.append(reward_episode)
                    episode_durations.append(t)
                    break
        print('Complete')
        meta = pd.DataFrame(
            {
                'cum_rewards': cum_rewards, 
                'episode_durations': episode_durations
             }
            )
        meta.to_csv('meta_dqn_10ms.csv')
        torch.save(self.target_net.state_dict(), 'treechop_dqn_10ms.pt')
