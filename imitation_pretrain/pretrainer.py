import minerl
import sys 
import os
import torch
from torch import nn
import tqdm
import gym
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dqn'))
from dqn import DQN
from dqn.preprocess_observations import process_state, parse_action2ind

class Pretrainer:
    def __init__(
            self, env_name="MineRLTreechop-v0", device='cpu'):
        minerl.data.download(directory='data', environment=env_name)
        self.env_name = env_name
        self.data = minerl.data.make(env_name, data_dir='data', num_workers=2)
        self.env = gym.make(env_name)
        self.device = device
        self.model = DQN(1, 9).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        step = 0
        losses = []
        for state, action, _, _, _ \
                in tqdm(self.data.batch_iter(num_epochs=6, batch_size=32, seq_len=1)):
            # Get pov observations
            obs = torch.tensor(
                process_state(state), dtype=torch.float32, device=self.device
                ).unsqueeze(0).to(self.device)

            # Translate batch of actions for the ActionShaping wrapper
            actions = torch.tensor(parse_action2ind(self.env, action),
                                    dtype=torch.float32, device=self.device
                                    ).unsqueeze(0).to(self.device)

            # Remove samples with no corresponding action
            # mask = actions != -1
            # obs = obs[mask]
            # actions = actions[mask]

            # Update weights with backprop
            logits = self.model(obs)
            loss = self.criterion(logits, actions)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Print loss
            step += 1
            losses.append(loss.item())
            if (step % 2000) == 0:
                mean_loss = sum(losses) / len(losses)
                tqdm.write(f'Step {step:>5} | Training loss = {mean_loss:.3f}')
                losses.clear()
        torch.save(self.model.state_dict(), self.env_name + 'pretrained')