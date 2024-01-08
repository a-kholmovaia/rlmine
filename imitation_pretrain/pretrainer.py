import minerl
import sys 
import os
import torch
from torch import nn
from tqdm import tqdm
import gym
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dqn'))
from dqn import DQN
from preprocess_observations import process_states_batch, parse_action2ind

class Pretrainer:
    def __init__(
            self, env_name="MineRLTreechop-v0", device='cuda'):
        minerl.data.download(directory='data', environment=env_name)
        self.env_name = env_name
        self.data = minerl.data.make(env_name, data_dir='data', num_workers=1)
        self.env = gym.make(env_name)
        self.device = device
        self.model = DQN(1, 9).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, epochs=6, batch_size=32):
        step = 0
        losses = []
        for state, action, _, _, _ \
                in tqdm(self.data.batch_iter(num_epochs=epochs, batch_size=batch_size, seq_len=1)):
            # Get pov observations
            obs = torch.tensor(
                process_states_batch(state), dtype=torch.float32, device=self.device
                ).unsqueeze(0).transpose(0, 1) #to get batch_sizex1x64x64

            # Translate batch of actions for the ActionShaping wrapper
            actions = torch.tensor(parse_action2ind(self.env, action, batch_size),
                                    ).type(torch.LongTensor).to(self.device)
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
            if (step % 200) == 0:
                mean_loss = sum(losses) / len(losses)
                tqdm.write(f'Step {step:>5} | Training loss = {mean_loss:.3f}')
                losses.clear()
        torch.save(self.model.state_dict(), self.env_name + '_cnn_pretrained.pt')