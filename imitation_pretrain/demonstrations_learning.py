import minerl
import sys 
import os
import torch
from torch import nn
from tqdm import tqdm
import gym
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dqn'))
from dqn import DQN
from preprocess_observations import process_states_batch, parse_action2ind, process_state
import pandas as pd
class Pretrainer:
    def __init__(
            self, env_name="MineRLTreechop-v0", device='cuda'):
        minerl.data.download(directory='data', environment=env_name)
        self.env_name = env_name
        self.data = minerl.data.make(env_name, data_dir='data', num_workers=1)
        self.env = gym.make(env_name)
        self.device = device
        self.n_actions = len(self.env.action_space.noop().keys())
        print(f'Number of actions: {self.n_actions}')

        state = process_state(self.env.reset())
        print(f'Shape of observations: {state[0].shape}')
        self.model = DQN(self.env, list(state[0].shape), self.n_actions).to(self.device)
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
                )#.unsqueeze(0).transpose(0, 1) #to get batch_sizex1x64x64

            # Translate batch of actions for the ActionShaping wrapper
            actions = torch.tensor(parse_action2ind(self.env, action, batch_size),
                                    ).type(torch.LongTensor).to(self.device)

            # Update weights with backprop
            logits = self.model(obs)
            loss = self.criterion(logits, actions)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Print loss
            step += 1
            losses.append(loss.item())
            if (step % 1_000) == 0:
                mean_loss = sum(losses[-1_000:]) / 1_000
                tqdm.write(f'Step {step:>5} | Training loss = {mean_loss:.3f}')
        meta = pd.DataFrame(
            {
                'loss': losses
             }
            )
        meta.to_csv(f'meta_loss_{epochs}.csv')
        torch.save(self.model.state_dict(), self.env_name + '_cnn_pretrained.pt')
        print('Model saved')