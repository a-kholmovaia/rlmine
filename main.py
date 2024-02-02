# RL environment
import gym
import minerl
from gym.wrappers import Monitor
# Others
import numpy as np
from tqdm.notebook import tqdm
import torch
from dqn.dqn import DQN
import logging
from dqn.preprocess_observations import process_state, parse_action_ind2dict
logging.disable(logging.ERROR)

if __name__ == '__main__':
    PATH = 'imitation_pretrain/MineRLTreechop-v0_cnn_pretrained.pt'
    # Define the sequence of actions
    model = DQN(1,9).to('cuda')
    model.load_state_dict(torch.load(PATH))

    env = gym.make('MineRLTreechop-v0')

    env = Monitor(env, 'videos', force=True)

    env.seed(21)
    obs = env.reset()
    cum_reward = 0
    while True:
        env.render()
        obs = process_state(obs).to('cuda')
        action_index = model(obs).max(1).indices.view(1, 1)
        action = parse_action_ind2dict(env, action_index)

        # Update the environment with the new action space
        obs, reward, done, _ = env.step(action)
        cum_reward += reward
        print(f'Reward: {cum_reward}')
        if done:
            print('\n\ndone\n\n')
            env.env.close()
            break
            