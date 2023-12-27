import minerl
import itertools
import time
import gym
from trainer import Trainer

if __name__ == '__main__':
    env = gym.make('MineRLTreechop-v0')
    trainer = Trainer(env, 'cuda')
    trainer.train(num_episodes=256)    