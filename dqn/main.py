import minerl
import itertools
import time
import gym
from trainer import Trainer
import os
import sys

if __name__ == '__main__':
    env = gym.make('MineRLTreechop-v0')
    path = 'pretrained_models/MineRLTreechop-v0_cnn_pretrained10.pt'
    #path = None
    trainer = Trainer(env, 'cuda')
    trainer.train(num_episodes=1000)    