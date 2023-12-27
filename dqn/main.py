import minerl
import itertools
import time
import gym
from dqn.trainer import Trainer

def step_data(environment='MineRLObtainDiamondDense-v0'):
    d = minerl.data.make(environment)

    # Iterate through batches of data
    for obs, rew, done, act in itertools.islice(d.seq_iter(1, 32), 600):
        print("Act shape:", len(act), act)
        print("Obs shape:", len(obs), [elem for elem in obs.items() if elem[0] != 'pov'])
        print("Rew shape:", len(rew), rew)
        print("Done shape:", len(done), done)
        time.sleep(0.1)

if __name__ == '__main__':
    env = gym.make('MineRLTreechop-v0')
    trainer = Trainer(env, 'cuda')
    trainer.train()    