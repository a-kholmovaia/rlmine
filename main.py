# RL environment
import gym
import minerl
from gym.wrappers import Monitor
# Others
import numpy as np
from tqdm.notebook import tqdm
import logging
logging.disable(logging.ERROR)

if __name__ == '__main__':
    # Define the sequence of actions
    script = ['forward'] * 10 + [''] * 40 + ['jump'] * 1 + ['back'] * 15

    env = gym.make('MineRLTreechop-v0')

    env = Monitor(env, 'videos', force=True)

    env.seed(21)
    obs = env.reset()

    for action in script:
        env.render()
        # Get the action space (dict of possible actions)
        action_space = env.action_space.noop()

        # Activate the selected action in the script
        action_space[action] = 1

        # Update the environment with the new action space
        obs, reward, done, _ = env.step(action_space)
        if done:
            print('\n\ndone\n\n')
            env.env.close()
            