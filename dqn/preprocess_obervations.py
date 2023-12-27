import random
import torch
import torchvision.transforms as transforms


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

    
def process(state):
    """ Preprocess observation """
    return transform(state["pov"])

def parse_action(env, action_index):
    """ Returns action dict with the selected action index """
    action_space = env.action_space.noop()
    action = list(action_space.keys())[action_index]
    print(f'Action index: {action_index}, action: {action}')
    if action == 'camera':
        action_space[action] = [random.randint(-180, 180), random.randint(-180, 180)]
    else:
        action_space[action] = 1
    return action_space
    


    

"""
Observation Space
Dict({
    "pov": "Box(low=0, high=255, shape=(64, 64, 3))"
})

Action Space
Dict({
    "attack": "Discrete(2)",
    "back": "Discrete(2)",
    "camera": "Box(low=-180.0, high=180.0, shape=(2,))",
    "forward": "Discrete(2)",
    "jump": "Discrete(2)",
    "left": "Discrete(2)",
    "right": "Discrete(2)",
    "sneak": "Discrete(2)",
    "sprint": "Discrete(2)"
})
"""
