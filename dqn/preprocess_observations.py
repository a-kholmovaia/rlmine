import random
import torch
import torchvision.transforms as transforms


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

    
def process_states_batch(state):
    """ Preprocess observation batches"""#
    images = torch.tensor(state['pov'], dtype=torch.float32).squeeze().transpose(1, -1)
    processed_state = [transform(transforms.ToPILImage()(x_)).unsqueeze(0) for x_ in images]
    return torch.cat(processed_state)

def process_state(state):
    """ Preprocess observation """
    tranformed_state = transform(transforms.ToPILImage()(state['pov']))
    return torch.tensor(tranformed_state, dtype=torch.float32).unsqueeze(0)

def parse_action_ind2dict(env, action_index):
    """ Returns action dict with the selected action index """
    action_space = env.action_space.noop()
    action = list(action_space.keys())[action_index]
    print(f'Action: {action}')
    if action == 'camera':
        action_space[action] = [random.randint(-180, 180), random.randint(-180, 180)]
        #action_space['attack'] = 1
    else:
        action_space[action] = 1
    return action_space

def parse_action2ind(env, actions:dict, batch_size:int):
    """ Returns action index for a selected action """
    action_space = env.action_space.noop()
    action_idx = []
    for j in range(batch_size):
        action = None
        for i in actions.items():
            if i[0] != "camera":
                if i[1][j][0] > 0:
                    action = i[0]
                    break
        if action == None:
            action = "camera"
        action_ind = list(action_space.keys()).index(action)
        action_idx.append(action_ind)
    return action_idx




    

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
