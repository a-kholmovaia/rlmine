from torch import nn
import torch
import random
import math
class DQN(nn.Module):
    EPS_START = 0.5
    EPS_END = 0.05
    EPS_DECAY = 1000

    def __init__(self, env, input_dim, n_actions=10):
        super(DQN, self).__init__()
        self.env = env
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cnn_base = nn.Sequential(
            nn.Conv2d(input_dim[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Dropout(0.3),
    
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Flatten()
        )

        self.fc_input_dim = self.feature_size()
        print(f'CNN feature size: {self.fc_input_dim}')
        self.fc_head = nn.Sequential(
            nn.Linear(self.fc_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, self.n_actions)
        )
 
    def forward(self, x):
        cnn_features = self.cnn_base(x)
        return self.fc_head(cnn_features)

    
    def get_action(self, state, steps_done: int):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * steps_done / self.EPS_DECAY)
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.forward(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[random.randint(0, self.n_actions-1)]], device=self.device, dtype=torch.long)
        

        self.steps_done += 1
    
    def feature_size(self):
        return self.cnn_base(torch.autograd.Variable(torch.zeros(1, *self.input_dim))).view(1,-1).size(1)