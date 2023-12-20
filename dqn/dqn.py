from torch import nn


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3*n_observations, 256, kernel_size=(3,3), stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.dropc1 = nn.Dropout(0.3)
 
        self.conv2 = nn.Conv2d(256, 128, kernel_size=(3,3), stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.dropc2 = nn.Dropout(0.3)

        self.conv3 = nn.Conv2d(128, 32, kernel_size=(3,3), stride=1, padding=1)
        self.act3 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
 
        self.flat = nn.Flatten()

        self.fc1 = nn.Linear(8192, 1024)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(1024, 512)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.5)
 
        self.fc3 = nn.Linear(512, 256)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)
 
        self.fc4 = nn.Linear(128, n_actions)
 
    def forward(self, x):
        # input 3x32x32, output 32x32x32
        x = self.act1(self.conv1(x))
        x = self.dropc1(x)
        # input 32x32x32, output 32x32x32
        x = self.act2(self.conv2(x))
        x = self.dropc2(x)
        x = self.act3(self.conv3(x))
        # input 32x32x32, output 32x16x16
        x = self.pool(x)
        # input 32x16x16, output 8192
        x = self.flat(x)

        # input 8192, output 512
        x = self.act1(self.fc1(x))
        x = self.drop1(x)

        x = self.act2(self.fc2(x))
        x = self.drop2(x)
        
        x = self.act3(self.fc3(x))
        x = self.drop3(x)
        # input 512, output 10
        x = self.fc4(x)

        return x