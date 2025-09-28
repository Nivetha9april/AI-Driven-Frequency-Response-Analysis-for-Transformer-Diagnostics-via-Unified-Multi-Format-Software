import torch.nn as nn
import torch

class FRA_CNN(nn.Module):
    def __init__(self, n_classes=8):
        super(FRA_CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)

        # temporary layer for figuring out size
        self.flatten_dim = None
        self.fc1 = None
        self.fc2 = None
        self.n_classes = n_classes

    def _get_flatten_dim(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        return x.view(x.size(0), -1).shape[1]

    def build_fc_layers(self, input_dim):
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, self.n_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)

        # Build FC layers dynamically the first time
        if self.fc1 is None:
            self.build_fc_layers(x.shape[1])

        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
