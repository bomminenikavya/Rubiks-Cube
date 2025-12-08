import torch.nn as nn
import torch

class DQN(nn.Module):
    def __init__(self,input_dim=88,output_dim=12):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(input_dim,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,output_dim)
        )
    def forward(self,x):
        return self.net(x)
