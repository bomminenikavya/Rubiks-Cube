import torch.nn as nn
import torch

class DCAModel(nn.Module):
    def __init__(self,input_dim=88):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(input_dim,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,1)
        )
    def forward(self,x):
        return self.net(x)
