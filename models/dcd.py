import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DC_D(nn.Module):
    def __init__(self, out_dim, img_info):
        super().__init__()
        H, W, C = img_info['H'], img_info['W'], img_info['C']
        self.out_dim = out_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(C, 32, 5),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5),
            nn.LeakyReLU(0.01),
            nn.MaxPool2d(2),
            nn.Flatten(1, -1),
            nn.Linear(4 * 4 * 64, 4 * 4 * 64),
            nn.LeakyReLU(0.01),
            nn.Linear(4 * 4 * 64, 128),
            nn.LeakyReLU(0.01)
        )
        self.fc = nn.Linear(128, self.out_dim)

    def forward(self, x):
        return self.fc(self.encoder(x))
