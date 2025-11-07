import torch
import torch.nn as nn

MIN_LOG_STD = -4.0
MAX_LOG_STD = 2.0


class ReachNet(nn.Module):
    def __init__(self, in_dim, out_dim, n_mix=5, hid=128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid),    nn.ReLU())
        self.pi   = nn.Linear(hid, n_mix)
        self.mu   = nn.Linear(hid, n_mix * out_dim)
        self.logS = nn.Linear(hid, n_mix * out_dim)
        self.n_mix, self.out_dim = n_mix, out_dim

    def forward(self, x):                      # x = [s0, g]
        h  = self.trunk(x)
        pi = torch.softmax(self.pi(h), -1)                 # (B,M)
        mu = self.mu(h).view(-1, self.n_mix, self.out_dim) # (B,M,D)
        logS = self.logS(h).clamp(MIN_LOG_STD, MAX_LOG_STD).view_as(mu)
        return pi, mu, logS
