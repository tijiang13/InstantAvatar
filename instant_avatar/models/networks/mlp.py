import torch
import numpy as np
import torch.nn as nn


class PositionalEncoding:
    """ Postional Encoding in NeRF paper"""
    def __init__(self, input_dim, multires) -> None:
        self.freq_bands = torch.logspace(0, multires-1, multires, base=2)
        self.periodic_fns = [torch.sin, torch.cos]
        self.out_dim = (len(self.freq_bands) * len(self.periodic_fns) + 1) * input_dim

    def __call__(self, x):
        embedding = [x]
        for freq_band in self.freq_bands:
            for fn in self.periodic_fns:
                embedding.append(fn(x * np.pi * freq_band))
        return torch.cat(embedding, dim=-1)


class NeRFNet(nn.Module):
    def __init__(self, opt):
        super().__init__()

        # embedder
        self.embed_fn_pts = PositionalEncoding(3, multires=10)
        if opt.use_viewdir:
            self.embed_fn_dir = PositionalEncoding(3, multires=6)
            n_dir = self.embed_fn_dir.out_dim
        else:
            self.embed_fn_dir = None
            n_dir = 0

        # Block 0
        self.block0 = nn.Sequential(
            nn.Linear(self.embed_fn_pts.out_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
        )

        # Block 1
        self.block1 = nn.Sequential(
            nn.Linear(self.embed_fn_pts.out_dim + 256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 257),
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Linear(n_dir + 256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )

        self.opt = opt

    def forward(self, x, d):
        pts_encoding = self.embed_fn_pts(x)
        x = self.block0(pts_encoding)
        x = self.block1(torch.cat([pts_encoding, x], dim=-1))
        sigma = x[:, 0]

        if self.embed_fn_dir is None:
            x = self.block2(x[:, 1:])
        else:
            dir_encoding = self.embed_fn_dir(d)
            x = self.block2(torch.cat([dir_encoding, x[:, 1:]], dim=-1))
        return x, torch.relu(sigma).float()