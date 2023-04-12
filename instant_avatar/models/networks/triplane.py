import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn
import math

EPS = 1e-3

class TriPlane(nn.Module):
    def __init__(self, features=32, resX=256, resY=256, resZ=256):
        super().__init__()
        self.plane_xy = nn.Parameter(torch.randn(1, features, resX, resY))
        self.plane_xz = nn.Parameter(torch.randn(1, features, resX, resZ))
        self.plane_yz = nn.Parameter(torch.randn(1, features, resY, resZ))
        self.dim = features
        self.n_input_dims = 3
        self.n_output_dims = 3 * features

    def forward(self, x):
        assert x.max() <= 1 + EPS and x.min() >= -EPS, f"x must be in [0, 1], got {x.min()} and {x.max()}"
        x = x * 2 - 1
        shape = x.shape
        coords = x.reshape(1, -1, 1, 3)
        # align_corners=True ==> the extrema (-1 and 1) considered as the center of the corner pixels
        # F.grid_sample: [1, C, H, W], [1, N, 1, 2] -> [1, C, N, 1]
        feat_xy = F.grid_sample(self.plane_xy, coords[..., [0, 1]], align_corners=True)[0, :, :, 0].transpose(0, 1)
        feat_xz = F.grid_sample(self.plane_xz, coords[..., [0, 2]], align_corners=True)[0, :, :, 0].transpose(0, 1)
        feat_yz = F.grid_sample(self.plane_yz, coords[..., [1, 2]], align_corners=True)[0, :, :, 0].transpose(0, 1)
        feat = torch.cat([feat_xy, feat_xz, feat_yz], dim=1)
        feat = feat.reshape(*shape[:-1], 3 * self.dim)
        return feat

class NeRFNGPNet(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.encoder = TriPlane(32, 256, 256, 256)
        self.sigma_net = tcnn.Network(
            n_input_dims=self.encoder.n_output_dims,
            n_output_dims=16,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            }
        )

        self.color_net = tcnn.Network(
            n_input_dims=15,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            },
        )
        self.register_buffer("center", torch.FloatTensor(opt.center))
        self.register_buffer("scale", torch.FloatTensor(opt.scale))
        self.opt = opt

    def initialize(self, bbox):
        if hasattr(self, "bbox"):
            return
        c = (bbox[0] + bbox[1]) / 2
        s = (bbox[1] - bbox[0])
        self.center = c
        self.scale = s
        self.bbox = bbox

    def forward(self, x, d, cond=None):
        # normalize pts and view_dir to [0, 1]
        x = (x - self.center) / self.scale + 0.5
        # assert x.min() >= 0 and x.max() <= 1
        x = x.clamp(min=0, max=1)
        x = self.encoder(x)
        x = self.sigma_net(x)
        sigma = x[..., 0].float()
        color = self.color_net(x[..., 1:]).float()
        return color, sigma