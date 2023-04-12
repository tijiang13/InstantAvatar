from dataclasses import dataclass
import torch


@dataclass
class Rays:
    o: torch.FloatTensor # (N, 3)
    d: torch.FloatTensor # (N, 3)

    near: torch.FloatTensor = None # (N, )
    far: torch.FloatTensor = None  # (N, )