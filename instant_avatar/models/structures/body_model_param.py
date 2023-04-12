import torch
import torch.nn as nn

class SMPLParamEmbedding(nn.Module):
    """optimize SMPL params on the fly"""
    def __init__(self,
                 **kwargs) -> None:
        super().__init__()

        # fill in init value
        for k, v in kwargs.items():
            setattr(self, k, nn.Embedding.from_pretrained(v, freeze=False))
        self.keys = ["betas", "global_orient", "transl", "body_pose"]

    def forward(self, idx):
        return {
            "betas": self.betas(torch.zeros_like(idx)),
            "body_pose": self.body_pose(idx),
            "global_orient": self.global_orient(idx),
            "transl": self.transl(idx),
        }

    def tv_loss(self, idx):
        loss = 0

        N = len(self.global_orient.weight)
        idx_p = (idx - 1).clip(min=0)
        idx_n = (idx + 1).clip(max=N - 1)
        for (k, v) in self.items():
            if k == "betas":
                continue
            loss = loss + (v(idx) - v(idx_p)).square().mean()
            loss = loss + (v(idx_n) - v(idx)).square().mean()
        return loss