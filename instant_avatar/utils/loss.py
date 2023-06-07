import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd
from third_parties.lpips import LPIPS


class NGPLoss(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.lpips = LPIPS(net="vgg", pretrained=True)
        for param in self.lpips.parameters(): param.requires_grad=False
        self.opt = opt

    def forward(self, predicts, targets):
        losses = {}
        loss = 0

        loss_rgb = F.mse_loss(predicts["rgb_coarse"], targets["rgb"], reduction="mean")
        loss += self.opt.w_rgb * loss_rgb
        losses["mse_loss"] = loss_rgb

        loss_alpha = F.mse_loss(predicts["alpha_coarse"], targets["alpha"])
        loss += self.opt.w_alpha * loss_alpha
        losses["loss_alpha_coarse"] = loss_alpha

        if self.opt.get("w_lpips", 0) > 0 and len(predicts["rgb_coarse"].shape) == 5:
            loss_lpips = self.lpips(predicts["rgb_coarse"][..., [2, 1, 0]].flatten(0, 1).permute(0, 3, 1, 2).clip(max=1),
                                    targets["rgb"][..., [2, 1, 0]].flatten(0, 1).permute(0, 3, 1, 2)).sum()
            losses["loss_lpips"] = loss_lpips
            loss += loss_lpips * self.opt.w_lpips
        
        if self.opt.get("w_depth_reg", 0) > 0 and len(predicts["rgb_coarse"].shape) == 5:
            alpha_sum = predicts["alpha_coarse"].sum(dim=(-1, -2))
            depth_avg = (predicts["depth_coarse"] * predicts["alpha_coarse"]).sum(dim=(-1, -2)) / (alpha_sum + 1e-3)
            loss_depth_reg = predicts["alpha_coarse"] * (predicts["depth_coarse"] - depth_avg[..., None, None]).abs()
            loss_depth_reg = loss_depth_reg.mean()
            losses["loss_depth_reg"] = loss_depth_reg
            loss += self.opt.w_depth_reg * loss_depth_reg

        OFFSET = 0.313262
        reg_alpha   = (-torch.log(torch.exp(-predicts["alpha_coarse"]) + torch.exp(predicts["alpha_coarse"] - 1))).mean() + OFFSET
        reg_density = (-torch.log(torch.exp(-predicts["weight_coarse"]) + torch.exp(predicts["weight_coarse"] - 1))).mean() + OFFSET
        losses["reg_alpha"] = reg_alpha
        losses["reg_density"] = reg_density
        loss += self.opt.w_reg * reg_alpha
        loss += self.opt.w_reg * reg_density

        losses["loss"] = loss
        return losses


class NeRFLoss(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt

    def forward(self, predicts, targets):
        losses = {}
        loss = 0

        loss_rgb = F.mse_loss(predicts["rgb_coarse"], targets["rgb"], reduction="mean")
        loss += self.opt.w_rgb * loss_rgb
        losses["mse_loss"] = loss_rgb

        loss_alpha = F.mse_loss(predicts["alpha_coarse"], targets["alpha"])
        loss += self.opt.w_alpha * loss_alpha
        losses["loss_alpha_coarse"] = loss_alpha

        OFFSET = 0.313262
        reg_alpha   = (-torch.log(torch.exp(-predicts["alpha_coarse"]) + torch.exp(predicts["alpha_coarse"] - 1))).mean() + OFFSET
        reg_density = (-torch.log(torch.exp(-predicts["weight_coarse"]) + torch.exp(predicts["weight_coarse"] - 1))).mean() + OFFSET
        losses["reg_alpha"] = reg_alpha
        losses["reg_density"] = reg_density
        loss += self.opt.w_reg * reg_alpha
        loss += self.opt.w_reg * reg_density

        losses["loss"] = loss
        return losses
