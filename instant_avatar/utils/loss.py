import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class NGPLoss(nn.Module):
    def __init__(self, opt) -> None:
        super().__init__()
        self.opt = opt

    def forward(self, predicts, targets):
        losses = {}
        loss = 0

        loss_rgb = F.huber_loss(predicts["rgb_coarse"], targets["rgb"], reduction="mean", delta=0.1)
        loss += self.opt.w_rgb * loss_rgb
        losses["huber_loss"] = loss_rgb

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

        loss_alpha = F.l1_loss(predicts["alpha_coarse"], targets["alpha"])
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


class Evaluator(nn.Module):
    def __init__(self):
        super().__init__()
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex")
        self.psnr = PeakSignalNoiseRatio(data_range=1)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1)

    # custom_fwd: turn off mixed precision to avoid numerical instability during evaluation
    @custom_fwd(cast_inputs=torch.float32)
    def forward(self, rgb, rgb_gt):
        # torchmetrics assumes NCHW format
        rgb = rgb.permute(0, 3, 1, 2).clamp(max=1.0)
        rgb_gt = rgb_gt.permute(0, 3, 1, 2)

        return {
            "psnr": self.psnr(rgb, rgb_gt),
            "ssim": self.ssim(rgb, rgb_gt),
            "lpips": self.lpips(rgb, rgb_gt),
        }
