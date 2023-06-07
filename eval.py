import glob
import os
import cv2
import pytorch_lightning as pl
import hydra
from omegaconf import OmegaConf
import torch
import torch.nn as nn
from torch.cuda.amp import custom_fwd
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class Evaluator(nn.Module):
    """adapted from https://github.com/JanaldoChen/Anim-NeRF/blob/main/models/evaluator.py"""
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


@hydra.main(config_path="./confs", config_name="SNARF_NGP_refine")
def main(opt):
    pl.seed_everything(opt.seed)
    torch.set_printoptions(precision=6)
    print(f"Switch to {os.getcwd()}")

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"checkpoints/refinement/",
        filename="epoch={epoch:04d}-val_psnr={val/psnr:.1f}",
        auto_insert_metric_name=False,
        **opt.checkpoint
    )
    lr_monitor = pl.callbacks.LearningRateMonitor()

    # refine test data
    opt.dataset.opt.train.start = opt.dataset.opt.test.start
    opt.dataset.opt.train.end = opt.dataset.opt.test.end
    opt.dataset.opt.train.skip = opt.dataset.opt.test.skip

    opt.dataset.opt.val.start = opt.dataset.opt.test.start
    opt.dataset.opt.val.end = opt.dataset.opt.test.end
    opt.dataset.opt.val.skip = opt.dataset.opt.test.skip

    datamodule = hydra.utils.instantiate(opt.dataset, _recursive_=False)

    # load checkpoint
    model = hydra.utils.instantiate(opt.model, datamodule=datamodule, _recursive_=False)
    state_dict = model.state_dict()

    checkpoint = sorted(glob.glob("checkpoints/*.ckpt"))[-1]
    for k, v in torch.load(checkpoint)["state_dict"].items():
        if not k.startswith("SMPL_param"):
            state_dict[k] = v
    model.load_state_dict(state_dict)

    # freeze all the parameters other than SMPL pose
    for k, param in model.named_parameters():
        if not k.startswith("SMPL_param"):
            param.requires_grad = False

    trainer = pl.Trainer(gpus=1,
                         accelerator="gpu",
                         callbacks=[checkpoint_callback, lr_monitor],
                         num_sanity_val_steps=0,  # disable sanity check
                         weights_summary=None,
                         logger=False,
                         enable_progress_bar=False,
                         **opt.train)

    checkpoints = sorted(glob.glob("checkpoints/refinement/*.ckpt"))
    if len(checkpoints) > 0 and opt.resume:
        print("Resume from", checkpoints[-1])
        trainer.fit(model, ckpt_path=checkpoints[-1])
    else:
        print("Saving configs.")
        OmegaConf.save(opt, "config_refine.yaml")
        trainer.fit(model)

    trainer.test(model)[0]
    imgs = [cv2.imread(fn) for fn in glob.glob("test/*.png")]
    imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]
    imgs = [torch.tensor(img).cuda().float() / 255.0 for img in imgs]

    evaluator = Evaluator()
    evaluator = evaluator.cuda()
    evaluator.eval()

    H, W = imgs[0].shape[:2]
    W //= 3
    with torch.no_grad():
        results = [evaluator(img[None, :, W:2*W], img[None, :, :W]) for img in imgs]
    
    with open("results.txt", "w") as f:
        psnr = torch.stack([r['psnr'] for r in results]).mean().item()
        print(f"PSNR: {psnr:.2f}")
        f.write(f"PSNR: {psnr:.2f}\n")

        ssim = torch.stack([r['ssim'] for r in results]).mean().item()
        print(f"SSIM: {ssim:.4f}")
        f.write(f"SSIM: {ssim:.4f}\n")

        lpips = torch.stack([r['lpips'] for r in results]).mean().item()
        print(f"LPIPS: {lpips:.4f}")
        f.write(f"LPIPS: {lpips:.4f}\n")


if __name__ == "__main__":
    main()
