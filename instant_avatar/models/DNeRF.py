from .structures.body_model_param import SMPLParamEmbedding
from ..deformers.smpl_deformer import SMPLDeformer
from .structures.utils import Rays
from .networks.ngp import NeRFNGPNet
from ..utils.loss import Evaluator
import torch
from torch_ema import ExponentialMovingAverage
import numpy as np
import pytorch_lightning as pl
import hydra
import cv2
import os
import torch.nn.functional as F

import logging
logger = logging.getLogger("instant-avatar.DNeRF")
logger.addHandler(logging.FileHandler("DNeRF.log"))

class DNeRFModel(pl.LightningModule):
    def __init__(self, opt, datamodule) -> None:
        super().__init__()
        self.automatic_optimization = False

        self.net_coarse = hydra.utils.instantiate(opt.network)
        if opt.optimize_SMPL.enable:
            self.SMPL_param = SMPLParamEmbedding(**datamodule.trainset.get_SMPL_params())
        self.deformer = hydra.utils.instantiate(opt.deformer)
        self.loss_fn = hydra.utils.instantiate(opt.loss)
        self.renderer = hydra.utils.instantiate(opt.renderer)
        self.evaluator = Evaluator()
        self.datamodule = datamodule
        self.opt = opt

    def configure_optimizers(self):
        # use one optimizer with different learning rate for params
        params, body_model_params, encoding_params = [], [], []
        for (name, param) in self.named_parameters():
            # skip lpips
            if name.startswith("evaluator"):
                continue

            if name.startswith("SMPL_param"):
                body_model_params.append(param)
            else:
                if "encoder" in name:
                    encoding_params.append(param)
                else:
                    params.append(param)
        optimizer = torch.optim.Adam([
            {"params": encoding_params},
            {"params": params},
            {"params": body_model_params, "lr": self.opt.optimize_SMPL.get("lr", 5e-4)},
        ], **self.opt.optimizer)

        max_epochs = self.opt.scheduler.max_epochs
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
            lambda epoch: (1 - epoch / max_epochs) ** 1.5
        )

        # additional configure for gradscaler
        self.scaler = torch.cuda.amp.GradScaler(init_scale=1024.0)

        # additional configure for ema
        if hasattr(self.opt, "ema") and hasattr(self.opt.ema, "decay"):
            ema = ExponentialMovingAverage(encoding_params + params, decay=self.opt.ema.decay)
            self.ema = ema
        return [optimizer], [scheduler]

    def forward(self, batch, eval_mode=False):
        eval_mode = not self.training
        rays = Rays(o=batch["rays_o"], d=batch["rays_d"], near=batch["near"], far=batch["far"])
        self.deformer.transform_rays_w2s(rays)
        use_noise = self.global_step < 1000 and not self.opt.optimize_SMPL.get("is_refine", False) and not eval_mode
        return self.renderer(rays,
                             lambda x, _: self.deformer(x, self.net_coarse, eval_mode),
                             eval_mode=eval_mode,
                             noise=1 if use_noise else 0,
                             bg_color=batch.get("bg_color", None))

    @torch.no_grad()
    def render_image_fast(self, batch, img_size):
        if hasattr(self, "SMPL_param") and self.opt.optimize_SMPL.get("is_refine", False):
            body_params = self.SMPL_param(batch["idx"])
            for k in ["global_orient", "body_pose", "transl"]:
                assert batch[k].shape == body_params[k].shape
                batch[k] = body_params[k]

            if isinstance(self.deformer, SMPLDeformer):
                batch["betas"] = body_params["betas"]

            # update near & far with refined SMPL
            dist = torch.norm(batch["transl"], dim=-1, keepdim=True).detach()
            batch["near"][:] = dist - 1
            batch["far"][:] = dist + 1

        self.deformer.prepare_deformer(batch)
        if hasattr(self.renderer, "density_grid_test"):
            self.renderer.density_grid_test.initialize(self.deformer, self.net_coarse)

        if hasattr(self, "ema"):
            with self.ema.average_parameters():
                d = self.forward(batch, eval_mode=True)
        else:
            d = self.forward(batch, eval_mode=True)

        rgb = d["rgb_coarse"].reshape(-1, *img_size, 3)
        depth = d["depth_coarse"].reshape(-1, *img_size)
        alpha = d["alpha_coarse"].reshape(-1, *img_size)
        counter = d["counter_coarse"].reshape(-1, *img_size)
        return rgb, depth, alpha, counter

    def update_density_grid(self):
        if self.global_step % 16 == 0 and hasattr(self.renderer, "density_grid_train"):
            _, density, valid = self.renderer.density_grid_train.update(self.deformer,
                                                                        self.net_coarse,
                                                                        self.global_step)
            reg = 20 * density[~valid].mean()
            if self.global_step < 500:
                reg += 0.5 * density.mean()
            return reg
        else:
            return None

    def training_step(self, batch, *args, **kwargs):
        if hasattr(self, "SMPL_param"):
            body_params = self.SMPL_param(batch["idx"])
            for k in ["global_orient", "body_pose", "transl"]:
                batch[k] = body_params[k]
                gt = self.datamodule.trainset.smpl_params[k]
                gt = torch.from_numpy(gt).cuda()
                self.log(f"train/{k}", F.l1_loss(getattr(self.SMPL_param, k).weight, gt))

            # update betas if use SMPLDeformer
            if isinstance(self.deformer, SMPLDeformer):
                batch["betas"] = body_params["betas"]

            # update near & far with refined SMPL
            dist = torch.norm(batch["transl"], dim=-1, keepdim=True).detach()
            batch["near"][:] = dist - 1
            batch["far"][:] = dist + 1

        self.deformer.prepare_deformer(batch)
        reg = self.update_density_grid()
        if isinstance(self.net_coarse, NeRFNGPNet):
            self.net_coarse.initialize(self.deformer.bbox)

        predicts = self.forward(batch, eval_mode=False)
        losses = self.loss_fn(predicts, batch)

        if not (reg is None or self.opt.optimize_SMPL.get("is_refine", False)):
            losses["reg"] = reg
            losses["loss"] += reg

        # add some extra loss here
        losses["psnr"] = self.evaluator.psnr(predicts["rgb_coarse"], batch["rgb"])

        for k, v in losses.items():
            self.log(f"train/{k}", v)
        if self.precision == 16:
            self.log("precision/scale",
                     self.trainer.precision_plugin.scaler.get_scale())

        if self.automatic_optimization:
            return losses["loss"]
        else:
            loss = losses["loss"]
            optimizer = self.optimizers(False)

            try:
                optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            except Exception as e:
                logger.warning(e)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if hasattr(self, "ema"):
            self.ema.update()

    def on_validation_epoch_end(self, *args, **kwargs):
        # self.deformer.initialized = False
        scheduler = self.lr_schedulers()
        scheduler.step()

    # def on_validation_epoch_start(self, *args, **kwargs):
    #     self.deformer.initialized = False

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, *args, **kwargs):
        img_size = self.datamodule.valset.image_shape
        rgb, depth, alpha, counter = self.render_image_fast(batch, img_size)

        rgb_gt = batch["rgb"].reshape(-1, *img_size, 3)
        alpha_gt = batch["alpha"].reshape(-1, *img_size)

        losses = {
            # add some extra loss here
            **self.evaluator(rgb, rgb_gt),

            # add regular losses
            "rgb_loss": (rgb - rgb_gt).square().mean(),
            "counter_avg": counter.mean(),
            "counter_max": counter.max(),
        }
        for k, v in losses.items():
            self.log(f"val/{k}", v, on_epoch=True)

        # extra visualization for debugging
        if batch_idx == 0:
            os.makedirs("animation/progression/", exist_ok=True)
            cv2.imwrite(f"animation/progression/{self.global_step:06d}.png", rgb[0].cpu().numpy() * 255)
            # visualize heatmap (blue ~ 0, red ~ 1)
            errmap = (rgb - rgb_gt).square().sum(-1).sqrt().cpu().numpy()[0] / np.sqrt(3)
            errmap = cv2.applyColorMap((errmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
            errmap_rgb = torch.from_numpy(errmap).to(rgb.device)[None] / 255

            errmap = (alpha - alpha_gt).abs().cpu().numpy()[0]
            errmap = cv2.applyColorMap((errmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
            errmap_alpha = torch.from_numpy(errmap).to(rgb.device)[None] / 255

            img = torch.cat([rgb_gt, errmap_rgb, errmap_alpha], dim=0)
            self.logger.experiment.add_images(f"val/errmap",
                                              img[..., [2, 1, 0]],
                                              global_step=self.global_step,
                                              dataformats="NHWC")

            # visualize novel pose
            # batch["global_orient"][:] = 0
            batch["body_pose"][:] = 0
            batch["body_pose"][:, 2] = 0.5
            batch["body_pose"][:, 5] = -0.5

            dist = torch.sqrt(torch.square(batch["transl"]).sum(-1))
            batch["near"] = torch.ones_like(batch["rays_d"][..., 0]) * (dist - 1)
            batch["far"] = torch.ones_like(batch["rays_d"][..., 0]) * (dist + 1)

            rgb_cano, *_ = self.render_image_fast(batch, img_size)
            img = torch.cat([rgb_gt, rgb, rgb_cano], dim=0)
            self.logger.experiment.add_images(f"val/cano_pose", 
                                              img[..., [2, 1, 0]],
                                              global_step=self.global_step,
                                              dataformats="NHWC")
        return losses

    @torch.no_grad()
    def test_step(self, batch, batch_idx, *args, **kwargs):
        img_size = self.datamodule.testset.image_shape
        rgb, *_ = self.render_image_fast(batch, img_size)
        rgb_gt = batch["rgb"].reshape(-1, *img_size, 3)
        losses = {
            # add some extra loss here 
            **self.evaluator(rgb, rgb_gt),
            "rgb_loss": (rgb - rgb_gt).square().mean(),
        }

        errmap = (rgb - rgb_gt).square().sum(-1).sqrt().cpu().numpy()[0] / np.sqrt(3)
        errmap = cv2.applyColorMap((errmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        errmap = torch.from_numpy(errmap).to(rgb.device)[None] / 255

        if batch_idx == 0:
            os.makedirs("outputs/", exist_ok=True)

        img = torch.cat([rgb_gt, rgb, errmap], dim=2)
        cv2.imwrite(f"outputs/{batch_idx}.png", img.cpu().numpy()[0] * 255)

        for k, v in losses.items():
            self.log(f"test/{k}", v, on_epoch=True)
        return {}


    ######################
    # DATA RELATED HOOKS #
    ######################
    def train_dataloader(self):
        return self.datamodule.train_dataloader()

    def val_dataloader(self):
        return self.datamodule.val_dataloader()

    def test_dataloader(self):
        return self.datamodule.test_dataloader()
