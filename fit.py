import glob
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import AdvancedProfiler
import logging
import hydra
from omegaconf import OmegaConf
from pathlib import Path
import numpy as np


@hydra.main(config_path="./confs", config_name="SNARF_NGP_fitting")
def main(opt):
    pl.seed_everything(opt.seed)
    torch.set_printoptions(precision=6)
    print(f"Switch to {os.getcwd()}")

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"checkpoints/fit/",
        filename="epoch={epoch:04d}-val_psnr={val/psnr:.1f}",
        auto_insert_metric_name=False,
        save_last=True,
        **opt.checkpoint
    )
    lr_monitor = pl.callbacks.LearningRateMonitor()

    pl_logger = TensorBoardLogger("tensorboard", name="fit", version=0)
    pl_profiler = AdvancedProfiler("profiler", "advance_profiler")

    datamodule = hydra.utils.instantiate(opt.dataset, _recursive_=False)
    model = hydra.utils.instantiate(opt.model, datamodule=datamodule, _recursive_=False)
    trainer = pl.Trainer(gpus=1,
                         accelerator="gpu",
                         callbacks=[checkpoint_callback, lr_monitor],
                         num_sanity_val_steps=0,  # disable sanity check
                         logger=pl_logger,
                        #  gradient_clip_val=0.1,
                        #  profiler=pl_profiler,
                         **opt.train)

    checkpoints = sorted(glob.glob("checkpoints/fit/*.ckpt"))
    if len(checkpoints) > 0 and opt.resume:
        print("Resume from", checkpoints[-1])
        trainer.fit(model, ckpt_path=checkpoints[-1])
    else:
        print("Saving configs.")
        OmegaConf.save(opt, "config_fit.yaml")
        trainer.fit(model)

    # export fit parameters
    optimized_params = {}
    for k in model.SMPL_param.keys:
        v = getattr(model.SMPL_param, k)
        optimized_params[k] = v.weight.clone().detach().cpu().numpy()

    root = hydra.utils.to_absolute_path(opt.dataset.opt.dataroot)
    root = Path(root) / "poses"
    root.mkdir(exist_ok=True)

    param_path = root / "train.npz"
    if True or not os.path.exists(param_path):
        print(f"Save optimized pose to {param_path}")
        np.savez(str(param_path), **optimized_params)
    else:
        while True:
            choice = input(f"Found optimized params in {param_path}. Overwrite? (y/n)")
            if choice.lower() == "y":
                np.savez(str(param_path), **optimized_params)
                break
            elif choice.lower() == "n":
                break
            else:
                print("Invalid input. Please enter 'y' or 'n'.")

if __name__ == "__main__":
    main()
