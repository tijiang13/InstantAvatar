import glob
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import hydra
from omegaconf import OmegaConf


@hydra.main(config_path="./confs", config_name="SNARF_NGP")
def main(opt):
    pl.seed_everything(opt.seed)
    torch.set_printoptions(precision=6)
    print(f"Switch to {os.getcwd()}")

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"checkpoints/",
        filename="epoch={epoch:04d}-val_psnr={val/psnr:.1f}",
        auto_insert_metric_name=False,
        save_last=True,
        **opt.checkpoint
    )
    lr_monitor = pl.callbacks.LearningRateMonitor()

    pl_logger = TensorBoardLogger("tensorboard", name="default", version=0)

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

    checkpoints = sorted(glob.glob("checkpoints/*.ckpt"))
    if len(checkpoints) > 0 and opt.resume:
        print("Resume from", checkpoints[-1])
        trainer.fit(model, ckpt_path=checkpoints[-1])
    else:
        print("Saving configs.")
        OmegaConf.save(opt, "config.yaml")
        trainer.fit(model)


if __name__ == "__main__":
    main()
