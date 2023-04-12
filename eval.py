import glob
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import AdvancedProfiler
import logging
import hydra
from omegaconf import OmegaConf


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
    pl_logger = TensorBoardLogger("tensorboard", name="refine", version=0)

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
                         logger=pl_logger,
                         log_every_n_steps=1,
                         **opt.train)

    checkpoints = sorted(glob.glob("checkpoints/refinement/*.ckpt"))
    if len(checkpoints) > 0 and opt.resume:
        print("Resume from", checkpoints[-1])
        trainer.fit(model, ckpt_path=checkpoints[-1])
    else:
        print("Saving configs.")
        OmegaConf.save(opt, "config_refine.yaml")
        trainer.fit(model)

    result = trainer.test(model)[0]
    result_str = ""
    s = " | ".join([f"{k.split('/')[1]:^10}" for k in result.keys()])
    result_str += f"| {0:^15} | {s} |\n"
    s = " | ".join([f"{v:^10.3f}" for v in result.values()])
    result_str += f"| {opt.dataset.subject:^15} | {s} |\n"
    with open("refine_result.txt", "w") as f:
        f.write(result_str)


if __name__ == "__main__":
    main()
