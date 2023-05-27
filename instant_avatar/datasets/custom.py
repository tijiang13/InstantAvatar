from pathlib import Path
import os
import numpy as np
import hydra
import torch
import cv2
import glob
import pytorch_lightning as pl
from torch.utils.data import DataLoader


def get_ray_directions(H, W):
    x, y = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    xy = np.stack([x, y, np.ones_like(x)], axis=-1)
    return xy

def make_rays(K, c2w, H, W):
    xy = get_ray_directions(H, W).reshape(-1, 3).astype(np.float32)
    d_c = xy @ np.linalg.inv(K).T
    d_w = d_c @ c2w[:3, :3].T
    d_w = d_w / np.linalg.norm(d_w, axis=1, keepdims=True)
    o_w = np.tile(c2w[:3, 3], (len(d_w), 1))
    o_w = o_w.reshape(H, W, 3)
    d_w = d_w.reshape(H, W, 3)
    return o_w.astype(np.float32), d_w.astype(np.float32)

def load_smpl_param(path):
    smpl_params = dict(np.load(str(path)))
    if "thetas" in smpl_params:
        smpl_params["body_pose"] = smpl_params["thetas"][..., 3:]
        smpl_params["global_orient"] = smpl_params["thetas"][..., :3]
    return {
        "betas": smpl_params["betas"].astype(np.float32).reshape(1, 10),
        "body_pose": smpl_params["body_pose"].astype(np.float32),
        "global_orient": smpl_params["global_orient"].astype(np.float32),
        "transl": smpl_params["transl"].astype(np.float32),
    }

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, subject, split, opt):
        camera = np.load(str(root / "cameras.npz"))
        K = camera["intrinsic"]
        c2w = np.linalg.inv(camera["extrinsic"])
        height = camera["height"]
        width = camera["width"]

        self.downscale = opt.downscale
        if self.downscale > 1:
            height = int(height / self.downscale)
            width = int(width / self.downscale)
            K[:2] /= self.downscale

        self.rays_o, self.rays_d = make_rays(K, c2w, height, width)

        # prepare image and mask
        start = opt.start
        end = opt.end + 1
        skip = opt.get("skip", 1)
        self.img_lists = sorted(glob.glob(f"{root}/images/*.png"))[start:end:skip]
        self.msk_lists = sorted(glob.glob(f"{root}/masks/*.png"))[start:end:skip]

        if os.path.exists(root / f"poses/{split}.npz"):
            cached_path = root / f"poses/{split}.npz"
        else:
            cached_path = None
        
        if opt.get("fitting", False):
            # for fitting, optimize SMPL from scratch
            cached_path = None

        if cached_path and os.path.exists(cached_path):
            print(f"[{split}] Loading from", cached_path)
            self.smpl_params = load_smpl_param(cached_path)
        else:
            print(f"[{split}] No optimized smpl found.")
            self.smpl_params = load_smpl_param(root / "poses_optimized.npz")
            for k, v in self.smpl_params.items():
                if k != "betas":
                    self.smpl_params[k] = v[start:end:skip]

        self.split = split
        self.downscale = opt.downscale
        self.near = opt.get("near", None)
        self.far = opt.get("far", None)
        self.image_shape = (height, width)
        if split == "train":
            self.sampler = hydra.utils.instantiate(opt.sampler)

    def get_SMPL_params(self):
        return {
            k: torch.from_numpy(v.copy()) for k, v in self.smpl_params.items()
        }

    def __len__(self):
        return len(self.img_lists)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_lists[idx])
        msk = cv2.imread(self.msk_lists[idx], cv2.IMREAD_GRAYSCALE) / 255
        if self.downscale > 1:
            img = cv2.resize(img, dsize=None, fx=1/self.downscale, fy=1/self.downscale)
            msk = cv2.resize(msk, dsize=None, fx=1/self.downscale, fy=1/self.downscale)

        img = (img[..., :3] / 255).astype(np.float32)
        msk = msk.astype(np.float32)
        # apply mask
        if self.split == "train":
            bg_color = np.random.rand(*img.shape).astype(np.float32)
            img = img * msk[..., None] + (1 - msk[..., None]) * bg_color
        else:
            bg_color = np.ones_like(img).astype(np.float32)
            img = img * msk[..., None] + (1 - msk[..., None])
            
        if self.split == "train":
            (msk, img, rays_o, rays_d, bg_color) = \
                    self.sampler.sample(msk, img, self.rays_o, self.rays_d, bg_color) 
        else:
            rays_o = self.rays_o.reshape(-1, 3)
            rays_d = self.rays_d.reshape(-1, 3)
            img = img.reshape(-1, 3)
            msk = msk.reshape(-1)

        datum = {
            # NeRF
            "rgb": img.astype(np.float32),
            "rays_o": rays_o,
            "rays_d": rays_d,

            # SMPL parameters
            "betas": self.smpl_params["betas"][0],
            "global_orient": self.smpl_params["global_orient"][idx],
            "body_pose": self.smpl_params["body_pose"][idx],
            "transl": self.smpl_params["transl"][idx],

            # auxiliary
            "alpha": msk,
            "bg_color": bg_color,
            "idx": idx,
        }
        if self.near is not None and self.far is not None:
            datum["near"] = np.ones_like(rays_d[..., 0]) * self.near
            datum["far"] = np.ones_like(rays_d[..., 0]) * self.far
        else:
            # distance from camera (0, 0, 0) to midhip
            # TODO: we could replace it with bbox in the canonical space
            dist = np.sqrt(np.square(self.smpl_params["transl"][idx]).sum(-1))
            datum["near"] = np.ones_like(rays_d[..., 0]) * (dist - 1)
            datum["far"] = np.ones_like(rays_d[..., 0]) * (dist + 1)
        return datum


class CustomDataModule(pl.LightningDataModule):
    def __init__(self, opt, **kwargs):
        super().__init__()

        data_dir = Path(hydra.utils.to_absolute_path(opt.dataroot))
        for split in ("train", "val", "test"):
            dataset = CustomDataset(data_dir, opt.subject, split, opt.get(split))
            setattr(self, f"{split}set", dataset)
        self.opt = opt

    def train_dataloader(self):
        if hasattr(self, "trainset"):
            return DataLoader(self.trainset,
                              shuffle=True,
                              num_workers=self.opt.train.num_workers,
                              persistent_workers=True and self.opt.train.num_workers > 0,
                              pin_memory=True,
                              batch_size=1)
        else:
            return super().train_dataloader()

    def val_dataloader(self):
        if hasattr(self, "valset"):
            return DataLoader(self.valset,
                              shuffle=False,
                              num_workers=self.opt.val.num_workers,
                              persistent_workers=True and self.opt.val.num_workers > 0,
                              pin_memory=True,
                              batch_size=1)
        else:
            return super().test_dataloader()

    def test_dataloader(self):
        if hasattr(self, "testset"):
            return DataLoader(self.testset,
                              shuffle=False,
                              num_workers=self.opt.test.num_workers,
                              persistent_workers=True and self.opt.test.num_workers > 0,
                              pin_memory=True,
                              batch_size=1)
        else:
            return super().test_dataloader()
