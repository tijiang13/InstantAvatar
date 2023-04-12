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

class MocapDataset(torch.utils.data.Dataset):
    def __init__(self, root, subject, split, opt):
        camera = np.load(str(root / "cameras.npz"))
        K = camera["intrinsic"]
        c2w = np.linalg.inv(camera["extrinsic"])
        height = camera["height"]
        width = camera["width"]

        self.downscale = opt.downscale
        if self.downscale > 1:
            height = height // self.downscale
            width = width // self.downscale
            K[:2] /= self.downscale

        self.rays_o, self.rays_d = make_rays(K, c2w, height, width)

        # prepare image and mask
        start = opt.start
        end = opt.end + 1
        skip = opt.get("skip", 1)
        self.img_lists = sorted(glob.glob(f"{root}/images/*.png"))[start:end:skip]
        self.msk_lists = sorted(glob.glob(f"{root}/masks/*.png"))[start:end:skip]

        # prepare SMPL params
        self.smpl_params = load_smpl_param(root / "poses.npz")
        for k, v in self.smpl_params.items():
            if k != "betas":
                self.smpl_params[k] = v[start:end:skip]

        self.split = split
        self.num_samples = opt.num_samples
        self.downscale = opt.downscale
        self.near = opt.get("near", None)
        self.far = opt.get("far", None)
        self.image_shape = (height, width)

    def get_SMPL_params(self):
        return {
            k: torch.from_numpy(v.copy()) for k, v in self.smpl_params.items()
        }

    def __len__(self):
        return len(self.img_lists)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_lists[idx])
        msk = (cv2.imread(self.msk_lists[idx])[..., 0] > 0).astype(np.float32)

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
            
        # prepare NeRF data
        rays_o = self.rays_o
        rays_d = self.rays_d
        if self.split == "train":
            # fg & bg
            kernel_size = 32 // self.downscale
            kernel = np.ones((kernel_size, kernel_size), np.uint8)

            msk_i = cv2.erode(msk, kernel)
            msk_o = cv2.dilate(msk, kernel)
            msk_e = msk_o - msk_i

            img = img.reshape(-1, 3)
            msk = msk.reshape(-1)
            bg_color = bg_color.reshape(-1, 3)
            msk_e = msk_e.reshape(-1)

            mask_loc, *_ = np.where(msk)
            edge_loc, *_ = np.where(msk_e)

            n1 = int(self.num_samples * 0.6)
            n2 = int(self.num_samples * 0.3)
            n3 = self.num_samples - n1 - n2

            mask_idx = np.random.randint(0, len(mask_loc), n1)
            edge_idx = np.random.randint(0, len(edge_loc), n2)
            rand_idx = np.random.randint(0, len(img), n3)
            indices = np.concatenate([rand_idx, edge_loc[edge_idx], mask_loc[mask_idx]], axis=0)

            img = img[indices]
            msk = msk[indices]
            bg_color = bg_color[indices]
            rays_o = rays_o[indices]
            rays_d = rays_d[indices]
        else:
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
            dist = np.sqrt(np.square(self.smpl_params["transl"][idx] - rays_o).sum(-1))
            datum["near"] = np.ones_like(rays_d[..., 0]) * (dist - 1)
            datum["far"] = np.ones_like(rays_d[..., 0]) * (dist + 1)
        return datum


class MocapDataModule(pl.LightningDataModule):
    def __init__(self, opt, **kwargs):
        super().__init__()

        data_dir = Path(hydra.utils.to_absolute_path(opt.dataroot))
        for split in ("train", "val", "test"):
            dataset = MocapDataset(data_dir, opt.subject, split, opt.get(split))
            setattr(self, f"{split}set", dataset)
        self.opt = opt

    def get_dataloader(self, split, opt):
        return DataLoader(getattr(self, f"{split}set"),
                          shuffle=(split=="train"),
                          num_workers=opt.num_workers,
                          persistent_workers=True and opt.num_workers > 0,
                          pin_memory=True,
                          batch_size=1)

    def train_dataloader(self):
        return self.get_dataloader("train", self.opt.train)

    def val_dataloader(self):
        return self.get_dataloader("val", self.opt.val)

    def test_dataloader(self):
        return self.get_dataloader("test", self.opt.test)
