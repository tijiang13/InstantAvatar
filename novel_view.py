import glob
import os
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import hydra
from tqdm import tqdm
import imageio


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

class AnimateDataset(torch.utils.data.Dataset):
    def __init__(self, num_frames, betas, downscale=1):
        H = 1080
        W = 1080

        K = np.eye(3)
        K[0, 0] = K[1, 1] = 2000
        K[0, 2] = H // 2
        K[1, 2] = W // 2
        
        if downscale > 1:
            H = H // downscale
            W = W // downscale
            K[:2] /= downscale
        self.H = H
        self.W = W

        c2w = np.eye(4)
        self.rays_o, self.rays_d = make_rays(K, c2w, H, W)

        global_orient = np.array([[np.pi, 0, 0]])
        body_pose = np.zeros((1, 69))
        body_pose[:, 2] = 0.5
        body_pose[:, 5] = -0.5
        transl = np.array([[0, 0.5, 5]])

        self.betas = betas.astype(np.float32)
        self.body_pose = body_pose.astype(np.float32)
        self.global_orient = global_orient.astype(np.float32)
        self.transl = transl.astype(np.float32)
        self.num_frames = num_frames

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        # prepare NeRF data
        rays_o = self.rays_o
        rays_d = self.rays_d

        datum = {
            # NeRF
            "rays_o": rays_o,
            "rays_d": rays_d,

            # SMPL parameters
            "betas": self.betas.reshape(10),
            "global_orient": self.global_orient[0],
            "body_pose": self.body_pose[0],
            "transl": self.transl[0],
        }

        angle = 2 * np.pi * idx / self.num_frames
        R = cv2.Rodrigues(np.array([0, angle, 0]))[0]
        R_gt = cv2.Rodrigues(datum["global_orient"])[0]
        R_gt = R @ R_gt
        R_gt = cv2.Rodrigues(R_gt)[0].astype(np.float32)
        datum["global_orient"] = R_gt.reshape(3)

        # distance from camera (0, 0, 0) to midhip
        datum["near"] = np.ones_like(rays_d[..., 0]) * 0
        datum["far"] = np.ones_like(rays_d[..., 0]) * 10
        return datum


@hydra.main(config_path="./confs", config_name="SNARF_NGP")
def main(opt):
    pl.seed_everything(opt.seed)
    torch.set_printoptions(precision=6)
    print(f"Switch to {os.getcwd()}")

    datamodule = hydra.utils.instantiate(opt.dataset, _recursive_=False)
    model = hydra.utils.instantiate(opt.model, datamodule=datamodule, _recursive_=False)
    model = model.cuda()
    model.eval()

    checkpoints = sorted(glob.glob("checkpoints/*.ckpt"))
    print("Resume from", checkpoints[-1])
    checkpoint = torch.load(checkpoints[-1])
    model.load_state_dict(checkpoint["state_dict"])

    num_frames = 60
    dataset = AnimateDataset(num_frames,
                             betas=datamodule.trainset.smpl_params["betas"],
                             downscale=2)
    datamodule.testset.image_shape = (dataset.H, dataset.W)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    animation = "rotation"
    folder = f"animation/{animation}/"
    os.makedirs(folder, exist_ok=True)

    with torch.inference_mode():
        imgs = []
        for i, batch in tqdm(enumerate(dataloader)):
            batch = {k: v.cuda() for k, v in batch.items()}
            rgb, _, alpha, _ = model.render_image_fast(batch, (dataset.H, dataset.W))
            img = torch.cat([rgb, alpha[..., None]], dim=-1)
            imgs.append(img)
            cv2.imwrite("{}/{}.png".format(folder, i), (img.cpu().numpy() * 255).astype(np.uint8)[0])
    imgs = [(img.cpu().numpy() * 255).astype(np.uint8)[0] for img in imgs]
    imgs = [cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA) for img in imgs]
    imageio.mimsave(f"{folder}/../{animation}.gif", imgs, fps=30)


if __name__ == "__main__":
    main()
