import numpy as np
import smplx
from tqdm import tqdm
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.renderer import (
    PerspectiveCameras,
    MeshRenderer,
    MeshRasterizer,
    RasterizationSettings,
    BlendParams,
    SoftSilhouetteShader,
)
from pytorch3d.structures import Meshes


DEVICE = "cuda"


def optimize(optimizer, closure, max_iter=10):
    pbar = tqdm(range(max_iter))
    for i in pbar:
        loss = optimizer.step(closure)
        pbar.set_postfix_str(f"loss: {loss.detach().cpu().numpy():.6f}")


def project(projection_matrices, keypoints_3d):
    p = torch.einsum("ij,mnj->mni", projection_matrices[:3, :3], keypoints_3d) + projection_matrices[:3, 3]
    p = p[..., :2] / p[..., 2:3]
    return p


def build_renderer(camera, IMG_SIZE):
    K = camera["intrinsic"]
    K = torch.from_numpy(K).float().to(DEVICE)

    R = torch.eye(3, device=DEVICE)[None]
    R[:, 0] *= -1
    R[:, 1] *= -1
    t = torch.zeros(1, 3, device=DEVICE)


    cameras = PerspectiveCameras(
        focal_length=K[None, [0, 1], [0, 1]],
        principal_point=K[None, [0, 1], [2, 2]],
        R=R,
        T=t,
        image_size=[IMG_SIZE],
        in_ndc=False,
        device=DEVICE,
    )
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

    raster_settings = RasterizationSettings(
        image_size=IMG_SIZE,
        blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
        faces_per_pixel=100,
    )

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftSilhouetteShader(
            blend_params=blend_params
        )
    )
    return renderer

class BODY25JointMapper:
    SMPL_TO_BODY25 = [
        24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4, 7, 25, 26, 27, 28, 29,
        30, 31, 32, 33, 34
    ]

    def __init__(self):
        self.mapping = self.SMPL_TO_BODY25

    def __call__(self, smpl_output, *args, **kwargs):
        return smpl_output.joints[:, self.mapping]


HEATMAP_THRES = 0.30
PAF_THRES = 0.05
PAF_RATIO_THRES = 0.95
NUM_SAMPLE = 10
MIN_POSE_JOINT_COUNT = 4
MIN_POSE_LIMB_SCORE = 0.4
NUM_JOINTS = 25
BODY25_POSE_INDEX = [(0, 1), (14, 15), (22, 23), (16, 17), (18, 19), (24, 25),
                     (26, 27), (6, 7), (2, 3), (4, 5), (8, 9), (10, 11),
                     (12, 13), (30, 31), (32, 33), (36, 37), (34, 35),
                     (38, 39), (20, 21), (28, 29), (40, 41), (42, 43),
                     (44, 45), (46, 47), (48, 49), (50, 51)]
BODY25_PART_PAIRS = [(1, 8), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
                     (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14),
                     (1, 0), (0, 15), (15, 17), (0, 16), (16, 18), (2, 17),
                     (5, 18), (14, 19), (19, 20), (14, 21), (11, 22), (22, 23),
                     (11, 24)]

joints_name = [
    "Nose",         # 0  => 24 (SMPL)
    "Neck",         # 1  => 12
    "RShoulder",    # 2  => 17
    "RElbow",       # 3  => 19
    "RWrist",       # 4  => 21
    "LShoulder",    # 5  => 16
    "LElbow",       # 6  => 18
    "LWrist",       # 7  => 20
    "MidHip",       # 8  => 0
    "RHip",         # 9  => 2
    "RKnee",        # 10 => 5
    "RAnkle",       # 11 => 8
    "LHip",         # 12 => 1
    "LKnee",        # 13 => 4
    "LAnkle",       # 14 => 7
    "REye",         # 15 => 25
    "LEye",         # 16 => 26
    "REar",         # 17 => 27
    "LEar",         # 18 => 28
    "LBigToe",      # 19 => 29
    "LSmallToe",    # 20 => 30
    "LHeel",        # 21 => 31
    "RBigToe",      # 22 => 32
    "RSmallToe",    # 23 => 33
    "RHeel",        # 24 => 34
]

SELECT_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 
                 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]


import cv2
def draw_detect(frame: np.ndarray, poses2d: np.ndarray, color=(255, 255, 0)):
    for person in poses2d:
        # draw parts
        for (i, j) in BODY25_PART_PAIRS:
            if person.shape[-1] > 2 and min(person[[i, j], 2]) < 1e-3:
                continue
            frame = cv2.line(frame, tuple(person[i, :2].astype(int)),
                             tuple(person[j, :2].astype(int)), (color), 1)

        # draw joints
        for joint in person:
            if len(joint) > 2 and joint[-1] == 0:
                continue
            pos = joint[:2].astype(int)
            frame = cv2.circle(frame, tuple(pos), 2, (color), 2, cv2.FILLED)
    return frame

@torch.no_grad()
def main(root, gender, keypoints_threshold, use_silhouette, downscale=1):
    camera = dict(np.load(f"{root}/cameras.npz"))
    if downscale > 1:
        camera["intrinsic"][:2] /= downscale
    projection_matrices = camera["intrinsic"] @ camera["extrinsic"][:3]
    projection_matrices = torch.from_numpy(projection_matrices).float().to(DEVICE)

    # prepare data
    joint_mapper = BODY25JointMapper()
    smpl_params = dict(np.load(f"{root}/poses.npz"))
    keypoints_2d = np.load(f"{root}/keypoints.npy")
    keypoints_2d = torch.from_numpy(keypoints_2d).float().to(DEVICE)

    params = {}
    for k, v in smpl_params.items():
        if k == "thetas":
            tensor = torch.from_numpy(v[:, :3]).clone().to(DEVICE)
            params["global_orient"] = nn.Parameter(tensor)
            tensor = torch.from_numpy(v[:, 3:]).clone().to(DEVICE)
            params["body_pose"] = nn.Parameter(tensor)
        elif k == "betas":
            tensor = torch.from_numpy(v).clone().to(DEVICE)
            params[k] = nn.Parameter(tensor[None])
            # params[k] = tensor[None]
        else:
            tensor = torch.from_numpy(v).clone().to(DEVICE)
            params[k] = nn.Parameter(tensor)

    body_model = smplx.SMPL("./data/SMPLX/smpl", gender=gender)
    body_model.to(DEVICE)

    # optimize with keypoints
    optimizer = torch.optim.Adam(params.values(), lr=1e-3)
    def closure():
        optimizer.zero_grad()
        smpl_output = body_model(**params)
        keypoints_pred = project(projection_matrices, joint_mapper(smpl_output))
        
        error = (keypoints_2d[..., :2] - keypoints_pred).square().sum(-1).sqrt()

        m1 = (keypoints_2d[..., 2] > keypoints_threshold)
        mask = m1
        error = error * mask.float()

        loss = error[:, SELECT_JOINTS]
        loss = loss.mean()

        reg = (smpl_output.vertices[1:] - smpl_output.vertices[:-1]).square().sum(-1).sqrt()
        loss += reg.mean()

        loss.backward()
        return loss
    optimize(optimizer, closure, max_iter=200)

    if use_silhouette:
        masks = sorted(glob.glob(f"{root}/masks/*"))
        masks = [cv2.imread(p)[..., 0] for p in masks]
        if downscale > 1:
            masks = [cv2.resize(m, dsize=None, fx=1/downscale, fy=1/downscale) for m in masks]
        masks = np.stack(masks, axis=0)

        img_size = masks[0].shape[:2]
        renderer = build_renderer(camera, img_size)

        for i in range(len(masks)):
            mask = torch.from_numpy(masks[i:i+1]).float().to(DEVICE) / 255

            optimizer = torch.optim.LBFGS(params.values(), line_search_fn="strong_wolfe")
            def closure():
                optimizer.zero_grad()

                # silhouette loss
                smpl_output = body_model(
                    betas=params["betas"].clone().detach(),
                    global_orient=params["global_orient"][i:i+1],
                    body_pose=params["body_pose"][i:i+1],
                    transl=params["transl"][i:i+1],
                )

                # keypoints loss
                keypoints_pred = project(projection_matrices, joint_mapper(smpl_output))
                loss_keypoints = (keypoints_2d[i:i+1, :, :2] - keypoints_pred).square().sum(-1).sqrt()

                m1 = (keypoints_2d[i:i+1, :, 2] > 0)
                loss_keypoints = (loss_keypoints * m1.float()).mean()

                meshes = Meshes(
                    verts=smpl_output.vertices,
                    faces=body_model.faces_tensor[None].repeat(1, 1, 1),
                )
                silhouette = renderer(meshes)[..., 3]
                loss_silhouette = F.mse_loss(mask, silhouette)

                loss = loss_silhouette # + loss_keypoints
                loss.backward()
                return loss

            optimize(optimizer, closure, max_iter=10)

    smpl_params = dict(smpl_params)
    for k in smpl_params:
        if k == "betas":
            smpl_params[k] = params[k][0].detach().cpu().numpy()
        elif k == "thetas":
            smpl_params[k][:, :3] = params["global_orient"].detach().cpu().numpy()
            smpl_params[k][:, 3:] = params["body_pose"].detach().cpu().numpy()
        elif k == "body_pose":
            smpl_params[k] = params[k].detach().cpu().numpy()
            smpl_params[k][:, -12:] = 0
        else:
            smpl_params[k] = params[k].detach().cpu().numpy()
    np.savez(f"{root}/poses_optimized.npz", **smpl_params)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--gender", type=str, default="male")
    parser.add_argument("--keypoints-threshold", type=float, default=0.2)
    parser.add_argument("--silhouette", action="store_true")
    parser.add_argument("--downscale", type=float, default=1.0)
    args = parser.parse_args()
    main(args.data_dir, args.gender, args.keypoints_threshold, args.silhouette, args.downscale)
