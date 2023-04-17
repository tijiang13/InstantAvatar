import glob
import os
import numpy as np
import cv2
from dataclasses import dataclass
from aitviewer.viewer import Viewer
from aitviewer.headless import HeadlessRenderer
from aitviewer.scene.camera import OpenCVCamera
from aitviewer.renderables.billboard import Billboard
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.models.smpl import SMPLLayer
from aitviewer.configuration import CONFIG as C
from aitviewer.utils.so3 import aa2rot_numpy


@dataclass
class OPENPOSE_SKELETON:
    PARTS = [
        (0, 1), (0, 15), (15, 17), (0, 16), (16, 18), (1, 8), (8, 9), (9, 10), (10, 11),
        (11, 22), (22, 23), (11, 24), (8, 12), (12, 13), (13, 14), (14, 21), (14, 19),
        (19, 20), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
    ]
    JOINTS = [
        "Nose", "Neck", "(R) Shoulder", "(R) Elbow", "(R) Wrist", "(L) Shoulder", "(L) Elbow",
        "(L) Wrist", "Mid Hip", "(R) Hip", "(R) Knee", "(R) Ankle", "(L) Hip", "(L) Knee",
        "(L) Ankle", "(R) Eye", "(L) Eye", "(R) Ear", "(L) Ear", "(L) B. Toe", "(L) S. Toe",
        "(L) Heel", "(R) B. Toe", "(R) S. Toe", "(R) Heel",
    ]
    COLORS = [
        (255, 0, 85), (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
        (85, 255, 0), (0, 255, 0), (255, 0, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
        (0, 170, 255), (0, 85, 255), (0, 0, 255), (255, 0, 170), (170, 0, 255), (255, 0, 255),
        (85, 0, 255), (0, 0, 255), (0, 0, 255), (0, 0, 255), (0, 255, 255), (0, 255, 255),
        (0, 255, 255)
    ]


def load_smpl_param(path):
    smpl_params = dict(np.load(str(path)))
    if "thetas" in smpl_params:
        smpl_params["body_pose"] = smpl_params["thetas"][..., 3:]
        smpl_params["global_orient"] = smpl_params["thetas"][..., :3]
    return {
        "betas": smpl_params["betas"].astype(np.float32),
        "body_pose": smpl_params["body_pose"].astype(np.float32),
        "global_orient": smpl_params["global_orient"].astype(np.float32),
        "transl": smpl_params["transl"].astype(np.float32),
    }


def make_draw_func(keypoints=None, msk_paths=None, threshold=0.2):
    def _draw_func(img, current_frame_id):
        if keypoints is not None:
            kp = keypoints[current_frame_id]

            for i in range(len(OPENPOSE_SKELETON.JOINTS)):
                if kp[i, 2] > threshold:
                    x, y = kp[i, :2]
                    cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)

            for i, (x, y) in enumerate(OPENPOSE_SKELETON.PARTS):
                color = OPENPOSE_SKELETON.COLORS[i]
                if kp[x, 2] > threshold and kp[y, 2] > threshold:
                    cv2.line(img, tuple(kp[x, :2].astype(np.int32)),
                             tuple(kp[y, :2].astype(np.int32)), color, 2)

        if msk_paths is not None:
            if msk_paths[current_frame_id].endswith(".png"):
                msk = cv2.imread(msk_paths[current_frame_id], cv2.IMREAD_GRAYSCALE)
            else:
                msk = np.load(msk_paths[current_frame_id])
            # img[msk == 0] = (0, 255, 0)
        return img
    return _draw_func


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--gender", type=str, default="male")
    parser.add_argument("--pose", type=str, default=None)
    parser.add_argument("--openpose_threshold", type=float, default=0.2)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    camera = dict(np.load(f"{args.path}/cameras.npz"))
    img_paths = sorted(glob.glob(f"{args.path}/images/*"))
    msk_paths = sorted(glob.glob(f"{args.path}/masks/*.npy"))
    if len(msk_paths) == 0:
        msk_paths = sorted(glob.glob(f"{args.path}/masks/*.png"))
    keypoints = np.load(f"{args.path}/keypoints.npy")
    if args.pose and os.path.exists(args.pose):
        smpl_params = load_smpl_param(args.pose)
    else:
        smpl_params = load_smpl_param(f"{args.path}/poses.npz")

    if args.headless:
        viewer = HeadlessRenderer()
    else:
        viewer = Viewer()

    # load camera
    intrinsic = camera["intrinsic"]
    extrinsic = camera["extrinsic"]
    extrinsic[1:] *= -1
    H = camera["height"]
    W = camera["width"]
    cam = OpenCVCamera(intrinsic, extrinsic[:3], W, H, viewer=viewer)
    viewer.scene.add(cam)

    # load images
    draw_func = make_draw_func(keypoints, msk_paths, threshold=args.openpose_threshold)
    pc = Billboard.from_camera_and_distance(cam, 8.0, W, H, img_paths,
                                            image_process_fn=draw_func)
    viewer.scene.add(pc)

    # load poses
    smpl_layer = SMPLLayer(model_type='smpl',
                           gender=args.gender,
                           device=C.device)
    smpl_seq = SMPLSequence(poses_body=smpl_params["body_pose"],
                            smpl_layer=smpl_layer,
                            poses_root=smpl_params["global_orient"],
                            betas=smpl_params["betas"],
                            trans=smpl_params["transl"],
                            rotation=aa2rot_numpy(np.array([1, 0, 0]) * np.pi))
    viewer.scene.add(smpl_seq)

    # viewr settings
    viewer.set_temp_camera(cam)
    viewer.scene.floor.enabled = False
    viewer.scene.origin.enabled = False
    viewer.shadows_enabled = False

    if args.headless:
        viewer.save_video(video_dir=f"{args.path}/output.mp4", output_fps=args.fps)
    else:
        viewer.run()
