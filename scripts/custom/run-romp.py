import glob
from tqdm import tqdm
import cv2
import numpy as np
import romp


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    settings = romp.main.default_settings 
    romp_model = romp.ROMP(settings)

    results = []
    for p in tqdm(sorted(glob.glob(f"{args.data_dir}/images/*"))):
        img = cv2.imread(p)
        result = romp_model(img)
        results.append(result)

    results = {
        k: np.concatenate([r[k] for r in results], axis=0) for k in result
    }

    np.savez(f"{args.data_dir}/poses.npz", **{
        "betas": results["smpl_betas"].mean(axis=0),
        "global_orient": results["smpl_thetas"][:, :3],
        "body_pose": results["smpl_thetas"][:, 3:],
        "transl": results["cam_trans"],
    })

    # ROMP assumes FOV=60
    fov = 60
    f = max(img.shape[:2]) / 2 * 1 / np.tan(np.radians(fov/2))
    K = np.eye(3)
    K[0, 0] = K[1, 1] = f
    K[0, 2] = img.shape[1] / 2
    K[1, 2] = img.shape[0] / 2
    np.savez(f"{args.data_dir}/cameras.npz", **{
        "intrinsic": K,
        "extrinsic": np.eye(4),
        "height": img.shape[0],
        "width": img.shape[1],
    })