import glob
from tqdm import tqdm
import cv2
import numpy as np
import romp
import os

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()
    settings = romp.main.default_settings 
    romp_model = romp.ROMP(settings)
    results = []
    
    # Use a list of common image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    # Modify the glob pattern to use os.path.join and check file extensions
    image_files = [
        p for p in sorted(glob.glob(os.path.join(args.data_dir, "images", "*")))
        if os.path.splitext(p)[1].lower() in image_extensions
    ]
    
    for p in tqdm(image_files):
        print(p)
        img = cv2.imread(p)
        result = romp_model(img)
        if result["body_pose"].shape[0] > 1:
            result = {k: v[0:1] for k, v in result.items()}
        results.append(result)
    
    results = {
        k: np.concatenate([r[k] for r in results], axis=0) for k in result
    }
    np.savez(os.path.join(args.data_dir, "poses.npz"), **{
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
    np.savez(os.path.join(args.data_dir, "cameras.npz"), **{
        "intrinsic": K,
        "extrinsic": np.eye(4),
        "height": img.shape[0],
        "width": img.shape[1],
    })
