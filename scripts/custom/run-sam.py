from segment_anything import SamPredictor, sam_model_registry
import cv2
import numpy as np
import glob
import os


CHECKPOINT = os.path.expanduser("~/third_party/segment-anything/ckpts/sam_vit_h_4b8939.pth")
MODEL = "vit_h"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    args = parser.parse_args()

    sam = sam_model_registry[MODEL](checkpoint=CHECKPOINT)
    sam.to("cuda")
    predictor = SamPredictor(sam)

    root = args.data_dir
    img_lists = sorted(glob.glob(f"{root}/images/*.png"))
    keypoints = np.load(f"{root}/keypoints.npy")
    os.makedirs(f"{root}/masks_sam", exist_ok=True)
    os.makedirs(f"{root}/masks_sam_images", exist_ok=True)
    for fn, pts in zip(img_lists, keypoints):
        img = cv2.imread(fn)
        predictor.set_image(img)
        m = pts[..., -1] > 0.5
        pts = pts[m]
        masks, _, _ = predictor.predict(pts[:, :2], np.ones_like(pts[:, 0]))
        mask = masks.sum(axis=0) > 0
        cv2.imwrite(fn.replace("images", "masks_sam"), mask * 255)

        img[~mask] = 0
        cv2.imwrite(fn.replace("images", "masked_sam_images"), img)
