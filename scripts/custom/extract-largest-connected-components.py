import os
import cv2
import glob
import numpy as np

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(os.path.join(args.data_dir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(args.data_dir, "masked_images"), exist_ok=True)
    for fn in glob.glob(f"{args.data_dir}/masks_sam/*.png"):
        img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)

        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
        # _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        largest_component_index = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        largest_component_mask = (labels == largest_component_index).astype(np.uint8) * 255

        cv2.imwrite(fn.replace("masks_sam", "masks"), largest_component_mask)

        mask = largest_component_mask > 0
        img = cv2.imread(fn.replace("masks_sam", "images"))
        img[~mask] = 0
        cv2.imwrite(fn.replace("masks_sam", "masked_images"), img)
