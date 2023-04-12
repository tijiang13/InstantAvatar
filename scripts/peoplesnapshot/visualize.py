import glob
import cv2
import numpy as np


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()

    imgs = sorted(glob.glob(f"{args.path}/images/*.png"))
    msks = sorted(glob.glob(f"{args.path}/masks/*.npy"))

    for img, mask in zip(imgs, msks):
        img = cv2.imread(img)
        mask = np.load(mask)

        img[mask == 0] = 0
        cv2.imshow("vis", img)
        cv2.waitKey(10)

