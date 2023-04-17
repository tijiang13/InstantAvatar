# InstantAvatar
[[Project Page]](https://tijiang13.github.io/InstantAvatar/)

## Prepare Data
```bash
# Step 1: Download data from: https://graphics.tu-bs.de/people-snapshot
# Step 2: Preprocess using our script
python scripts/peoplesnapshot/preprocess_PeopleSnapshot.py --root <PATH_TO_PEOPLESNAPSHOT> --subject male-3-casual

# Step 3: Download SMPL from: https://smpl.is.tue.mpg.de/ and place the model in ./data/SMPLX/smpl/
# └── SMPLX/smpl/
#         ├── SMPL_FEMALE.pkl
#         ├── SMPL_MALE.pkl
#         └── SMPL_NEUTRAL.pkl
```

## Quick Start
Quickly learn and animate an avatar with `bash ./bash/run-demo.sh`

<p float="left">
<img src="./media/peoplesnapshot/male-3-casual.gif" width="270" height="270">
<img src="./media/peoplesnapshot/female-4-casual.gif" width="270" height="270">
</p>

## Play with Your Own Video
Here we use the in the wild video provided by [Neuman](https://github.com/apple/ml-neuman) as an example:

1. create a yaml file specifying the details about the sequence in `./confs/dataset/`. In this example it's provided in `./confs/dataset/neuman/seattle.yaml`.
2. download the data from [Neuman's Repo](https://github.com/apple/ml-neuman), and run `cp <path-to-neuman-dataset>/seattle/images ./data/custom/seattle/`
3. run the bash script `bash scripts/custom/process-sequence.sh ./data/custom/seattle neutral` to preprocess the images, which
  1. uses [Openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) to estimate the 2D keypoints,
  2. uses [Segment-Anything](https://github.com/facebookresearch/segment-anything) to segment the scene
  3. uses [ROMP](https://github.com/Arthur151/ROMP) to estimate camera and smpl parameters
4. run the bash script `bash ./bash/run-neuman-demo.sh` to learn an avatar

<p float="left">
  <img src="./media/neuman/lab-input.gif" width="426" height="270">
  <img src="./media/neuman/lab.gif" width="270" height="270">
</p>

<p float="left">
  <img src="./media/neuman/seattle-input.gif" width="426" height="270">
  <img src="./media/neuman/seattle.gif" width="270" height="270">
</p>

And you can animate the avatar easily:

<img src="./media/neuman/seattle-dance.gif" width="270" height="270">

## Acknowledge
We would like to acknowledge the following third-party repositories we used in this project:
- [[Tinycuda-nn]](https://github.com/NVlabs/tiny-cuda-nn)
- [[SMPLX]](https://github.com/vchoutas/smplx)
- [[Openpose]](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
- [[ROMP]](https://github.com/Arthur151/ROMP)
- [[Segment-anything]](https://github.com/facebookresearch/segment-anything)
- [Fast-SNARF](https://github.com/xuchen-ethz/fast-snarf)

We are grateful to the developers and contributors of these repositories for their hard work and dedication to the open-source community. Without their contributions, our project would not have been possible.

## Citation
```
@article{jiang2022instantavatar,
  author    = {Jiang, Tianjian and Chen, Xu and Song, Jie and Hilliges, Otmar},
  title     = {InstantAvatar: Learning Avatars from Monocular Video in 60 Seconds},
  journal   = {arXiv},
  year      = {2022},
}
```
