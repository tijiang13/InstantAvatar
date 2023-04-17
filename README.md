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

<img src="./media/peoplesnapshot/male-3-casual.gif" width="270" height="270">
<img src="./media/peoplesnapshot/female-4-casual.gif" width="270" height="270">

## Play with Your Own Video
Here we use the in the wild video provided by [Neuman](https://github.com/apple/ml-neuman) as an example:

1. create a yaml file specifying the details about the sequence in `./confs/dataset/`. In this example it's provided in `./confs/dataset/neuman/seattle.yaml`.
2. download the data from [Neuman's Repo](https://github.com/apple/ml-neuman), and run `cp <path-to-neuman-dataset>/seattle/images ./data/custom/seattle/`
3. run the bash script `bash ./bash/run-neuman-demo.sh`

<img src="./media/neuman/lab.gif" width="270" height="270">
<img src="./media/neuman/seattle.gif" width="270" height="270">

And you can animate the avatar easily:

<img src="./media/neuman/seattle-dance.gif" width="270" height="270">


## Citation
```
@article{jiang2022instantavatar,
  author    = {Jiang, Tianjian and Chen, Xu and Song, Jie and Hilliges, Otmar},
  title     = {InstantAvatar: Learning Avatars from Monocular Video in 60 Seconds},
  journal   = {arXiv},
  year      = {2022},
}
```
