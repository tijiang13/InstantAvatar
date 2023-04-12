# InstantAvatar
[[Project Page]](https://tijiang13.github.io/InstantAvatar/)

## Prepare Data
```bash
# Step 1: Download data from: https://graphics.tu-bs.de/people-snapshot
# Step 2: Preprocess using our script
python scripts/peoplesnapshot/preprocess_PeopleSnapshot.py --root <PATH_TO_PEOPLESNAPSHOT> --subject male-3-casual

# Step 3: Download SMPL from: https://smpl.is.tue.mpg.de/
```

If everything setups properly, the layout of `data/` folder should be:
```
./data/
├── animation
│   └── aist_demo.npz
├── PeopleSnapshot
│   ├── female-1-casual
│   ├── female-3-casual
│   ├── male-2-casual
│   └── male-3-casual
└── SMPLX
    └── smpl
        ├── SMPL_FEMALE.pkl
        ├── SMPL_MALE.pkl
        └── SMPL_NEUTRAL.pkl
```

## Demo 
Quickly learn and animate an avatar with optimized poses. 

```bash
bash ./bash/run-demo.sh
```

## Evaluation
Evaluate the results on 4 sequences of peoplesnapshot with optimized poses (`male-3-casual`, `male-4-casual`, `female-3-casual`, `female-4-casual`)

```bash
bash ./bash/run-peoplesnapshot.sh
```

We also support fitting SMPL poses on the fly, when the provided SMPL registration is not perfect (however it will converge slower, ~10 mins on a 3090 card)
```bash
bash ./bash/run-fitting.sh
```

## Citation
```
@article{jiang2022instantavatar,
  author    = {Jiang, Tianjian and Chen, Xu and Song, Jie and Hilliges, Otmar},
  title     = {InstantAvatar: Learning Avatars from Monocular Video in 60 Seconds},
  journal   = {arXiv},
  year      = {2022},
}
```
