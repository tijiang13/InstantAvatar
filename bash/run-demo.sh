experiment="demo"
SEQUENCES=("male-3-casual")
for SEQUENCE in ${SEQUENCES[@]}; do
    dataset="peoplesnapshot/$SEQUENCE"
    python train.py --config-name demo dataset=$dataset experiment=$experiment
    python animate.py --config-name demo dataset=$dataset experiment=$experiment
done
