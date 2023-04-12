experiment="fitting"
SEQUENCES=("male-3-sport")
for SEQUENCE in ${SEQUENCES[@]}; do
    dataset="peoplesnapshot/$SEQUENCE"
    python fit.py --config-name SNARF_NGP_fitting dataset=$dataset experiment=$experiment
    python train.py --config-name SNARF_NGP dataset=$dataset experiment=$experiment
done
