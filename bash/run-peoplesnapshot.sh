experiment="baseline"
SEQUENCES=("female-3-casual" "male-4-casual" "male-3-casual" "female-4-casual")
for SEQUENCE in ${SEQUENCES[@]}; do
    dataset="peoplesnapshot/$SEQUENCE"
    python train.py --config-name SNARF_NGP dataset=$dataset experiment=$experiment train.max_epochs=50
    python eval.py --config-name SNARF_NGP_refine dataset=$dataset experiment=$experiment
done