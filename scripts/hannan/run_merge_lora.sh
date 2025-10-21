#!/bin/bash

START_STEP=6000
END_STEP=54000
STEP_INCREMENT=6000

# TYPES=("alpha_1e5_001" "alpha_1e5_005" "alpha_1e5_010" "alpha_1e6_001" "alpha_1e6_005" "alpha_1e6_010" "alpha_5e5_001" "alpha_5e5_005" "alpha_5e5_010" "Default")
# TYPES=("Default")
# TYPES=("alpha_1e5_005")
TYPES=("128" "256" "512")
for TYPE in "${TYPES[@]}"
do
    for (( STEP=$START_STEP; STEP<=$END_STEP; STEP+=$STEP_INCREMENT ))
        do
            echo "Submitting SLURM job for merge_lora.sh with STEP=$STEP TYPE=$TYPE"
            sbatch /data/horse/ws/hama901h-BFTranslation/scripts/merge_lora.sh $STEP $TYPE
    done
done
