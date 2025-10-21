#!/bin/bash

# TYPES=("Default" "alpha_1e5_001" "alpha_1e5_005" "alpha_1e5_010" "alpha_1e6_001" "alpha_1e6_005" "alpha_1e6_010" "alpha_5e5_001" "alpha_5e5_005" "alpha_5e5_010")
TYPES=("alpha_128" "alpha_256" "alpha_512")  

for TYPE in "${TYPES[@]}"
do
    echo "Submitting SLURM job for final_merge.sh with TYPE=$TYPE"
    sbatch /data/horse/ws/hama901h-BFTranslation/scripts/final_merge.sh $TYPE
done
