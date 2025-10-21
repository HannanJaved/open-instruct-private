#!/bin/bash

# Define the range of steps and model names
START_STEP=2000
END_STEP=54000
STEP_SIZE=2000
# MODEL_NAMES=("Default" "alpha_1e5_001" "alpha_1e5_005" "alpha_1e5_010" "alpha_1e6_001" "alpha_1e6_005" "alpha_1e6_010" "alpha_5e5_001" "alpha_5e5_005" "alpha_5e5_010")
MODEL_NAMES=("Default")

# Loop through each model name and step
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    for ((STEP=$START_STEP; STEP<=$END_STEP; STEP+=STEP_SIZE)); do
        OUTPUT_FILE="lm_eval_results/Rank256/${MODEL_NAME}-${STEP}.out"
        ERROR_FILE="lm_eval_results/Rank256/${MODEL_NAME}-${STEP}.err"

        echo "Submitting job for MODEL_NAME=$MODEL_NAME and STEP=$STEP"

        # Pass the output and error file names directly to sbatch
        sbatch --export=MODEL_NAME=$MODEL_NAME,STEP=$STEP \
               --job-name="Rank256-${MODEL_NAME}-${STEP}" \
               --output="$OUTPUT_FILE" \
               --error="$ERROR_FILE" \
               tulu_eval.sh
    done
done
