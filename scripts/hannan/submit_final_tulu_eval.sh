#!/bin/bash

MODEL_NAMES=("Default" "alpha_1e5_001" "alpha_1e5_005" "alpha_1e5_010" "alpha_1e6_001" "alpha_1e6_005" "alpha_1e6_010" "alpha_5e5_001" "alpha_5e5_005" "alpha_5e5_010")

# Loop through each model name and step
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    OUTPUT_FILE="lm_eval_results/Rank1024/${MODEL_NAME}-Final.out"
    ERROR_FILE="lm_eval_results/Rank1024/${MODEL_NAME}-Final.err"

    echo "Submitting job for MODEL_NAME=$MODEL_NAME for Final checkpoint"

    # Pass the output and error file names directly to sbatch
    sbatch --export=MODEL_NAME=$MODEL_NAME \
            --job-name="Rank1024-${MODEL_NAME}-Final" \
            --output="$OUTPUT_FILE" \
            --error="$ERROR_FILE" \
            tulu_final_eval.sh
done
