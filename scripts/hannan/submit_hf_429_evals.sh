#!/bin/bash

# Parse configurations from hf_429_err_files_script.txt
# Extract unique combinations of Rank, model_name, and step
CONFIGS=(
    "Rank64 Default 8000"
    "Rank64 Default 10000"
    "Rank64 Default 28000"
    "Rank1024 alpha_5e5_010 32000"
)

# Loop through each configuration
for config in "${CONFIGS[@]}"; do
    # Parse the configuration
    read -r RANK MODEL_NAME STEP <<< "$config"
    
    # Skip Final checkpoints for now (they need different handling)
    # if [[ "$STEP" == "Final" ]]; then
    #     echo "Skipping Final checkpoint for $RANK/$MODEL_NAME"
    #     continue
    # fi
    
    OUTPUT_FILE="lm_eval_results/${RANK}/${MODEL_NAME}-${STEP}.out"
    ERROR_FILE="lm_eval_results/${RANK}/${MODEL_NAME}-${STEP}"
    
    echo "Submitting job for RANK=$RANK MODEL_NAME=$MODEL_NAME STEP=$STEP"
    
    # Submit the job with the specific rank configuration
    sbatch --export=RANK=$RANK,MODEL_NAME=$MODEL_NAME,STEP=$STEP \
           --job-name="${RANK}-${MODEL_NAME}-${STEP}" \
           --output="$OUTPUT_FILE" \
           --error="$ERROR_FILE" \
           hf_429_eval.sh
done