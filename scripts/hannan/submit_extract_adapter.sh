#!/bin/bash

# Base paths
BASE_MODEL_DIR=/data/horse/ws/hama901h-BFTranslation/checkpoints/meta-llama/Llama-3.1-8B
    
BASE_CHECKPOINT_PATH="${BASE_MODEL_DIR}/tulu3/w_checkpoints/Rank256/alpha_5e5_010/"
BASE_ADAPTER_SAVE_PATH="${BASE_MODEL_DIR}/tulu3/w_checkpoints/Rank256/alpha_5e5_010/"

# Alpha configurations to iterate over 
# ALPHA_CONFIGS=("alpha_128")
# ALPHA_CONFIGS=("alpha_256") 
ALPHA_CONFIGS=("alpha_512")
# ALPHA_CONFIGS=("alpha_1e5_001" "alpha_1e5_005" "alpha_1e5_010" "alpha_1e6_001" "alpha_1e6_005" "alpha_1e6_010" "alpha_5e5_001" "alpha_5e5_005" "alpha_5e5_010" "Default")
# ALPHA=("Default")


# Iterate over each alpha configuration
for ALPHA in "${ALPHA_CONFIGS[@]}"; do
    # Iterate over steps from 2000 to 54000 with a step size of 2000
    # for STEP in $(seq 24000 2000 24000); do
    for STEP in $(seq 6000 6000 54000); do
        export CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH}/${ALPHA}/step_${STEP}"
        export ADAPTER_SAVE_PATH="${BASE_ADAPTER_SAVE_PATH}/${ALPHA}/step_${STEP}"
        export LORA_RANK=256
        export LORA_ALPHA=512
        export LORA_DROPOUT=0.1

        echo "Submitting job for ${ALPHA}, step ${STEP}, with rank ${LORA_RANK} and alpha ${LORA_ALPHA}..."
        sbatch scripts/extract_adapter.sh
    done
done

echo "All jobs submitted successfully."
