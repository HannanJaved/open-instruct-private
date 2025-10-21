#!/bin/bash
#SBATCH --job-name=merge-test
#SBATCH --output=merge-test.out
#SBATCH --error=merge-test.err
#SBATCH --nodes=1        
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1          
#SBATCH --cpus-per-task=4        
#SBATCH --mem=32G                
#SBATCH --time=01:00:00          
#SBATCH --partition=capella

echo "Starting LoRA merge job..."
echo "Job ID: $SLURM_JOB_ID"

module load CUDA

source /home/hama901h/miniconda3/etc/profile.d/conda.sh
conda activate /home/hama901h/miniconda3/envs/dolma

cd /data/horse/ws/hama901h-BFTranslation/

TYPE=$1
# TYPE="alpha_1e5_010" 
# STEP="2000"
BASE_MODEL_DIR="/data/horse/ws/hama901h-BFTranslation/checkpoints/meta-llama/Llama-3.1-8B"

# ADAPTER_DIR="${BASE_MODEL_DIR}/tulu3/Rank256/${TYPE}/step_${STEP}/"
# ADAPTER_DIR="${BASE_MODEL_DIR}/tulu3/w_checkpoints/Rank256/${TYPE}/"
ADAPTER_DIR="${BASE_MODEL_DIR}/tulu3/w_checkpoints/Rank256/alpha_5e5_010/${TYPE}/"

# OUTPUT_DIR="${BASE_MODEL_DIR}/tulu3/w_checkpoints/Rank256/${TYPE}/step_${STEP}/"
# OUTPUT_DIR="${BASE_MODEL_DIR}/tulu3/w_checkpoints/Rank256/${TYPE}/final/"

OUTPUT_DIR="${BASE_MODEL_DIR}/tulu3/w_checkpoints/Rank256/alpha_5e5_010/${TYPE}/final/"
# VOCAB_SIZE=128256
VOCAB_SIZE=128264

# Run the Python script with the specified arguments
python scripts/merge_lora.py \
    --base-model "$BASE_MODEL_DIR" \
    --adapter "$ADAPTER_DIR" \
    --output "$OUTPUT_DIR" \
    --vocab-size $VOCAB_SIZE

echo "Merge job completed successfully."