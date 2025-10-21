#!/bin/bash
#SBATCH --job-name=${RANK}-${MODEL_NAME}-${STEP}
#SBATCH --nodes=1        
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4          
#SBATCH --cpus-per-task=4        
#SBATCH --mem=64G                
#SBATCH --time=23:50:00          
#SBATCH --partition=alpha  

source /home/hama901h/miniconda3/etc/profile.d/conda.sh
conda activate /home/hama901h/miniconda3/envs/dolma
cd /data/horse/ws/hama901h-BFTranslation/

BASE_PATH="/data/horse/ws/hama901h-BFTranslation/checkpoints/meta-llama/Llama-3.1-8B"
LORA_MODEL_DIR="${BASE_PATH}/tulu3/w_checkpoints/${RANK}"
# RANK, MODEL_NAME and STEP are now passed as environment variables
LORA_MODEL_DIR="${LORA_MODEL_DIR}/${MODEL_NAME}"
ADAPTER_DIR="${LORA_MODEL_DIR}/step_${STEP}/"

# Define evaluation tasks (e.g., hellaswag, arc_challenge, mmlu)
# Remove gsm8k,gsm_plus to avoid generate_until requests
EVAL_TASKS="mmlu,hellaswag,arc_easy,arc_challenge,piqa,commonsense_qa,winogrande"

export CUDA_LAUNCH_BLOCKING=1
BATCH_SIZE=1

echo "Starting evaluation with LoRA model..."
echo "Rank: $RANK"
echo "Base Model: $BASE_PATH"
echo "LoRA Adapter: $ADAPTER_DIR"
echo "Tasks: $EVAL_TASKS"

# Check if adapter directory exists
if [[ ! -d "$ADAPTER_DIR" ]]; then
    echo "Error: Adapter directory does not exist: $ADAPTER_DIR"
    exit 1
fi

lm_eval --model hf \
    --model_args pretrained=$ADAPTER_DIR,trust_remote_code=True,parallelize=True,offload_folder='/data/horse/ws/hama901h-BFTranslation/lm-evaluation-harness/offload_folder',max_memory_per_gpu="20GB",tokenizer=$ADAPTER_DIR,use_fast_tokenizer=False \
    --tasks "$EVAL_TASKS" \
    --batch_size "$BATCH_SIZE" \
    --output_path "$ADAPTER_DIR/eval_results.json"