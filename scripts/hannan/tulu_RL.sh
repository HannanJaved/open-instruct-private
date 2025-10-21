#!/bin/bash
#SBATCH --job-name=256_5e5_010
#SBATCH --output=logs/Tulu-RL/Rank256/256_5e5_010.out
#SBATCH --error=logs/Tulu-RL/Rank256/256_5e5_010.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4          
#SBATCH --cpus-per-task=4        
#SBATCH --mem=64G                
#SBATCH --time=23:50:00          
#SBATCH --partition=capella  
#SBATCH --exclusive

module load CUDA

# source /home/hama901h/miniconda3/etc/profile.d/conda.sh
# conda activate /home/hama901h/miniconda3/envs/dolma

# source /data/horse/ws/hama901h-BFTranslation/venv-open-instruct/bin/activate

source /data/horse/ws/hama901h-BFTranslation/venv-TRL/bin/activate

cd /data/horse/ws/hama901h-BFTranslation/  
# IMPORTANT:
# This script uses Ray internally to manage multiple GPU workers and vLLM.
# Do NOT launch it with torchrun/mpirun. Doing so spawns multiple independent
# processes that each try to start a Ray cluster and a torch.distributed store,
# which leads to TCPStore timeouts like the one you saw.
#
# Run a single Python process and let Ray allocate GPUs. We request 4 GPUs
# from SLURM above: 3 are used by PPO/DeepSpeed actors and 1 by vLLM.

# Optional: avoid Triton autotune cache on NFS (harmless warning otherwise)
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/$USER/triton-cache}

# Ensure no stray torchrun env contaminates Ray/Deepspeed inside actors
unset MASTER_ADDR MASTER_PORT RANK WORLD_SIZE LOCAL_RANK LOCAL_WORLD_SIZE

python -u /data/horse/ws/hama901h-BFTranslation/open-instruct/open_instruct/ppo_vllm_thread_ray_gtrl.py \
    --dataset_mixer_list allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 1.0 \
    --dataset_mixer_list_splits train \
    --dataset_mixer_eval_list allenai/RLVR-GSM-MATH-IF-Mixed-Constraints 16 \
    --dataset_mixer_eval_list_splits train \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --response_length 2048 \
    --model_name_or_path /data/horse/ws/hama901h-BFTranslation/checkpoints/meta-llama/Llama-3.1-8B/tulu3/w_checkpoints/Rank256/alpha_5e5_010/final \
    --reward_model_path /data/horse/ws/hama901h-BFTranslation/checkpoints/meta-llama/Llama-3.1-Tulu-3-8B-RM \
    --non_stop_penalty \
    --stop_token eos \
    --temperature 1.0 \
    --chat_template_name tulu \
    --learning_rate 3e-7 \
    --total_episodes 10000000 \
    --penalty_reward_value -10.0 \
    --deepspeed_stage 3 \
    --per_device_train_batch_size 2 \
    --local_rollout_forward_batch_size 2 \
    --local_mini_batch_size 32 \
    --local_rollout_batch_size 32 \
    --actor_num_gpus_per_node 3 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.05 \
    --apply_verifiable_reward true \
    --output_dir output/rlvr_8b \
    --seed 3 \
    --num_evals 3 \
    --save_freq 100 \
    --reward_model_multiplier 0.0 \
    --gradient_checkpointing \
    --with_tracking
  # --dataset_local_cache_dir /data/horse/ws/hama901h-BFTranslation/data/alpaca \
