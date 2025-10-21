#!/bin/bash
#SBATCH --job-name=1024-1e5-001
#SBATCH --output=logs/Tulu/Finetuning/Rank4096/Llama3.1-8B-LORA-Finetune-1e5-001.out
#SBATCH --error=logs/Tulu/Finetuning/Rank4096/Llama3.1-8B-LORA-Finetune-1e5-001.err
#SBATCH --nodes=4        
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4          
#SBATCH --cpus-per-task=4        
#SBATCH --mem=256G                
#SBATCH --time=23:50:00          
#SBATCH --partition=capella  

module load CUDA

source /home/hama901h/miniconda3/etc/profile.d/conda.sh
conda activate /home/hama901h/miniconda3/envs/dolma

cd /data/horse/ws/hama901h-BFTranslation/  

# Get the master node address
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=29500

# Set environment variables for distributed training.
# torchrun will handle setting RANK and WORLD_SIZE
# export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE * 4)) # 2 * 1 * 4 = 8
# export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# NCCL settings for better debugging and stability
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^docker0,lo

# Increase timeouts
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# torchrun --nproc_per_node=4 --nnodes=2 open-instruct/open_instruct/finetune.py \
#   --model_name_or_path /data/horse/ws/hama901h-BFTranslation/checkpoints/meta-llama/Llama-3.1-8B \
#   --output_dir /data/horse/ws/hama901h-BFTranslation/checkpoints/meta-llama/Llama-3.1-8B/tulu3/ \
#   --num_train_epochs 1 \
#   --per_device_train_batch_size 8 \
#   --learning_rate 1e-6 \
#   --max_seq_length 512 \
#   --gradient_accumulation_steps 4 \
#   --use_lora True \
#   --use_qlora False \
#   --warmup_ratio 0.05

srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=$SLURM_GPUS_ON_NODE \
  --node_rank=$SLURM_NODEID \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  open-instruct/open_instruct/finetune.py \
  --model_name_or_path /data/horse/ws/hama901h-BFTranslation/checkpoints/meta-llama/Llama-3.1-8B \
  --output_dir /data/horse/ws/hama901h-BFTranslation/checkpoints/meta-llama/Llama-3.1-8B/tulu3/Rank1024/ \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --max_seq_length 512 \
  --gradient_accumulation_steps 1 \
  --use_lora True \
  --use_qlora False \
  --lora_rank 1024 \
  --lora_alpha 256 \
  --learning_rate 1e-5 \
  --warmup_ratio 0.01