#!/bin/bash
# Multi-node distributed training configuration with DeepSpeed ZeRO-3:
# - 4 nodes x 4 GPUs = 16 total GPUs
# - Per-device batch size: 2
# - Gradient accumulation steps: 4
# - Effective batch size: 2 x 16 x 4 = 128 (exact target)
#
# KEY MEMORY OPTIMIZATIONS:
# 1. DeepSpeed ZeRO-3: Shards model parameters, gradients, and optimizer states
# 2. Gradient Checkpointing: Trades compute for memory (reduces activation memory)
# 3. Flash Attention 2: More memory-efficient attention computation
# 4. BF16 mixed precision: Reduces memory footprint
#
#SBATCH --job-name=full_tulu_SFT
#SBATCH --output=/data/horse/ws/hama901h-BFTranslation/logs/Tulu/Finetuning/SFT_DPO/%x.out
#SBATCH --error=/data/horse/ws/hama901h-BFTranslation/logs/Tulu/Finetuning/SFT_DPO/%x.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=2-00:00:00
#SBATCH --partition=capella
#SBATCH --exclusive

module load CUDA

source /home/hama901h/miniconda3/etc/profile.d/conda.sh
conda activate /home/hama901h/miniconda3/envs/dolma

cd /data/horse/ws/hama901h-BFTranslation/

# Get master node hostname for distributed training
export NCCL_SOCKET_IFNAME='ibp3s0.8002,ibp35s0.8002,ibp163s0.8002,ibp195s0.8002'
# try limited membership instead of full
export NCCL_IB_PKEY=0x2

export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2
export NCCL_MIN_CHANNELS=32
export NCCL_DEBUG=INFO
export NCCL_IB_RETRY_CNT=10
export NCCL_MIN_NCHANNELS=11
export NCCL_TREE_THRESHOLD=4294967296
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_DISTRIBUTED_TIMEOUT=300
export TORCHELASTIC_MAX_FAILED_CONNECTIONS=60
export TORCH_DISTRIBUTED_HEARTBEAT_TIMEOUT=300
export TORCH_DISTRIBUTED_COODINATOR_TIMEOUT=300
export OMP_NUM_THREADS=18

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}

export RDZV_HOST=$head_node
export RDZV_PORT=29400

echo "head_node=$head_node"

NPROC_PER_NODE=$(nvidia-smi -L | wc -l)

echo NPROC_PER_NODE=$NPROC_PER_NODE

# Launch with accelerate using DeepSpeed ZeRO-3
srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint="$RDZV_HOST:$RDZV_PORT" \
    open-instruct/open_instruct/finetune.py \
      --model_name_or_path /data/horse/ws/hama901h-BFTranslation/checkpoints/meta-llama/Llama-3.1-8B \
      --output_dir /data/horse/ws/hama901h-BFTranslation/checkpoints/meta-llama/Llama-3.1-8B/tulu3/w_checkpoints/full_run/SFT/ \
      --num_train_epochs 2 \
      --per_device_train_batch_size 1 \
      --max_seq_length 4096 \
      --gradient_accumulation_steps 8 \
      --use_lora False \
      --use_qlora False \
      --learning_rate 5e-6 \
      --lr_scheduler_type linear \
      --warmup_ratio 0.03 \
      --weight_decay 0.0 \
      --gradient_checkpointing True \
      --use_flash_attn True \
      --use_slow_tokenizer \
      --reduce_loss sum \
      --report_to wandb \
      --with_tracking \
      --wandb_project_name instruction-tuning \
      --wandb_entity openeurollm-project \
      --logging_steps 1 \
      --validation_split_percentage 1 \
      --validation_steps 10 \
      --dataset_skip_cache False \
      --do_not_randomize_output_dir True \
      --add_seed_and_date_to_exp_name False \
      --exp_name "Llama-3.1-8B_Tulu3_Full_SFT_Run1"
# "
 

# srun bash -c "
#   accelerate launch \
#     --mixed_precision bf16 \
#     --num_machines 4 \
#     --num_processes 16 \
#     --machine_rank \$SLURM_NODEID \
#     --main_process_ip $MASTER_ADDR \
#     --main_process_port $MASTER_PORT \
#     --use_deepspeed \
#     --deepspeed_config_file open-instruct/configs/ds_configs/stage3_no_offloading_accelerate.conf \
#     --deepspeed_multinode_launcher standard \

# --lora_rank 64 \
  # --lora_alpha 64 \
  # --dataset_local_cache_dir /data/horse/ws/hama901h-BFTranslation/data/alpaca

# python open-instruct/open_instruct/finetune.py \
#   --model_name_or_path /path/to/your/model/checkpoints \
#   --dataset_local_cache_dir /path/to/your/dataset \
#   --output_dir /path/to/output/directory \
#   --num_train_epochs 3 \
#   --per_device_train_batch_size 8 \
#   --learning_rate 1e-5 \
#   --max_seq_length 512 \
#   --gradient_accumulation_steps 4 \
#   --use_lora False \
#   --use_qlora False

