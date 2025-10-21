#!/bin/bash
#SBATCH --job-name=8B-SmolLM2-360M
#SBATCH --output=logs/SmolLM2-360M/no_warmup_full_6epochs.out
#SBATCH --error=logs/SmolLM2-360M/no_warmup_full_6epochs.err
#SBATCH --nodes=1        
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4          
#SBATCH --cpus-per-task=4        
#SBATCH --mem=64G                
#SBATCH --time=23:50:00          
#SBATCH --partition=capella  
#SBATCH --gpu-bind=none 

source /home/hama901h/miniconda3/etc/profile.d/conda.sh
conda activate /home/hama901h/miniconda3/envs/py39

cd /data/horse/ws/hama901h-BFTranslation/  

# Get the node names and master node information
# export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
# export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# echo "MASTER_ADDR:MASTER_PORT="${MASTER_ADDR}:${MASTER_PORT}
# export WORLD_SIZE=$SLURM_NTASKS
# # export RANK=$SLURM_PROCID

# # Set longer timeouts for torch.distributed
# export NCCL_TIMEOUT=3600
# # export NCCL_DEBUG=INFO
# # export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export TORCHELASTIC_MAX_INITIALIZATION_SECONDS=3600
# export NCCL_SOCKET_IFNAME=^lo,docker

# # Print debugging information
# echo "MASTER_ADDR: $MASTER_ADDR"
# echo "WORLD_SIZE: $WORLD_SIZE"
# # echo "RANK: $RANK"
# echo "SLURM_PROCID: $SLURM_PROCID"
# echo "SLURM_NODEID: $SLURM_NODEID"

# # Modified torchrun command with proper multi-node configuration
# srun torchrun --nproc_per_node=4 \
#          --nnodes=2 \
#          --node_rank=$SLURM_PROCID \
#          --master_addr=$MASTER_ADDR \
#          --master_port=$MASTER_PORT \
#          --rdzv_backend=c10d \
#          --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
#          --rdzv_id=$SLURM_JOBID \
#          synthetic-data-gen-litgpt/litgpt/finetune/finetune_BFModel.py \
#          --model_type forward

torchrun --nproc_per_node=4 --nnodes=1 synthetic-data-gen-litgpt/litgpt/finetune/finetune_BFModel.py \
    --model_type forward

echo "Finetuning complete."