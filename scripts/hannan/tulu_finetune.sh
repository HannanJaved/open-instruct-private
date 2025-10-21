#!/bin/bash
#SBATCH --job-name=256-alpha_128_5e5_010
#SBATCH --output=logs/Rank256/256-alpha_128_5e5_010.out
#SBATCH --error=logs/Rank256/256-alpha_128_5e5_010.err
#SBATCH --nodes=1        
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4          
#SBATCH --cpus-per-task=4        
#SBATCH --mem=128G                
#SBATCH --time=23:50:00          
#SBATCH --partition=capella  
#SBATCH --exclusive

module load CUDA

source /home/hama901h/miniconda3/etc/profile.d/conda.sh
conda activate /home/hama901h/miniconda3/envs/dolma

cd /data/horse/ws/hama901h-BFTranslation/  

torchrun --nproc_per_node=4 --nnodes=1 open-instruct/open_instruct/finetune.py \
  --model_name_or_path /data/horse/ws/hama901h-BFTranslation/checkpoints/meta-llama/Llama-3.1-8B \
  --output_dir /data/horse/ws/hama901h-BFTranslation/checkpoints/meta-llama/Llama-3.1-8B/tulu3/w_checkpoints/Rank256/alpha_5e5_010/alpha_128 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --max_seq_length 512 \
  --gradient_accumulation_steps 4 \
  --use_lora True \
  --use_qlora False \
  --lora_rank 256 \
  --lora_alpha 128 \
  --learning_rate 5e-5 \
  --warmup_ratio 0.1
  # --dataset_local_cache_dir /data/horse/ws/hama901h-BFTranslation/data/alpaca \

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

