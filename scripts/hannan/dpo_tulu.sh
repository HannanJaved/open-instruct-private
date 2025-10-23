#!/bin/bash
#SBATCH --job-name=full_DPO_256_5e5_010
#SBATCH --output=logs/DPO/Rank256/full_256_5e5_010.out
#SBATCH --error=logs/DPO/Rank256/full_256_5e5_010.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4          
#SBATCH --cpus-per-task=4        
#SBATCH --mem=128G                
#SBATCH --time=2-00:00:00          
#SBATCH --partition=capella  
#SBATCH --exclusive

module load CUDA

source /home/hama901h/miniconda3/etc/profile.d/conda.sh
conda activate /home/hama901h/miniconda3/envs/dolma

cd /data/horse/ws/hama901h-BFTranslation/

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes 4 \
    --use_deepspeed \
    --deepspeed_config_file /data/horse/ws/hama901h-BFTranslation/open-instruct/configs/ds_configs/stage3_no_offloading_accelerate.conf \
    /data/horse/ws/hama901h-BFTranslation/open-instruct/open_instruct/dpo_tune_cache.py \
    --exp_name tulu3_8b_dpo \
    --dataset_mixer_list allenai/llama-3.1-tulu-3-8b-preference-mixture 1.0 \
    --dataset_mixer_list_splits train \
    --max_seq_length 1024 \
    --model_name_or_path /data/horse/ws/hama901h-BFTranslation/checkpoints/meta-llama/Llama-3.1-8B/tulu3/w_checkpoints/Rank256/alpha_5e5_010/final \
    --tokenizer_name /data/horse/ws/hama901h-BFTranslation/checkpoints/meta-llama/Llama-3.1-8B/tulu3/w_checkpoints/Rank256/alpha_5e5_010/final \
    --use_slow_tokenizer \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-7 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --weight_decay 0.0 \
    --num_train_epochs 1 \
    --dpo_loss_type dpo_norm \
    --dpo_beta 5 \
    --use_flash_attn False \
    --gradient_checkpointing True \
    --report_to wandb \
    --with_tracking \
    --logging_steps 5 \
    --output_dir /data/horse/ws/hama901h-BFTranslation/checkpoints/meta-llama/Llama-3.1-8B/tulu3/w_checkpoints/Rank256/alpha_5e5_010/DPO/full_final \
    --seed 8 \
    --run_name full_DPO_256_5e5_010 \
    # --use_lora True \
    # --lora_rank 256 \
    # --lora_alpha 64