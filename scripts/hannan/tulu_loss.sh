#!/bin/bash
#SBATCH --job-name=Loss_Rank64/alpha_1e5_001/final
#SBATCH --output=logs/losses/Rank64/alpha_1e5_001/final.out
#SBATCH --error=logs/losses/Rank64/alpha_1e5_001/final.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4          
#SBATCH --cpus-per-task=4        
#SBATCH --mem=64G                
#SBATCH --time=23:50:00          
#SBATCH --partition=capella  
#SBATCH --exclusive

module load CUDA

source /home/hama901h/miniconda3/etc/profile.d/conda.sh
conda activate /home/hama901h/miniconda3/envs/dolma

cd /data/horse/ws/hama901h-BFTranslation/  

srun python open-instruct/open_instruct/evaluate.py \
    --model_name_or_path /data/horse/ws/hama901h-BFTranslation/checkpoints/meta-llama/Llama-3.1-8B/tulu3/w_checkpoints/Rank64/alpha_1e5_001/final \
    --dataset_mixer_list allenai/tulu-3-sft-mixture 1.0 \
    --dataset_mixer_list_splits train \
    --per_device_eval_batch_size 8 \
    --max_seq_length 2048 \
    --output_dir eval_output/ \
    --reduce_loss mean \
    --use_flash_attn False

