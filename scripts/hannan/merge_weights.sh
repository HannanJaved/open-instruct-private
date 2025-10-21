#!/bin/bash
#SBATCH --job-name=1
#SBATCH --output=logs/1.out
#SBATCH --error=logs/1.err
#SBATCH --nodes=1        
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1          
#SBATCH --cpus-per-task=4        
#SBATCH --mem=32G                
#SBATCH --time=08:00:00          
#SBATCH --partition=capella  

source /home/hama901h/miniconda3/etc/profile.d/conda.sh
conda activate /home/hama901h/miniconda3/envs/py39

cd /data/horse/ws/hama901h-BFTranslation/ 

litgpt merge_lora "checkpoints/meta-llama/Meta-Llama-3.1-8B-Instruct/out/finetune/lora/hotpotqa/forward/final"