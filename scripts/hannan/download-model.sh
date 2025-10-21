#!/bin/bash
#SBATCH --job-name=download-model
#SBATCH --error=download-model.err
#SBATCH --output=download-model.out
#SBATCH --nodes=1        
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4        
#SBATCH --mem=32G                
#SBATCH --time=01:00:00          
#SBATCH --partition=capella
#SBATCH --gres=gpu:1  

# source /data/horse/ws/hama901h-BFTranslation/venv-evalchemy/bin/activate

# cd /data/horse/ws/hama901h-BFTranslation/checkpoints/meta-llama

git lfs install
cd /data/horse/ws/hama901h-BFTranslation/checkpoints/meta-llama/
git clone https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B-RM

# cd /data/horse/ws/hama901h-BFTranslation/evalchemy

# pip install -e .
# pip install -e eval/chat_benchmarks/alpaca_eval