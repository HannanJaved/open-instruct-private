#!/bin/bash
#SBATCH --job-name=Pythia410
#SBATCH --output=logs/eval/Tulu3/Pythia410_Alpaca_3epochs.out
#SBATCH --error=logs/eval/Tulu3/Pythia410_Alpaca_3epochs.err
#SBATCH --nodes=1                        
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1           
#SBATCH --cpus-per-task=4        
#SBATCH --mem=64G                
#SBATCH --time=24:00:00          
#SBATCH --partition=capella

module load CUDA

source /home/hama901h/miniconda3/etc/profile.d/conda.sh
conda activate /home/hama901h/miniconda3/envs/dolma

cd /data/horse/ws/hama901h-BFTranslation/

# Run the finetuning script with torchrun for distributed training.
python synthetic-data-gen-litgpt/litgpt/eval/evaluate_forward.py

echo "Evaluated."