#!/bin/bash
#SBATCH --job-name=merge-test
#SBATCH --output=merge-test.out
#SBATCH --error=merge-test.err
#SBATCH --nodes=1        
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1          
#SBATCH --cpus-per-task=4        
#SBATCH --mem=64G                
#SBATCH --time=23:50:00          
#SBATCH --partition=capella

module load CUDA

source /home/hama901h/miniconda3/etc/profile.d/conda.sh
conda activate /home/hama901h/miniconda3/envs/dolma

cd /data/horse/ws/hama901h-BFTranslation/

# Run the Python script with the specified arguments
python scripts/extract_adapter.py --checkpoint_path "$CHECKPOINT_PATH" --adapter_save_path "$ADAPTER_SAVE_PATH" --lora_rank $LORA_RANK --lora_alpha $LORA_ALPHA --lora_dropout $LORA_DROPOUT
    
echo "Merge job completed successfully."