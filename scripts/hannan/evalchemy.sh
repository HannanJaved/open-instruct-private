#!/bin/bash
#SBATCH --job-name=open-instruct-venv
#SBATCH --error=/data/horse/ws/hama901h-BFTranslation/logs/open-instruct-venv.err
#SBATCH --output=/data/horse/ws/hama901h-BFTranslation/logs/open-instruct-venv.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4        
#SBATCH --mem=32G                
#SBATCH --time=02:30:00          
#SBATCH --partition=capella
#SBATCH --gres=gpu:1
module load CUDA

echo "Starting job"

source /data/horse/ws/hama901h-BFTranslation/venv-open-instruct/bin/activate
echo "Virtual environment activated"

cd /data/horse/ws/hama901h-BFTranslation/open-instruct
echo "Changed directory to open-instruct"

pip install --upgrade pip "setuptools<70.0.0" wheel 
echo "Upgraded pip, setuptools, and wheel"

pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
echo "Installed torch and torchvision"

pip install packaging
echo "Installed packaging"

pip install flash-attn==2.7.2.post1 --no-build-isolation
echo "Installed flash-attn"

pip install -r requirements.txt
echo "Installed requirements from requirements.txt"

pip install -e .
echo "Installed the package in editable mode"

python -m nltk.downloader punkt
echo "Downloaded NLTK punkt tokenizer"

echo "Setup complete"
# # evalchemy run -c /data/horse/ws/hama901h-BFTranslation/config.yaml
# pip install -e .
# evalchemy run -c /data/horse/ws/hama901h-BFTranslation/config.yaml