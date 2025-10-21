import os
import shutil

# Define source and destination directories
source_dir = "/data/horse/ws/hama901h-BFTranslation/checkpoints/meta-llama/Llama-3.1-8B/tulu3/w_checkpoints/Rank64"
destination_dir = "/data/horse/ws/hama901h-BFTranslation/Tulu_adapters/Rank64"

# source_dir = "/data/horse/ws/hama901h-BFTranslation/Tulu_adapters/Rank64"
# destination_dir = "/data/horse/ws/hama901h-BFTranslation/checkpoints/meta-llama/Llama-3.1-8B/tulu3/w_checkpoints/Rank64"
# Iterate through all directories and files in the source directory
for root, _, files in os.walk(source_dir):
    for file in files:
        if file in {"adapter_config.json", "adapter_model.safetensors"}:
            # Construct full file paths
            source_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, source_dir)
            destination_path = os.path.join(destination_dir, relative_path)

            # Ensure the destination directory exists
            os.makedirs(destination_path, exist_ok=True)

            # Move the file
            shutil.move(source_file_path, os.path.join(destination_path, file))
            print(f"Moved: {source_file_path} -> {os.path.join(destination_path, file)}")
