import os
import json

BASE_DIR = "/data/horse/ws/hama901h-BFTranslation/checkpoints/meta-llama/Llama-3.1-8B/tulu3/w_checkpoints/Rank256"

for root, _, files in os.walk(BASE_DIR):
    for file in files:
        if file == "config.json":
            config_path = os.path.join(root, file)
            print(f"Updating vocab_size in {config_path}")
            with open(config_path, "r") as f:
                config = json.load(f)
            config["vocab_size"] = 128264
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
print("✅ All config.json files updated successfully.")
