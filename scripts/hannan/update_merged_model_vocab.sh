#!/bin/bash

# This script updates the vocab_size in the config.json for all merged model checkpoints.
# The error log indicates the correct vocab size should be 128264.

BASE_DIR="/data/horse/ws/hama901h-BFTranslation/checkpoints/meta-llama/Llama-3.1-8B/tulu3/w_checkpoints/Rank256"
find "$BASE_DIR" -name "config.json" | while read -r CONFIG_FILE; do
    echo "Updating vocab_size in $CONFIG_FILE to 128264"
    jq '.vocab_size = 128264' "$CONFIG_FILE" > "${CONFIG_FILE}.tmp" && mv "${CONFIG_FILE}.tmp" "$CONFIG_FILE"
done
echo "✅ All config.json files updated successfully."
