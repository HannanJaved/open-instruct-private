# filepath: /data/horse/ws/hama901h-BFTranslation/scripts/update_vocab_size.sh
#!/bin/bash

BASE_DIR="/data/horse/ws/hama901h-BFTranslation/checkpoints/meta-llama/Llama-3.1-8B/tulu3/w_checkpoints/Rank256"
find "$BASE_DIR" -name "config.json" | while read -r CONFIG_FILE; do
    echo "Updating vocab_size in $CONFIG_FILE"
    jq '.vocab_size = 128256' "$CONFIG_FILE" > "${CONFIG_FILE}.tmp" && mv "${CONFIG_FILE}.tmp" "$CONFIG_FILE"
done
echo "✅ All config.json files updated successfully."