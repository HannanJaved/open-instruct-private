import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model
from safetensors.torch import load_file

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Extract LoRA adapter from a checkpoint.")
parser.add_argument("--checkpoint_path", required=True, help="Path to the intermediate checkpoint folder.")
parser.add_argument("--adapter_save_path", required=True, help="Path to save the extracted adapter.")
parser.add_argument("--lora_rank", type=int, default=256, help="LoRA rank used during training.")
parser.add_argument("--lora_alpha", type=int, default=512, help="LoRA alpha used during training.")
parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout used during training.")
args = parser.parse_args()

CHECKPOINT_PATH = args.checkpoint_path
ADAPTER_SAVE_PATH = args.adapter_save_path

# --- CONFIGURE YOUR PATHS HERE ---
BASE_MODEL_PATH = "/data/horse/ws/hama901h-BFTranslation/checkpoints/meta-llama/Llama-3.1-8B"  # Path to the original base model
# --------------------------------

# -- LoRA Config --
# IMPORTANT: Use the same values you used for training!
LORA_RANK = args.lora_rank  # e.g., 256 
LORA_ALPHA = args.lora_alpha  # e.g., 512
LORA_DROPOUT = args.lora_dropout  # e.g., 0.1
# ------------------
# Common target modules for Llama models
TARGET_MODULES = ["q_proj", "o_proj", "v_proj", "k_proj", "gate_proj", "up_proj", "down_proj"]

# ----------------------------------------

print(f"Loading base model from: {BASE_MODEL_PATH}")
# Load the base model with the same settings used for training
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16, # Or torch.float16, match your model
    device_map="cpu",
)

print(f"Resizing model embeddings to the known correct size: 128264 ")
base_model.resize_token_embeddings(128264)

print("Creating LoRA model structure...")
# Create the same LoRA config you used for training
peft_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=TARGET_MODULES,
    bias="none",
    task_type="CAUSAL_LM",
)

# Apply the LoRA structure to the base model
model = get_peft_model(base_model, peft_config)
print("LoRA model structure created.")

# Path to the raw weights file from the checkpoint
weights_path = os.path.join(CHECKPOINT_PATH, "model.safetensors")
if not os.path.exists(weights_path):
    # Fallback for older checkpoints that might use .bin
    weights_path = os.path.join(CHECKPOINT_PATH, "pytorch_model.bin")

print(f"Loading state dictionary from: {weights_path}")
# Load the raw state dictionary
state_dict = load_file(weights_path)

print("Applying loaded weights to the model...")
# Load the state dict into the LoRA model structure
model.load_state_dict(state_dict, strict=True)

print(f"Saving portable adapter to: {ADAPTER_SAVE_PATH}")
# Now, save the model in the correct portable adapter format
model.save_pretrained(ADAPTER_SAVE_PATH)

print("\n✅ Adapter extracted successfully!")