import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import argparse

def main(args):
    """
    Merges a PEFT adapter into a base model and saves the merged model.
    """
    print(f"Loading base model from: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
    )

    print(f"Loading tokenizer from: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"Resizing model embeddings to the known correct size: {args.vocab_size}")
    base_model.resize_token_embeddings(args.vocab_size)

    print(f"Loading and applying PEFT adapter from: {args.adapter}")
    model = PeftModel.from_pretrained(base_model, args.adapter, device_map="cpu")

    print("Merging the adapter into the base model...")
    model = model.merge_and_unload()

    print(f"Saving the fully merged model to: {args.output}")
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    print("\n✅ Done! Your merged model is ready for evaluation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into a base model.")
    parser.add_argument("--base-model", type=str, required=True, help="Path to the original base model directory.")
    parser.add_argument("--adapter", type=str, required=True, help="Path to the LoRA adapter directory.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the new, fully merged model.")
    parser.add_argument("--vocab-size", type=int, required=True, help="The final, known vocabulary size of the model.")
    
    args = parser.parse_args()
    main(args)