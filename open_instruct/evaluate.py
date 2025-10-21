#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# isort: off
import os

os.environ["NCCL_CUMEM_ENABLE"] = "0"  # NOQA
try:
    import deepspeed
except Exception:
    pass
# isort: on
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import timedelta
from typing import List, Literal, Optional, Union

import datasets
import torch
import transformers
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.logging import get_logger
from accelerate.utils import InitProcessGroupKwargs, set_seed
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, DataCollatorForSeq2Seq
from transformers.training_args import _convert_str_dict

from open_instruct.dataset_transformation import (
    INPUT_IDS_KEY,
    TOKENIZED_SFT_DATASET_KEYS,
    TokenizerConfig,
    get_cached_dataset_tulu,
    visualize_token,
)

from open_instruct.utils import (
    ArgumentParserPlus,
    is_beaker_job,
)

logger = get_logger(__name__)


@dataclass
class EvalArguments:
    """
    Arguments for evaluation jobs.
    """

    _VALID_DICT_FIELDS = ["additional_model_arguments"]

    model_name_or_path: str = field(
        metadata={"help": "The model checkpoint for evaluation."}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    use_flash_attn: bool = field(
        default=True, metadata={"help": "Whether to use flash attention in the model"}
    )
    model_revision: Optional[str] = field(
        default=None,
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    additional_model_arguments: Optional[Union[dict, str]] = field(
        default_factory=dict, metadata={"help": "A dictionary of additional model args used to construct the model."}
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, "
                "then only materialize its parameters when the pretrained weights are loaded."
            )
        },
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "If True, will load model as LORA model."},
    )
    lora_adapter_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to LORA adapter if use_lora is True."},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_mixer: Optional[dict] = field(
        default=None, metadata={"help": "A dictionary of datasets (local or HF) to sample from."}
    )
    dataset_mixer_list: List[str] = field(default_factory=lambda: ["allenai/tulu-3-sft-mixture", "1.0"])
    """A list of datasets (local or HF) to sample from."""
    dataset_mixer_list_splits: List[str] = field(default_factory=lambda: ["train"])
    """The dataset splits to use for evaluation"""
    dataset_transform_fn: list[str] = field(
        default_factory=lambda: ["sft_tulu_tokenize_and_truncate_v1", "sft_tulu_filter_v1"]
    )
    """The list of transform functions to apply to the dataset."""
    dataset_target_columns: List[str] = field(default_factory=lambda: TOKENIZED_SFT_DATASET_KEYS)
    """The columns to use for the dataset."""
    dataset_cache_mode: Literal["hf", "local"] = "local"
    """The mode to use for caching the dataset."""
    dataset_local_cache_dir: str = "local_dataset_cache"
    """The directory to save the local dataset cache to."""
    dataset_config_hash: Optional[str] = None
    """The hash of the dataset configuration."""
    dataset_skip_cache: bool = False
    """Whether to skip the cache."""
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker evaluation, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None, metadata={"help": "The number of processes to use for the preprocessing."}
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. "
                "Sequences longer than this will be truncated,"
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached evaluation sets"}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    timeout: int = field(
        default=1800,
        metadata={
            "help": "Timeout for the evaluation process in seconds. "
            "Useful if tokenization process is long. Default is 1800 seconds (30 minutes)."
        },
    )
    reduce_loss: str = field(
        default="mean",
        metadata={
            "help": "How to reduce loss over tokens. Options are 'mean' or 'sum'."
        },
    )
    seed: int = field(default=42, metadata={"help": "Random seed for initialization and dataset shuffling."})
    output_dir: str = field(
        default="eval_output/",
        metadata={"help": "The output directory where the evaluation results will be written."},
    )
    load_balancing_loss: bool = field(
        default=False, metadata={"help": "Whether to include a load balancing loss (for OLMoE) or not."}
    )
    load_balancing_weight: float = field(
        default=0.5, metadata={"help": "Weight for load balancing loss if applicable."}
    )

    def __post_init__(self):
        if self.reduce_loss not in ["mean", "sum"]:
            raise ValueError("reduce_loss must be either 'mean' or 'sum'")
        if self.dataset_name is None and self.dataset_mixer is None and self.dataset_mixer_list is None:
            raise ValueError("Need either a dataset name, dataset mixer, or dataset mixer list.")
        if (
            (self.dataset_name is not None and (self.dataset_mixer is not None or self.dataset_mixer_list is not None))
            or (self.dataset_name is not None)
            or (self.dataset_mixer is not None and self.dataset_mixer_list is not None)
        ):
            raise ValueError("Cannot provide two dataset selection mechanisms.")
        if self.use_lora and self.lora_adapter_path is None:
            raise ValueError("If use_lora is True, lora_adapter_path must be provided.")

        # Parse in args that could be `dict` sent in from the CLI as a string
        for dict_field in self._VALID_DICT_FIELDS:
            passed_value = getattr(self, dict_field)
            if isinstance(passed_value, str) and passed_value.startswith("{"):
                loaded_dict = json.loads(passed_value)
                loaded_dict = _convert_str_dict(loaded_dict)
                setattr(self, dict_field, loaded_dict)


def evaluate_loss(model, dataloader, accelerator, embedding_size, args):
    """Evaluate model on the given dataloader and return average loss."""
    model.eval()
    total_loss = 0.0
    total_aux_loss = 0.0
    total_tokens = 0
    
    progress_bar = tqdm(dataloader, desc="Evaluating", disable=not accelerator.is_local_main_process)
    
    with torch.no_grad():
        for batch in progress_bar:
            # Count tokens
            if "attention_mask" in batch:
                tokens = batch["attention_mask"].sum().item()
            elif "position_ids" in batch:
                tokens = batch["position_ids"].numel()
            elif "cu_seq_lens_q" in batch:
                tokens = batch["cu_seq_lens_q"][-1].item()
            else:
                raise ValueError(f"Expected attention_mask or position_ids or cu_seq_lens_q in batch, found {batch.keys()}")
            
            # Forward pass
            if args.load_balancing_loss:
                outputs = model(**batch, use_cache=False, output_router_logits=True)
            else:
                outputs = model(**batch, use_cache=False)
            
            if args.reduce_loss == "mean":
                loss = outputs.loss
                total_loss += loss.item() * tokens
            else:
                # reduce loss is sum
                logits = outputs.logits
                labels = batch["labels"]
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                del logits
                
                loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
                shift_logits = shift_logits.view(-1, embedding_size)
                shift_labels = shift_labels.view(-1).to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
                del shift_logits
                total_loss += loss.item()
                tokens = shift_labels.ne(-100).sum().item()
            
            if args.load_balancing_loss:
                aux_loss = args.load_balancing_weight * outputs.aux_loss
                total_aux_loss += aux_loss.item()
            
            total_tokens += tokens
            del outputs
            
            # Update progress bar
            current_avg_loss = total_loss / max(total_tokens, 1)
            progress_bar.set_postfix({"loss": f"{current_avg_loss:.4f}", "tokens": total_tokens})
    
    avg_loss = total_loss / max(total_tokens, 1)
    avg_aux_loss = total_aux_loss / max(len(dataloader), 1) if args.load_balancing_loss else 0.0
    
    return avg_loss, avg_aux_loss, total_tokens


def main(args: EvalArguments, tc: TokenizerConfig):
    # Initialize the accelerator
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=args.timeout))
    dataloader_config = DataLoaderConfiguration(use_seedable_sampler=True)

    accelerator = Accelerator(
        dataloader_config=dataloader_config,
        kwargs_handlers=[timeout_kwargs],
    )

    # Setup tokenizer
    tc.tokenizer_revision = args.model_revision if tc.tokenizer_revision is None else tc.tokenizer_revision
    tc.tokenizer_name_or_path = (
        args.model_name_or_path if tc.tokenizer_name_or_path is None else tc.tokenizer_name_or_path
    )
    if tc.tokenizer_revision != args.model_revision and tc.tokenizer_name_or_path != args.model_name_or_path:
        warning = f"""Requested tokenizer revision `{tc.tokenizer_revision=}` is different
                   from the model revision `{args.model_revision=}` or the tokenizer name `{tc.tokenizer_name_or_path=}`
                   is different from the model name `{args.model_name_or_path=}`."""
        logger.warning(warning)
    tokenizer = tc.tokenizer

    # Set up runtime variables
    args.dataset_local_cache_dir = os.path.abspath(args.dataset_local_cache_dir)
    if is_beaker_job():
        args.dataset_local_cache_dir = "/weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache"

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Set seed
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    # Load dataset
    if args.dataset_mixer is not None:
        args.dataset_mixer_list = [item for pair in args.dataset_mixer.items() for item in pair]
    
    with accelerator.main_process_first():
        transform_fn_args = [{"max_seq_length": args.max_seq_length}, {}]
        eval_dataset = get_cached_dataset_tulu(
            dataset_mixer_list=args.dataset_mixer_list,
            dataset_mixer_list_splits=args.dataset_mixer_list_splits,
            tc=tc,
            dataset_transform_fn=args.dataset_transform_fn,
            transform_fn_args=transform_fn_args,
            target_columns=args.dataset_target_columns,
            dataset_cache_mode=args.dataset_cache_mode,
            dataset_config_hash=args.dataset_config_hash,
            hf_entity=None,
            dataset_local_cache_dir=args.dataset_local_cache_dir,
            dataset_skip_cache=args.dataset_skip_cache,
        )

        if args.max_eval_samples is not None and args.max_eval_samples < len(eval_dataset):
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        else:
            # Use only 5% of the dataset for validation
            print("Using 5% of the dataset for evaluation")
            num_samples = int(len(eval_dataset) * 0.05)
            eval_dataset = eval_dataset.select(range(num_samples))

        eval_dataset.set_format(type="pt")

    if accelerator.is_main_process:
        logger.info(f"Loaded evaluation dataset with {len(eval_dataset)} examples")
        visualize_token(eval_dataset[0][INPUT_IDS_KEY], tokenizer)

    # Load model config
    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name,
            revision=args.model_revision,
            trust_remote_code=tc.trust_remote_code,
            **args.additional_model_arguments,
        )
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            revision=args.model_revision,
            trust_remote_code=tc.trust_remote_code,
            **args.additional_model_arguments,
        )
    else:
        raise ValueError("You must provide either config_name or model_name_or_path.")

    # Load model
    logger.info(f"Loading model from {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        revision=args.model_revision,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        trust_remote_code=tc.trust_remote_code,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if args.use_flash_attn else "eager",
    )

    # Load LORA adapter if needed
    if args.use_lora:
        logger.info(f"Loading LORA adapter from {args.lora_adapter_path}")
        model = PeftModel.from_pretrained(model, args.lora_adapter_path)
        model = model.merge_and_unload()

    # Get embedding size
    embeddings = model.get_input_embeddings()
    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        embedding_size = embeddings.weight.shape[0]

    # DataLoader creation
    collate_fn = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest")

    logger.info("Creating dataloader")
    eval_dataloader = DataLoader(
        eval_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.per_device_eval_batch_size
    )

    # Prepare everything with accelerator
    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

    # Evaluate!
    logger.info("***** Running evaluation *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")
    logger.info(f"  Batch size per device = {args.per_device_eval_batch_size}")
    logger.info(f"  Total batch size = {args.per_device_eval_batch_size * accelerator.num_processes}")

    avg_loss, avg_aux_loss, total_tokens = evaluate_loss(
        model, eval_dataloader, accelerator, embedding_size, args
    )

    # Gather results from all processes
    avg_loss = accelerator.gather(torch.tensor(avg_loss, device=accelerator.device)).mean().item()
    total_tokens = accelerator.gather(torch.tensor(total_tokens, device=accelerator.device)).sum().item()
    
    if args.load_balancing_loss:
        avg_aux_loss = accelerator.gather(torch.tensor(avg_aux_loss, device=accelerator.device)).mean().item()

    # Print and save results
    if accelerator.is_main_process:
        logger.info("***** Evaluation Results *****")
        logger.info(f"  Average Loss: {avg_loss:.4f}")
        logger.info(f"  Total Tokens: {total_tokens}")
        if args.load_balancing_loss:
            logger.info(f"  Average Auxiliary Loss: {avg_aux_loss:.4f}")
        
        # Save results to file
        results = {
            "model_name_or_path": args.model_name_or_path,
            "dataset": args.dataset_mixer_list,
            "average_loss": avg_loss,
            "total_tokens": total_tokens,
            "num_examples": len(eval_dataset),
        }
        if args.load_balancing_loss:
            results["average_aux_loss"] = avg_aux_loss
        
        output_file = os.path.join(args.output_dir, "eval_results.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    parser = ArgumentParserPlus((EvalArguments, TokenizerConfig))
    args, tc = parser.parse_args_into_dataclasses()
    main(args, tc)
