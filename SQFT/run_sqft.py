#!/usr/bin/env python
# coding=utf-8

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import datasets
import torch
import transformers
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed
)

from utils.load_dataset import load_cs_dataset, load_gsm8k_dataset

logger = logging.getLogger(__name__)


@dataclass
class SQFTTrainingArguments(TrainingArguments):
    lora_r: int = field(default=32, metadata={"help": "Lora R dimension."})
    lora_alpha: float = field(default=64, metadata={"help": "Lora alpha."})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout."})
    target_modules: List[str] = field(
        default=None, metadata={"help": "Which module will be added the lora adapter."}
    )
    nls: bool = field(default=False, metadata={"help": "Whether to apply Neural LoRA Search (NLS) or not."})
    nls_target_modules: List[str] = field(
        default=None, metadata={"help": "Which module will be added the elastic lora adapter."}
    )
    search_space: List[int] = field(default=None, metadata={"help": "Low-rank search space of NLS training."})
    share_rank_within_layer: bool = field(default=True, metadata={"help": "Whether the adapter shares rank values within the layer."})
    sparse_adapter: bool = field(default=False, metadata={"help": "Enable SparsePEFT."})
    quantization_aware: bool = field(default=False, metadata={"help": "Enable quantization-aware SparsePEFT."})


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to the data used for training and evaluation.
    """
    dataset_name: Optional[str] = field(default=None, metadata={"help": "The dataset to use."})
    val_set_size: int = field(default=0, metadata={"help": "Validation set size."})
    cutoff_len: int = field(default=256, metadata={"help": "Cutoff length for tokenization."})


@dataclass
class ModelArguments:
    """
    Base model/tokenizer that will be fine-tuned.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to the pre-trained model or model identifier from huggingface.co/models"}
    )
    dtype: str = field(default="float16", metadata={"help": "Data type for the model."})
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    non_quant_model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, SQFTTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=model_args.dtype,
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    logger.info("Adding LoRA modules...")
    lora_config = LoraConfig(
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        lora_dropout=training_args.lora_dropout,
        target_modules=training_args.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
        sparse_adapter=training_args.sparse_adapter,
        quantization_aware=training_args.quantization_aware,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    # Init SQFTQuantAwareLinear params
    if training_args.quantization_aware:
        from modules.sqft_qa_linear import SQFTQuantAwareLinear

        def find_layers(module, layers=[SQFTQuantAwareLinear], name=''):
            """
            Recursively find layers of a specific type in a module.
            """
            if type(module) in layers:
                return {name: module}
            res = {}
            for name1, child in module.named_children():
                res.update(find_layers(
                    child, layers=layers, name=name + '.' + name1 if name != '' else name1
                ))
            return res

        non_quant_model = None
        non_quant_model_modules = None
        if model_args.non_quant_model_name_or_path is not None:
            non_quant_model = AutoModelForCausalLM.from_pretrained(
                model_args.non_quant_model_name_or_path,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=model_args.dtype,
            )
            non_quant_model_modules = {name: module for name, module in non_quant_model.model.layers.named_modules()}

        # Initialize the quantization layer using full/half-precision weight parameters
        sqft_quant_aware_layers = find_layers(model.base_model.model.model.layers)
        for layer_name in sqft_quant_aware_layers:
            sqft_quant_aware_layer = sqft_quant_aware_layers[layer_name]
            custom_weight = None
            if non_quant_model_modules is not None:
                custom_weight = non_quant_model_modules[layer_name].weight.data
            sqft_quant_aware_layer.init_params(custom_weight)
        del non_quant_model, non_quant_model_modules

    if training_args.nls:
        from modules.elastic_lora_linear import make_lora_elastic
        make_lora_elastic(
            model, 
            training_args.search_space, 
            target_modules=training_args.nls_target_modules, 
            share_rank_within_layer=training_args.share_rank_within_layer,
            config_save_dir=training_args.output_dir, 
        )
        logger.info(f"Training/evaluation parameters {training_args}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name is not None else model_args.model_name_or_path,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load data
    def tokenize(prompt, add_eos_token=True):
        """
        Tokenize the given prompt.
        """
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=data_args.cutoff_len,
            padding=True,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < data_args.cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        """
        Generate and tokenize the prompt from the data point.
        """
        full_prompt = data_point["full_prompt"]
        tokenized_full_prompt = tokenize(full_prompt)
        return tokenized_full_prompt

    if data_args.dataset_name == "cs":
        load_dataset_function = load_cs_dataset
    elif data_args.dataset_name == "gsm8k":
        load_dataset_function = load_gsm8k_dataset
    else:
        raise ValueError("Unsupported dataset name.")

    train_dataset, eval_dataset = None, None
    if training_args.do_train:
        train_dataset = load_dataset_function(split="train")
        validation_set_size = data_args.val_set_size
        if validation_set_size > 0:
            train_val = train_dataset.train_test_split(test_size=validation_set_size, shuffle=True, seed=42)
            train_dataset = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
            eval_dataset = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        else:
            train_dataset = train_dataset.shuffle().map(generate_and_tokenize_prompt)
            eval_dataset = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        )
    )
    model.config.use_cache = False

    # Training
    if training_args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


if __name__ == "__main__":
    main()
