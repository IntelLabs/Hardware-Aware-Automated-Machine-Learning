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
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed
)

logger = logging.getLogger(__name__)

try:
    from nncf.experimental.torch.nas.bootstrapNAS.training.model_creator_helpers import (
        create_compressed_model_from_algo_names,
    )
    from nncf.torch.model_creation import create_nncf_network
    from utils.create_sqft_nncf_config import create_sqft_nncf_config
    is_nncf_available = True
except ImportError:
    is_nncf_available = False
    logger.info("NNCF is not installed. Please install it if you need NNCF.")

SPECIFIC_TEMPLATE = {
    "phi3": "<|user|>\n{instruction} <|end|>\n<|assistant|> {output}"
}


@dataclass
class SQFTTrainingArguments(TrainingArguments):
    """
    Arguments for SQFT training, including LoRA and NNCF configurations.
    """
    lora_r: int = field(default=32, metadata={"help": "Lora R dimension."})
    lora_alpha: float = field(default=64, metadata={"help": "Lora alpha."})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout."})
    target_modules: List[str] = field(
        default=None, metadata={"help": "Which module will be added the lora adapter."}
    )
    nls: bool = field(default=False, metadata={"help": "Whether to apply Neural LoRA Search (NLS) or not."})
    target_module_groups: List[str] = field(
        default=None, metadata={"help": "Grouped modules for more fine-grained control, e.g., 'q_proj,v_proj up_proj'"}
    )
    search_space: List[str] = field(default=None, metadata={"help": "Low-rank search space of NLS training."})
    nncf_config: str = field(
        default=None, metadata={"help": "NNCF configuration .json file for compression-enabled training"}
    )
    sparse_adapter: bool = field(default=False, metadata={"help": "Enable SparsePEFT."})
    quantization_aware: bool = field(default=False, metadata={"help": "Enable quantization-aware SparsePEFT."})

    def __post_init__(self):
        super().__post_init__()
        if self.target_module_groups is not None:
            target_modules = []
            target_module_groups = []
            for group in self.target_module_groups:
                modules = group.split(",")
                target_modules.extend(modules)
                target_module_groups.append(modules)
            if self.target_modules is None:
                self.target_modules = target_modules
            self.target_module_groups = target_module_groups
            assert self.search_space is not None and len(self.search_space) == len(self.target_module_groups)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to the data used for training and evaluation.
    """
    dataset_path: Optional[str] = field(default=None, metadata={"help": "The path of the dataset to use."})
    val_set_size: int = field(default=0, metadata={"help": "Validation set size."})
    cutoff_len: int = field(default=256, metadata={"help": "Cutoff length for tokenization."})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    dtype: str = field(default="float16", metadata={"help": "Data type for model."})
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

    nncf_config = None
    compression_ctrl = None
    if training_args.nls:
        if not is_nncf_available:
            raise ImportError("NNCF is not installed. Please install it.")
        nncf_config = create_sqft_nncf_config(
            model,
            output_dir=training_args.output_dir,
            num_train_epochs=training_args.num_train_epochs,
            learning_rate=training_args.learning_rate,
            target_module_groups=training_args.target_module_groups,
            search_space=training_args.search_space,
        )

        nncf_network = create_nncf_network(model, nncf_config)
        compression_ctrl, model = create_compressed_model_from_algo_names(
            nncf_network, nncf_config, algo_names=["nls"]
        )

    # Init SQFTQuantAwareLinear params
    if training_args.quantization_aware:
        from modules.sqft_linear import SQFTQuantAwareLinear

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

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name is not None else model_args.model_name_or_path,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    prompt_template = SPECIFIC_TEMPLATE.get(model.config.model_type, None)

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

    def generate_prompt(data_point, template=None):
        if template is None:
            if data_point["input"]:
                return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                        ### Instruction:
                        {data_point["instruction"]}

                        ### Input:
                        {data_point["input"]}

                        ### Response:
                        {data_point["output"]}"""
            else:
                return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                        ### Instruction:
                        {data_point["instruction"]}

                        ### Response:
                        {data_point["output"]}"""
        return template.replace("{instruction}", data_point["instruction"]).replace("{output}", data_point["output"])

    def generate_and_tokenize_prompt(data_point):
        """
        Generate and tokenize the prompt from the data point.
        """
        full_prompt = generate_prompt(data_point, template=prompt_template)
        tokenized_full_prompt = tokenize(full_prompt)
        return tokenized_full_prompt

    train_dataset, eval_dataset = None, None
    if training_args.do_train:
        train_dataset = load_dataset("json", data_files=data_args.dataset_path)
        validation_set_size = data_args.val_set_size
        if validation_set_size > 0:
            train_val = train_dataset["train"].train_test_split(test_size=validation_set_size, shuffle=True, seed=42)
            train_dataset = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
            eval_dataset = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        else:
            train_dataset = train_dataset["train"].shuffle().map(generate_and_tokenize_prompt)
            eval_dataset = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        compression_ctrl=compression_ctrl,
    )

    if compression_ctrl is not None:
        if not (training_args.local_rank in [-1, 0] or training_args.no_cuda):
            compression_ctrl.distributed()
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
