#!/usr/bin/env python
# coding=utf-8
import copy
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from dataclasses import field
from typing import Optional

import datasets
import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig
from peft import PeftModel
from peft import get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import GenerationConfig
from transformers import HfArgumentParser
from transformers import LlamaTokenizer
from transformers import Trainer
from transformers import TrainingArguments
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils import send_example_telemetry
from transformers.utils.versions import require_version

from nncf import NNCFConfig
from nncf.experimental.torch.nas.bootstrapNAS.elasticity.elasticity_dim import ElasticityDim
from nncf.experimental.torch.nas.bootstrapNAS.training.model_creator_helpers import (
    create_compressed_model_from_algo_names,
)
from nncf.experimental.torch.nas.bootstrapNAS import BaseSearchAlgorithm
from nncf.torch.model_creation import create_nncf_network

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.31.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

logger = logging.getLogger(__name__)
TEST_DATASETS = ["AQuA", "mawps", "gsm8k", "SVAMP"]


@dataclass
class LonasTrainingArguments(TrainingArguments):
    lora_r: int = field(default=32, metadata={"help": "Lora R dimension."})
    lora_alpha: float = field(default=64, metadata={"help": " Lora alpha."})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout."})
    target_modules: str = field(
        default="q_proj,v_proj", metadata={"help": "Which module will be added the lora adapter."}
    )
    lora: bool = field(default=False, metadata={"help": "Whether to apply lora or not."})
    train_on_inputs: bool = field(default=True)
    do_test: bool = field(default=False)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_path: Optional[str] = field(default=None, metadata={"help": "The path of the dataset to use."})
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    val_set_size: int = field(default=0)
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximal total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximal length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
    cutoff_len: int = field(
        default=256,
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    lora_weights: str = field(default=None)
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, LonasTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_glue", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # load model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=model_args.cache_dir,
    )

    if training_args.lora and model_args.lora_weights is None:
        logger.info("Adding LoRA modules...")
        config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            target_modules=training_args.target_modules.split(","),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
    elif training_args.lora:
        logger.info("Loading LoRA modules...")
        model = PeftModel.from_pretrained(model, model_args.lora_weights, torch_dtype=torch.float16, device_map="auto")

    nncf_config = None
    if training_args.nncf_config is not None:
        nncf_config = NNCFConfig.from_json(training_args.nncf_config)

        if nncf_config.get("log_dir") is None:
            nncf_config["log_dir"] = training_args.output_dir

        if not os.path.exists(training_args.output_dir) and training_args.local_rank in [-1, 0]:
            os.makedirs(nncf_config["log_dir"])

    compression_ctrl = None
    if nncf_config is not None:
        nncf_network = create_nncf_network(model, nncf_config)
        algo_name = nncf_config.get("bootstrapNAS", {}).get("training", {}).get("algorithm", "progressive_shrinking")
        compression_ctrl, model = create_compressed_model_from_algo_names(
            nncf_network, nncf_config, algo_names=[algo_name]
        )

    # load tokenizer
    if "llama" in model_args.model_name_or_path:
        tokenizer = LlamaTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token

    # Load data
    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
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

    def generate_prompt(data_point):
        # sorry about the formatting disaster gotta move fast
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

    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt)
        if not training_args.train_on_inputs:
            user_prompt = generate_prompt({**data_point, "output": ""})
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]
        return tokenized_full_prompt

    train_dataset, eval_dataset = None, None
    if training_args.do_train or training_args.do_search:
        data = load_dataset("json", data_files=data_args.dataset_path)

        val_set_size = data_args.val_set_size
        if val_set_size > 0:
            train_val = data["train"].train_test_split(test_size=val_set_size, shuffle=True, seed=42)
            train_dataset = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
            eval_dataset = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
        else:
            train_dataset = data["train"].shuffle().map(generate_and_tokenize_prompt)
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

    if nncf_config is not None:
        if not (training_args.local_rank in [-1, 0] or training_args.no_cuda):
            compression_ctrl.distributed()

    model.config.use_cache = False

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    def extract_answer_number(dataset, sentence: str) -> float:
        dataset = dataset.lower()
        if dataset in ["gsm8k", "svamp", "mawps"]:
            sentence = sentence.replace(",", "")
            pred = [s for s in re.findall(r"-?\d+\.?\d*", sentence)]
            if not pred:
                return float("inf")
            pred_answer = float(pred[-1])
        else:
            raise NotImplementedError(" not support dataset: {}".format(dataset))
        if isinstance(pred_answer, str):
            try:
                pred_answer = float(pred_answer)
            except ValueError as e:
                pred_answer = float("inf")
        return pred_answer

    def extract_answer_letter(sentence: str) -> str:
        sentence_ = sentence.strip()
        pred_answers = re.findall(r"A|B|C|D|E", sentence_)
        if pred_answers:
            if not pred_answers:
                return ""
            return pred_answers[0]
        else:
            return ""

    def load_test_data(test_dataset) -> list:
        """
        read data from dataset file
        """
        file_path = f"datasets/{test_dataset}/test.json"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"can not find dataset file : {file_path}")
        json_data = json.load(open(file_path, "r"))
        return json_data

    def generate_prompt_eval(instruction, input=None):
        if input:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                            ### Instruction:
                            {instruction}

                            ### Input:
                            {input}

                            ### Response:
                            """
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                            ### Instruction:
                            {instruction}

                            ### Response:
                            """

    def evaluate_one_sample(
        instruction,
        input=None,
        model=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=256,
        **kwargs,
    ):
        prompt = generate_prompt_eval(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return output.split("### Response:")[1].strip()

    def evaluate(model, dataset_name, save_file):
        model.eval()
        dataset = load_test_data(dataset_name)
        total = len(dataset)

        correct = 0
        miss = 0.001
        output_data = []
        for idx, data in tqdm(enumerate(dataset)):
            instruction = data.get("instruction")
            outputs = evaluate_one_sample(instruction, model=model)
            label = data.get("answer")
            flag = False
            if dataset_name.lower() in ["aqua"]:
                predict = extract_answer_letter(outputs)
                if label == predict:
                    correct += 1
                    flag = True
            else:
                if isinstance(label, str):
                    label = float(label)
                predict = extract_answer_number(dataset_name, outputs)
                if abs(label - predict) <= miss:
                    correct += 1
                    flag = True
            new_data = copy.deepcopy(data)
            new_data["output_pred"] = outputs
            new_data["pred"] = predict
            new_data["flag"] = flag
            output_data.append(new_data)
            logger.info(f"id: {idx + 1}")
            logger.info(outputs)
            logger.info(f"prediction: {predict}")
            logger.info(f"label: {label}")
            logger.info(f"\rtest:{idx + 1}/{total} | accuracy {correct}  {correct / (idx + 1)}")
            with open(save_file, "w+") as f:
                json.dump(output_data, f, indent=4)

        acc = correct / total
        return acc

    def test_subnetwork(subnetwork, name):
        logger.info(f"*** Evaluation - {name} ***")
        non_zero_params = sum([(param.data != 0).sum().item() for _, param in subnetwork.named_parameters()])
        macs, weights = trainer.compression_ctrl.multi_elasticity_handler.count_flops_and_weights_for_active_subnet()
        metrics = {
            f"{name}_non_zero_params": non_zero_params,
            f"{name}_macs": str(macs / 2000000),
            f"{name}_weights": str(weights),
        }
        trainer.save_metrics("eval", metrics)
        trainer.log_metrics("eval", metrics)

        all_results = []
        metrics = {}
        for test_dataset in TEST_DATASETS:
            logger.info(f"*** Evaluation on {test_dataset} ***")
            save_file = os.path.join(training_args.output_dir, f"{name}.{test_dataset}.res.json")
            accuracy = evaluate(subnetwork, test_dataset, save_file)
            all_results.append(accuracy)
            metrics[f"{name}_{test_dataset}_accuracy"] = accuracy
            trainer.save_metrics("eval", metrics)
        metrics[f"{name}_avg_accuracy"] = sum(all_results) / len(all_results)
        trainer.save_metrics("eval", metrics)
        trainer.log_metrics("eval", metrics)

    # test accuracy (heuristic)
    if training_args.do_test and training_args.local_rank <= 0:
        if compression_ctrl is not None:
            trainer.compression_ctrl.multi_elasticity_handler.enable_all()
            compression_ctrl.multi_elasticity_handler.width_handler.width_num_params_indicator = -1
            # Heuristic subnetwork
            heuristic_config = {
                k: v[(len(v) - 1) // 2] for k, v in compression_ctrl.multi_elasticity_handler.width_search_space.items()
            }
            heuristic_config = {ElasticityDim.WIDTH: heuristic_config}
            trainer.compression_ctrl.multi_elasticity_handler.activate_subnet_for_config(heuristic_config)
            test_subnetwork(trainer.model, "heuristic")
        else:
            # LoRA
            all_results = []
            for test_dataset in TEST_DATASETS:
                logger.info(f"*** Evaluation on {test_dataset} ***")
                save_file = os.path.join(training_args.output_dir, f"{test_dataset}.res.json")
                non_zero_params = sum([(param.data != 0).sum().item() for _, param in trainer.model.named_parameters()])
                accuracy = evaluate(trainer.model, test_dataset, save_file)
                all_results.append(accuracy)
                metrics = {
                    f"{test_dataset}_accuracy": accuracy,
                    "non_zero_params": non_zero_params,
                }
                trainer.save_metrics("eval", metrics)
            avg_metrics = {
                "avg_accuracy": sum(all_results) / len(all_results),
            }
            trainer.save_metrics("eval", avg_metrics)
            trainer.log_metrics("eval", avg_metrics)

    # Searching
    if training_args.do_search and nncf_config is not None and training_args.local_rank <= 0:
        logger.info("*** Search ***")
        trainer.compression_ctrl.multi_elasticity_handler.enable_all()
        search_algo = BaseSearchAlgorithm.from_config(trainer.model, trainer.compression_ctrl, nncf_config)

        def is_number(input_str):
            try:
                float(input_str)
                return True
            except ValueError:
                return False

        def validate_model_fn(model_, eval_dataset):
            correct = 0
            miss = 0.001
            for data in tqdm(eval_dataset):
                instruction = data.get('instruction')
                outputs = evaluate_one_sample(instruction, model=model_)
                label = data.get('answer')
                if is_number(label):
                    if isinstance(label, str):
                        label = float(label)
                    predict = extract_answer_number("mawps", outputs)
                    if abs(label - predict) <= miss:
                        correct += 1
                else:
                    predict = extract_answer_letter(outputs)
                    if label == predict:
                        correct += 1
            acc = correct / len(eval_dataset)
            return acc

        # Test Maximal subnetwork and Heuristic subnetwork on the validation dataset
        # Maximal
        trainer.compression_ctrl.multi_elasticity_handler.activate_supernet()
        max_eval_acc = validate_model_fn(trainer.model, eval_dataset)

        # Heuristic
        compression_ctrl.multi_elasticity_handler.width_handler.width_num_params_indicator = -1
        heuristic_config = {k: v[(len(v) - 1) // 2]
                            for k, v in compression_ctrl.multi_elasticity_handler.width_search_space.items()}
        heuristic_config = {
            ElasticityDim.WIDTH: heuristic_config
        }
        trainer.compression_ctrl.multi_elasticity_handler.activate_subnet_for_config(heuristic_config)
        heu_eval_acc = validate_model_fn(trainer.model, eval_dataset)

        metrics = {
            "val_maximal_accuracy": max_eval_acc,
            "val_heuristic_accuracy": heu_eval_acc,
        }
        trainer.save_metrics("eval", metrics)
        trainer.log_metrics("eval", metrics)

        elasticity_ctrl, best_config, performance_metrics = search_algo.run(
            validate_model_fn, eval_dataset, training_args.output_dir
        )

        search_algo.search_progression_to_csv()
        search_algo.evaluators_to_csv()
        search_algo.visualize_search_progression()

        logger.info("Best config: {best_config}".format(best_config=best_config))
        logger.info("Performance metrics: {performance_metrics}".format(performance_metrics=performance_metrics))
        trainer.save_metrics("eval", {
            "performance_metrics": list(performance_metrics)
        })

        # test best config
        trainer.compression_ctrl.multi_elasticity_handler.activate_subnet_for_config(best_config)
        best_eval_acc = validate_model_fn(trainer.model, eval_dataset)
        trainer.save_metrics("eval", {
            "val_best_accuracy": best_eval_acc
        })

    kwargs = {"finetuned_from": model_args.model_name_or_path}

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
