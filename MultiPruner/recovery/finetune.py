# This script is based on the following source:
# https://github.com/tloen/alpaca-lora

import os
import sys
import argparse
import csv
import json
import logging

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, PeftModel
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from datasets import load_dataset

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Path to the pre-trained model."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="alpaca",
        help="Dataset to finetune.",
        choices=["alpaca"]
    )
    parser.add_argument(
        "--do_train",
        action="store_true",
        help="Flag to indicate whether to perform training."
    )
    parser.add_argument(
        "--lora",
        action="store_true",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
    )
    parser.add_argument(
        "--do_eval",
        action="store_true",
        help="Flag to indicate whether to perform evaluation."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="finetuned_model",
        help="Directory to save the fine-tuned models and evaluation results."
    )

    args = parser.parse_args()
    model_path = args.model_path
    do_train = args.do_train
    lora = args.lora
    lora_r = args.lora_r
    lora_alpha = args.lora_alpha
    lora_target_modules = args.lora_target_modules
    do_eval = args.do_eval
    dataset_name = args.dataset_name
    batch_size = args.batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps
    num_train_epochs = args.num_train_epochs
    max_steps = args.max_steps
    learning_rate = args.learning_rate
    output_path = args.output_path

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    set_seed(42)
    transformers.utils.logging.set_verbosity_info()

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": 0},
        trust_remote_code=True,
        torch_dtype="float16",
    )
    total_params = sum(p.numel() for p in model.parameters())

    if do_train and lora:

        for name, param in model.named_parameters():
            param.requires_grad = False

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=lora_target_modules.split(","),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.float32)
    elif do_eval and lora:
        model = PeftModel.from_pretrained(model, output_path, device_map={"": 0})

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if do_train:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        alpaca_template = {
            "description": "Template used by Alpaca-LoRA.",
            "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
            "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
            "response_split": "### Response:"
        }

        def generate_prompt(
            instruction: str,
            input = None,
            label = None,
        ) -> str:
            # returns the full prompt from instruction and optional input
            # if a label (=response, =output) is provided, it's also appended.
            if input:
                res = alpaca_template["prompt_input"].format(
                    instruction=instruction, input=input
                )
            else:
                res = alpaca_template["prompt_no_input"].format(
                    instruction=instruction
                )
            if label:
                res = f"{res}{label}"
            return res


        # Load data
        def tokenize(prompt, add_eos_token=True):
            result = tokenizer(
                prompt,
                truncation=True,
                max_length=256,
                padding=False,
                return_tensors=None,
            )
            if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < 256
                and add_eos_token
            ):
                result["input_ids"].append(tokenizer.eos_token_id)
                result["attention_mask"].append(1)

            result["labels"] = result["input_ids"].copy()

            return result

        def generate_and_tokenize_prompt(data_point):
            full_prompt = generate_prompt(
                data_point["instruction"],
                data_point["input"],
                data_point["output"],
            )
            tokenized_full_prompt = tokenize(full_prompt)

            user_prompt = generate_prompt(data_point["instruction"], data_point["input"])
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=False
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]  # could be sped up, probably
            return tokenized_full_prompt

        data = load_dataset("yahma/alpaca-cleaned")

        train_val = data["train"].train_test_split(
            test_size=2000, shuffle=True, seed=42
        )
        train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)

        # Initialize our Trainer
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                num_train_epochs=num_train_epochs,
                max_steps=max_steps,
                learning_rate=learning_rate,
                warmup_steps=100,
                optim="adamw_torch",
                fp16=True,
                output_dir=output_path,
                logging_steps=10,
                save_strategy="steps",
                save_steps=100,
                eval_strategy="steps",
                eval_steps=100,
                save_total_limit=2,
                load_best_model_at_end=True,
            ),
            train_dataset=train_data,
            eval_dataset=val_data,
            data_collator=DataCollatorForSeq2Seq(
                tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
            )
        )
        model.config.use_cache = False
        train_result = trainer.train()
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_data)
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluate on selected tasks
    if do_eval:
        model.eval()

        # Evaluate on wikitext2 dataset
        dataset = utils.get_dataset("wikitext2")
        test_dataset = dataset["test"]
        test_loader = utils.prepare_test_dataloader(
            dataset=test_dataset,
            tokenizer=tokenizer,
            seqlen=2048,
            batch_size=1
        )
        dataset_ppl = utils.evaluate_ppl(
            model=model,
            dataloader=test_loader,
            pad_token_id=model.config.eos_token_id,
        )
        dataset_ppl = round(dataset_ppl, 2)
        logging.info(f'wikitext2 PPL: {dataset_ppl}')

        # Evaluate on selected tasks
        hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)

        task_names = ["piqa", "winogrande", "hellaswag", "arc_easy", "arc_challenge"]
        logging.info(f"Selected Tasks: {task_names}")

        results = evaluator.simple_evaluate(hflm, tasks=task_names, num_fewshot=0, batch_size=batch_size, log_samples=False)['results']

        metric_vals = {task: round(result.get('acc_norm,none', result['acc,none']), 4) * 100 for task, result in results.items()}
        logging.info(json.dumps(metric_vals, indent=4))

        def calculate_avg_accuracy(task_names, results):
            n_tasks = len(task_names)
            acc_cumul = sum(result.get('acc_norm,none', result['acc,none']) for task, result in results.items())
            return round(acc_cumul / n_tasks, 4) * 100

        acc_avg = calculate_avg_accuracy(task_names, results)
        logging.info(f"Average accuracy across tasks: {acc_avg}")

        # Save evaluation results
        overall_results = {
            "ppl_wikitext2": dataset_ppl,
            "5cs_acc_avg": acc_avg,
            **metric_vals
        }
        eval_result_path = os.path.join(output_path, f"eval.res.json")
        with open(eval_result_path, "w") as f:
            json.dump(overall_results, f, indent=4)

        eval_result_csv_path = os.path.join(output_path, f"eval.res.csv")
        columns = ["total_params", "pruned_params", "ratio", "ppl_wikitext2"] + task_names + ["5cs_acc_avg"]
        with open(eval_result_csv_path, "w", newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            writer.writerow(overall_results)


if __name__ == "__main__":
    main()
