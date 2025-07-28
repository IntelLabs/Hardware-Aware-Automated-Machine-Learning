import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import json
import time
import sys

import torch
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    set_seed
)

from lmms_eval import evaluator

from ct3_config import add_ct3_args, parse_ct3_args
from models.model import convert_to_ttt_model
from models.lmms_model import Qwen2_VL_Lmms_Extended, Qwen2_5_VL_Lmms_Extended


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the pre-trained model.",
        required=True
    )
    parser.add_argument(
        "--task",
        type=str,
        default="mmmu_pro",
        help="Task of lmms evaluation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Limit for the number of evaluation samples."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Directory to save the results."
    )
    add_ct3_args(parser)
    args = parser.parse_args()
    model_path = args.model_path
    task = args.task
    limit = args.limit
    output_path = args.output_path
    ct3_config = parse_ct3_args(args)

    set_seed(42)

    # Generate result folder name
    result_folder_name = ct3_config.generate_result_folder_name(task, limit)
    output_path = os.path.join(output_path, result_folder_name)

    result_file_path = os.path.join(output_path, 'result.json')
    if os.path.exists(result_file_path):
        print(f"The file {result_file_path} already exists. Exiting the program.")
        sys.exit(1)
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, model_max_length=4096)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    processor = AutoProcessor.from_pretrained(model_path)
    
    if "Qwen2.5-VL" in model_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        ).eval()
    elif "Qwen2-VL" in model_path:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        ).eval()
    else:
        raise ValueError(f"Unsupported model type: {model_path}")
    
    model = convert_to_ttt_model(model, tokenizer, processor, ct3_config)

    print(f"Start evaluation...")
    if model.config.model_type == "qwen2_5_vl":
        # the max_pixels is set to 12845056 for Qwen2.5-VL
        # (https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/examples/models/qwen25vl.sh)
        lmms = Qwen2_5_VL_Lmms_Extended(
            model=model, 
            tokenizer=tokenizer,
            processor=processor,
            max_pixels=12845056,
            interleave_visuals=False,
        )
    elif model.config.model_type == "qwen2_vl":
        # the max_pixels is set to 2359296 for Qwen2-VL 
        # (https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/examples/models/qwen2vl.sh)
        lmms = Qwen2_VL_Lmms_Extended(
            model=model, 
            tokenizer=tokenizer,
            processor=processor,
            max_pixels=2359296,
            interleave_visuals=False,
        )
    else:
        raise ValueError(f"Unsupported model type: {model.config.model_type}")

    print(f"Selected Tasks: {task}")
    if limit == -1:
        limit = None

    start_time = time.time()
    lmms_eval_outputs = evaluator.simple_evaluate(
        model=lmms, 
        tasks=[task],
        batch_size=1, 
        limit=limit,
        log_samples=True,
    )
    end_time = time.time()

    # Save results
    samples = lmms_eval_outputs["samples"]
    with open(os.path.join(output_path, "log_samples.json"), "w") as f:
        json.dump(samples, f, indent=4)

    results = lmms_eval_outputs['results']
    results["total_time_taken"] = end_time - start_time
    results["config"] = vars(args)
    print(json.dumps(results, indent=4))

    with open(os.path.join(output_path, "result.json"), "w") as f:
        json.dump(results, f, indent=4)

    # save analysis result
    if ct3_config.ttt and ct3_config.analysis:
        model.save_analysis_data(output_path)


if __name__ == "__main__":
    main()
