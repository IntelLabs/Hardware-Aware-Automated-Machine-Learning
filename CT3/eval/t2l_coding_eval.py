# This script is based on modifications from 
# https://github.com/bigcode-project/bigcode-evaluation-harness/blob/6116c6a9a5672c69bd624373cfbc8938b7acc249/main.py
import os
import fnmatch
import json
import sys
import time

import datasets
import torch
import transformers
from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed
)

from bigcode_eval.arguments import EvalArguments
from bigcode_eval.evaluator import Evaluator
from bigcode_eval.tasks import ALL_TASKS

from t2l_config import add_t2l_args, parse_t2l_args
from models.t2l_model import load_t2l_model


class MultiChoice:
    def __init__(self, choices):
        self.choices = choices

    # Simple wildcard support (linux filename patterns)
    def __contains__(self, values):
        for value in values.split(","):
            if len(fnmatch.filter(self.choices, value)) == 0:
                return False

        return True

    def __iter__(self):
        for choice in self.choices:
            yield choice


def parse_args():
    parser = HfArgumentParser(EvalArguments)
    parser.add_argument(
        "--model",
        default="codeparrot/codeparrot-small",
        help="Model to evaluate, provide a repo name in Hugging Face hub or a local path",
    )
    parser.add_argument(
        "--modeltype",
        default="causal",
        help="AutoModel to use, it can be causal or seq2seq",
    )
    parser.add_argument(
        "--peft_model",
        type=str,
        default=None,
        help="Adapter to the PEFT base model. Can be utilized for loading PEFT adapters such as a LoRA trained model. The --model parameter needs to be the base model.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Model revision to use",
    )
    parser.add_argument(
        "--use_auth_token",
        action="store_true",
        help="Use the token generated when running `huggingface-cli login` (necessary for private model).",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Use a model with custom code, this requires executing code by the author of the model.",
    )
    parser.add_argument(
        "--tasks",
        default=None,
        choices=MultiChoice(ALL_TASKS),
        help=f"Evaluation tasks from {ALL_TASKS}",
    )
    parser.add_argument(
        "--instruction_tokens",
        default=None,
        help="A series of instruction tokens used for instruction-tuning benchamrks separated by comma e.g. <user_message>,<end_user_message>,<assistant_message>",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation on each worker, can be larger for HumanEval",
    )
    parser.add_argument(
        "--max_length_generation",
        type=int,
        default=512,
        help="Maximum length of generated sequence (prompt+generation)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        help="Model precision, from: fp32, fp16 or bf16",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8bit",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4bit",
    )
    parser.add_argument(
        "--left_padding",
        action="store_true",
        help="Force left padding, needed for models like chatglm3-6b",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of samples to solve and evaluate from the benchmark",
    )
    parser.add_argument(
        "--limit_start",
        type=int,
        default=0,
        help="Optional offset to start from when limiting the number of samples",
    )
    parser.add_argument(
        "--save_every_k_tasks",
        type=int,
        default=-1,
        help="Optional saving after every k tasks",
    )
    parser.add_argument(
        "--postprocess",
        action="store_false",
        help="Postprocess model outputs before execution, always on except during generation tests",
    )
    parser.add_argument(
        "--allow_code_execution",
        action="store_true",
        help="Allow code evaluation to execute external/untrusted Python code on your machine",
    )
    parser.add_argument(
        "--generation_only",
        action="store_true",
        help="Do code generation but no evaluation",
    )
    parser.add_argument(
        "--load_generations_path",
        type=str,
        default=None,
        help="Path of file with previously generated solutions, if provided generation is skipped and only evaluation is done",
    )
    parser.add_argument(
        "--load_data_path",
        type=str,
        default=None,
        help="Path of additional data to load for the tasks",
    )
    parser.add_argument(
        "--metric_output_path",
        type=str,
        default="evaluation_results.json",
        help="Path to save the results",
    )
    parser.add_argument(
        "--save_generations",
        action="store_true",
        help="Whether to save code generations",
    )
    parser.add_argument(
        "--load_generations_intermediate_paths",
        type=str,
        nargs="*",
        help="List of paths for saving the intermediate code generations",
    )
    parser.add_argument(
        "--save_generations_path",
        type=str,
        default="generations.json",
        help="Path for saving the code generations",
    )
    parser.add_argument(
        "--save_references",
        action="store_true",
        help="Whether to save reference solutions/tests",
    )
    parser.add_argument(
        "--save_references_path",
        type=str,
        default="references.json",
        help="Path for saving the references solutions/tests",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="prompt",
        help="Prompt type to use for generation in HumanEvalPack tasks",
    )
    parser.add_argument(
        "--max_memory_per_gpu",
        type=str,
        default=None,
        help="Max memroy to allocate per gpu, you can also use 'auto'",
    )
    parser.add_argument(
        "--check_references",
        action="store_true",
        help="Don't run generation but benchmark groundtruth (useful for debugging)",
    )
    # t2l
    parser.add_argument(
        "--output_path",
        type=str,
        help="Directory to save the results."
    )
    add_t2l_args(parser)
    return parser.parse_args()


def pattern_match(patterns, source_list):
    """Returns a list containing all values of the source_list that
    match at least one of the patterns"""
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)


def get_gpus_max_memory(max_memory, num_gpus):
    max_memory = {i: max_memory for i in range(num_gpus)}
    print("Loading model via these GPUs & max memories: ", max_memory)
    return max_memory


def main():
    args = parse_args()
    t2l_config = parse_t2l_args(args)
    output_path = args.output_path
    transformers.logging.set_verbosity_error()
    datasets.logging.set_verbosity_error()
    set_seed(42)

    # Generate result folder name
    result_folder_name = t2l_config.generate_result_folder_name(args.tasks, args.limit)
    output_path = os.path.join(output_path, result_folder_name)
    result_file_path = os.path.join(output_path, 'result.json')
    if os.path.exists(result_file_path):
        print(f"The file {result_file_path} already exists. Exiting the program.")
        sys.exit(1)
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    if args.tasks is None:
        task_names = ALL_TASKS
    else:
        task_names = pattern_match(args.tasks.split(","), ALL_TASKS)

    accelerator = Accelerator()
    if accelerator.is_main_process:
        print(f"Selected Tasks: {task_names}")

    results = {}
    start_time = time.time()
    # here we generate code and save it (evaluation is optional but True by default)
    dict_precisions = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if args.precision not in dict_precisions:
        raise ValueError(
            f"Non valid precision {args.precision}, choose from: fp16, fp32, bf16"
        )

    model_kwargs = {
        "revision": args.revision,
        "trust_remote_code": args.trust_remote_code,
        "token": args.use_auth_token,
    }
    if args.load_in_8bit:
        print("Loading model in 8bit")
        model_kwargs["load_in_8bit"] = args.load_in_8bit
        model_kwargs["device_map"] = {"": accelerator.process_index}
    elif args.load_in_4bit:
        print("Loading model in 4bit")
        model_kwargs["load_in_4bit"] = args.load_in_4bit
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["bnb_4bit_compute_dtype"] = torch.float16            
        model_kwargs["device_map"] = {"": accelerator.process_index}
    else:
        print(f"Loading model in {args.precision}")
        model_kwargs["torch_dtype"] = dict_precisions[args.precision]

        if args.max_memory_per_gpu:
            if args.max_memory_per_gpu != "auto":
                model_kwargs["max_memory"] = get_gpus_max_memory(
                    args.max_memory_per_gpu, accelerator.num_processes
                )
                model_kwargs["offload_folder"] = "offload"
            else:
                model_kwargs["device_map"] = "auto"
                print("Loading model in auto mode")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        **model_kwargs,
    )

    if args.left_padding:
        # left padding is required for some models like chatglm3-6b
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            revision=args.revision,
            trust_remote_code=args.trust_remote_code,
            token=args.use_auth_token,
            padding_side="left",  
        )
    else:
        # used by default for most models
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            revision=args.revision,
            trust_remote_code=args.trust_remote_code,
            token=args.use_auth_token,
            truncation_side="left",
            padding_side="right",  
        )
    if not tokenizer.eos_token:
        if tokenizer.bos_token:
            tokenizer.eos_token = tokenizer.bos_token
            print("bos_token used as eos_token")
        else:
            raise ValueError("No eos_token or bos_token found")
    try:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Some models like CodeGeeX2 have pad_token as a read-only property
    except AttributeError:
        print("Not setting pad_token to eos_token")
        pass

    # t2l
    model = load_t2l_model(t2l_config, model=model, tokenizer=tokenizer)

    evaluator = Evaluator(accelerator, model, tokenizer, args)

    if (
        args.load_generations_intermediate_paths
        and len(args.load_generations_intermediate_paths) != len(task_names)
    ):
        raise ValueError(
            "If passing --load_generations_intermediate_paths, \
            must pass equal number of files as number of tasks"
        )

    for idx, task in enumerate(task_names):
        intermediate_generations = None
        if args.load_generations_intermediate_paths:
            with open(args.load_generations_intermediate_paths[idx], "r") as f_in:
                # intermediate_generations: list[list[str | None]] of len n_tasks
                # where list[i] = generated codes or empty
                intermediate_generations = json.load(f_in)

        if args.generation_only:
            if accelerator.is_main_process:
                print("generation mode only")
            generations, references = evaluator.generate_text(
                task, intermediate_generations=intermediate_generations
            )
            if accelerator.is_main_process:
                save_generations_path = f"{os.path.splitext(args.save_generations_path)[0]}_{task}.json"
                save_references_path = f"references_{task}.json"
                evaluator.save_json_files(
                    generations,
                    references,
                    save_generations_path,
                    save_references_path,
                )
        else:
            results[task] = evaluator.evaluate(
                task, intermediate_generations=intermediate_generations
            )

    end_time = time.time()
    results["total_time_taken"] = end_time - start_time
    results["config"] = vars(args)
    print(json.dumps(results, indent=4))

    with open(os.path.join(output_path, 'result.json'), "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
