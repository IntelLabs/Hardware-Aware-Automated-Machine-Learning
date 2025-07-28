# Based on https://github.com/huggingface/optimum/blob/main/tests/benchmark/benchmark_gptq.py

import argparse
import gc
import time
import contextlib

import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    GenerationConfig,
    AutoModel,
    Qwen2_5_VLForConditionalGeneration,
    Gemma3ForConditionalGeneration,
)
from torch.profiler import profile, ProfilerActivity

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-batches",
        type=int,
        default=10,
        help="",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model to benchmark",
    )
    parser.add_argument(
        "--prompt-length",
        type=int,
        default=256,
        help="",
    )
    parser.add_argument(
        "--new-tokens",
        type=int,
        default=256,
        help="",
    )

    inference_phase_group = parser.add_argument_group("Inference phase")
    inference_phase_group.add_argument(
        "--prefill",
        action="store_true",
        help="For decoder models, benchmark only the prefill step with `prompt_length`.",
    )
    inference_phase_group.add_argument(
        "--decode",
        action="store_true",
        help="For decoder models, benchmark only the decode step (simulated using a prompt length of 1 & KV cache).",
    )

    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Use the parameter ranges for (batch_size, prompt_length, new_tokens) defined in the .py file instead of the CLI ones.",
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Calculate the generate speed (prompt processing + token generation)",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Revision of the model to benchmark",
    )

    return parser


def measure_flops(
    model, num_batches: int, input_ids: torch.Tensor, masks: torch.Tensor, generation_config=None
):
    assert generation_config.min_new_tokens == generation_config.max_new_tokens

    flops_list = []

    for _ in tqdm(range(num_batches)):
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_flops=True
        ) as prof:
            _ = model.generate(input_ids, attention_mask=masks, generation_config=generation_config)
        total_flops = prof.key_averages().total_average().flops
        flops_list.append(total_flops)

    return np.mean(flops_list)


def warmup(
    model,
    input_ids: torch.Tensor,
    masks: torch.Tensor,
    new_tokens: int,
    pad_token_id: int,
):
    print("Warmup...")
    gen_config = GenerationConfig(
        max_new_tokens=new_tokens,
        min_new_tokens=new_tokens,
        use_cache=True,
        pad_token_id=pad_token_id,
        num_beams=1,
        do_sample=False,
        eos_token_id=None,  # This is required for min_new_tokens to actually have an effect.
    )
    model.generation_config.eos_token_id = None  # greedy_search falls back on this eos_token_id that we need to set to None as well for min_new_tokens to have an effect.
    res = model.generate(input_ids, attention_mask=masks, generation_config=gen_config)

    assert res.shape[1] == new_tokens + input_ids.shape[1]
    del res

    return gen_config


def benchmark_flops(
    model,
    input_ids: torch.Tensor,
    masks: torch.Tensor,
    num_batches: int,
    new_tokens: int,
    pad_token_id: int,
):
    gc.collect()

    gen_config = warmup(
        model,
        input_ids,
        masks,
        new_tokens,
        pad_token_id,
    )

    print("Measuring FLOPs...")
    mean_flops = measure_flops(model, num_batches, input_ids, masks, gen_config)

    return mean_flops


parser = get_parser()
args = parser.parse_args()

if args.sweep:      
    batch_sizes = [1, 2, 4, 8, 16]
    prompt_lengths = [128]
    new_tokens = [4]
else:
    batch_sizes = [args.batch_size]
    prompt_lengths = [args.prompt_length]
    new_tokens = [args.new_tokens]

if args.prefill:
    print("Running the prefill benchmark: generating only one new token.")
    new_tokens = [1]
else:
    assert args.decode
    print("Running the decode benchmark: setting the prompt length to 1.")
    prompt_lengths = [1]


tokenizer = AutoTokenizer.from_pretrained(args.model)

if not hasattr(tokenizer, "pad_token") or tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

task = args.task
load_start = time.time_ns()

if args.model.startswith("Qwen/Qwen2.5-VL"):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
elif args.model.startswith("google/gemma-3"):
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model, 
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
    ).eval()
else:
    raise ValueError(f"Model {args.model} is not supported for benchmarking.")


device = torch.device("cuda")

load_end = time.time_ns()

act_order = None
bits = None
group_size = None
kernel = None


def raise_if_module_not_found(model: torch.nn.Module, layer_class, layer_name):
    for _, module in model.named_modules():
        if isinstance(module, layer_class):
            break
    else:
        raise ValueError(f"{layer_name} layer not found")


load_time = (load_end - load_start) * 1e-9
print(f"Model load time: {load_time:.1f} s")

model = model.eval()

file_name = "log_{}".format(args.model.replace("/", "-"))
if args.prefill:
    file_name = file_name + "_prefill"
else:
    assert args.decode
    file_name = file_name + "_decode"

if args.generate:
    output_file = open(file_name + ".csv", "w")
    header = "num_batches, batch_size, prompt_length, new_tokens, Load time (s), FLOPs\n"
    output_file.write(header)

    flops_results = {}

    for batch_size in tqdm(batch_sizes):
        for prompt_length in tqdm(prompt_lengths):
            for new_token in tqdm(new_tokens):
                print(f"---- Running: batch_size={batch_size}, prompt_length={prompt_length}, new_tokens={new_token}")
                input_ids = torch.randint(1, 1000, size=(batch_size, prompt_length)).to(device)
                masks = torch.ones(batch_size, prompt_length, dtype=torch.int32).to(device)

                with torch.no_grad():

                    mean_flops = benchmark_flops(
                        model,
                        input_ids,
                        masks,
                        args.num_batches,
                        new_token,
                        tokenizer.pad_token_id,
                    )

                index = (batch_size, prompt_length, new_token)

                flops_results[index] = mean_flops

                print(
                    f"FLOPs: {mean_flops:.3f}"
                )

                line = "{},{},{},{},{},{}\n".format(
                    args.num_batches,
                    batch_size,
                    prompt_length,
                    new_token,
                    f"{load_time:.2f}",
                    f"{mean_flops:.2f}",
                )
                print(header)
                print(line)
                output_file.write(line)
    output_file.close()
    print(f"Results saved to {file_name}.csv")
