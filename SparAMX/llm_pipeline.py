from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
import time
from collections import defaultdict
from functools import wraps
from statistics import median
from layer.dense_linear import DenseLinear
from layer.sparse_linear import SparseLinear
from layer.quantized_sparse_linear import QuantizedSparseLinear
from layer.quantized_dense_linear import QuantizedDenseLinear
from layer.avx_sparse_linear import AvxSparseLinear
from typing import Dict, Optional
# from layer.onednn_linear import OneDnnLinear
from safetensors import safe_open
import os
import argparse
import math
import types
import inspect
from transformers import StoppingCriteria, StoppingCriteriaList
from custom_llama_attention import CustomLlamaAttention
from transformers.models.llama.modeling_llama import LlamaSdpaAttention

import functools

# Set OpenMP environment variable for thread affinity
os.environ['OMP_PROC_BIND'] = 'true'  # Bind threads to cores
os.environ['OMP_PLACES'] = 'cores'  # Each thread will be bound to a core

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B")
parser.add_argument("--saved_model_path", type=str, default='default')
parser.add_argument("--context_length", type=int, default=16)
parser.add_argument("--num_generated_tokens", type=int, default=1)
parser.add_argument("--mode", type=str, default="stock")
parser.add_argument("--num_threads", type=int, default=64)
parser.add_argument("--batch_size", type=int, default=1)
# parser.add_argument("--num_warmup", type=int, default=16)
# parser.add_argument("--num_iterations", type=int, default=8)
parser.add_argument("--num_warmup", type=int, default=8)
parser.add_argument("--num_iterations", type=int, default=8)
parser.add_argument("--enable_custom_attention", action=argparse.BooleanOptionalAction, type=bool, default=False)
parser.add_argument("--use_custom_k", action=argparse.BooleanOptionalAction, type=bool, default=False)
parser.add_argument("--use_custom_v", action=argparse.BooleanOptionalAction, type=bool, default=False)
parser.add_argument("--k_pruning", type=int, default=50)
parser.add_argument("--v_pruning", type=int, default=50)
parser.add_argument("--enable_int8", action=argparse.BooleanOptionalAction, type=bool, default=False)
parser.add_argument("--use_cached_prefill", action=argparse.BooleanOptionalAction, type=bool, default=False)
parser.add_argument("--attention_mode", type=str, default="sparse")

args = parser.parse_args()

modes_to_configs = {
    'stock': {'custom_class': None, 'enable_custom_layer': False, 'load_presaved_model': False},
    'dense': {'custom_class': DenseLinear, 'enable_custom_layer': True, 'load_presaved_model': False},
    'sparse': {'custom_class': SparseLinear, 'enable_custom_layer': True, 'load_presaved_model': True},
    'sparseq': {'custom_class': QuantizedSparseLinear, 'enable_custom_layer': True, 'load_presaved_model': True},
    'denseq': {'custom_class': QuantizedDenseLinear, 'enable_custom_layer': True, 'load_presaved_model': True},
    'avx_sparse': {'custom_class': AvxSparseLinear, 'enable_custom_layer': True, 'load_presaved_model': True},
}

model_id = args.model_id
mode = args.mode
enable_custom_layer = modes_to_configs[mode]['enable_custom_layer']
load_presaved_model = modes_to_configs[mode]['load_presaved_model']
custom_layer_class = modes_to_configs[mode]['custom_class']
num_threads = args.num_threads
saved_model_path = args.saved_model_path
if saved_model_path == 'default':
    saved_model_path = f"processed_shears_llama-3-8b-0-sparsity_{num_threads}_threads"



class KVCacheManager:
    def __init__(self, cache_dir: str = "persisted_kv_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_cache_path(self, input_length: int, model_name: str, file_type: str) -> str:
        return os.path.join(self.cache_dir, f"{model_name}_len{input_length}_{file_type}.pt")
    
    def save_kv_cache(self, kv_cache: Dict[str, torch.Tensor], input_ids, input_length: int, model_name: str):
        cache_path = self.get_cache_path(input_length, model_name, 'kv_cache')
        torch.save(kv_cache, cache_path)
        cache_path = self.get_cache_path(input_length, model_name, 'input_ids')
        torch.save(input_ids, cache_path)
        
    def load_kv_cache(self, input_length: int, model_name: str) -> Optional[Dict[str, torch.Tensor]]:
        kv_cache_path = self.get_cache_path(input_length, model_name, 'kv_cache')
        input_ids_path = self.get_cache_path(input_length, model_name, 'input_ids')
        if os.path.exists(kv_cache_path):
            return torch.load(kv_cache_path), torch.load(input_ids_path)
        return None, None
    
    def _set_model_kv_cache(self, model, kv_cache: Dict[str, torch.Tensor]):
        """Set model's past_key_values from cached KV"""
        past_key_values = []
        for i in range(len(kv_cache)):
            layer_cache = kv_cache[f"layer_{i}"]
            past_key_values.append((layer_cache[0], layer_cache[1]))
        model.past_key_values = tuple(past_key_values)


class PrintTokensStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # Print the last token generated
        last_token_id = input_ids[0, -1].item()
        token_str = self.tokenizer.decode([last_token_id], skip_special_tokens=True)
        print(token_str, end="", flush=True)
        
        # Continue generating until we reach the end
        return False

# Define the TimeTracker class to keep track of time per layer type
class TimeTracker:
    def __init__(self):
        self.layer_type_times = defaultdict(float)

time_tracker = TimeTracker()

# Hook functions to measure time in each layer
def hook_pre(module, input):
    module._start_time = time.time()


def hook_post_with_name(name):
    def hook_post(module, input, output):
        elapsed = time.time() - module._start_time
        time_tracker.layer_type_times[name] += elapsed
    return hook_post


# def hook_post(module, input, output):
#     elapsed = time.time() - module._start_time
#     layer_type = type(module).__name__
#     layer_name = module._get_name()
#     # import pdb; pdb.set_trace()
#     time_tracker.layer_type_times[layer_name] += elapsed



# Function to attach hooks to all layers
def attach_hooks_to_all_layers(model):
    for name, module in model.named_modules():
        # Only attach hooks to modules that perform computation
        # if len(list(module.children())) == 0:
        module.register_forward_pre_hook(hook_pre)
        module.register_forward_hook(hook_post_with_name(name))

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    load_in_8bit=args.enable_int8,
    device_map="cpu")

tokenizer = AutoTokenizer.from_pretrained(model_id) 
model.eval()

from transformers.models.llama.modeling_llama import LlamaAttention, LlamaSdpaAttention

# Function to replace LlamaSdpaAttention with LlamaAttention in the model
def replace_model_attention_layers_with_LlamaAttention(model):
    def replace_attention_layers(module):
        for name, child in module.named_children():
            if isinstance(child, LlamaSdpaAttention):
                # Create the new LlamaAttention layer
                new_child = LlamaAttention(child.config, child.layer_idx)
                
                # Transfer the state from the old SDPA attention to the new LlamaAttention
                new_child.load_state_dict(child.state_dict())
                
                # Ensure the new module has the same device and dtype as the original
                device = next(child.parameters()).device
                dtype = next(child.parameters()).dtype
                new_child = new_child.to(device=device, dtype=dtype)
                
                # Replace the attention layer in the module
                setattr(module, name, new_child)
            else:
                # Recursively apply the function to all children
                replace_attention_layers(child)

    # Apply the replacement
    replace_attention_layers(model)


def make_model_compatible(model):
    """
    Makes LLaMA-2 model layers compatible by padding weights to ensure output features
    are divisible by 16 * 32. This function modifies the model in-place.
    
    Args:
        model: The LLaMA-2 model to be made compatible
    """
    def pad_layer(layer):
        if not hasattr(layer, 'out_features'):
            return False
        
        if layer.out_features % (16 * 32) != 0:
            # import pdb; pdb.set_trace()
            original_out_features = layer.out_features
            # Calculate padding needed
            pad_size = (16 * 32) - (original_out_features % (16 * 32))
            new_out_features = original_out_features + pad_size
            
            # Create new padded weight
            new_weight = torch.zeros(
                (new_out_features, layer.in_features),
                dtype=layer.weight.dtype,
                device=layer.weight.device
            )
            new_weight[:original_out_features, :] = layer.weight
            
            # Update layer parameters
            layer.weight.data = new_weight
            layer.out_features = new_out_features
            
            return True
        return False

    def modify_next_layer(layer, prev_pad_size):
        # import pdb; pdb.set_trace()
        if not hasattr(layer, 'in_features'):
            return
        
        # Create new weight matrix with modified input features
        new_weight = torch.zeros(
            (layer.out_features, layer.in_features + prev_pad_size),
            dtype=layer.weight.dtype,
            device=layer.weight.device
        )
        new_weight[:, :layer.in_features] = layer.weight
        
        # Update layer parameters
        layer.weight.data = new_weight
        layer.in_features += prev_pad_size

    def process_model(module):
        prev_pad_size = 0
        modified_layers = []
        
        # First pass: pad layers
        for name, layer in module.named_children():
            # Recurse into submodules
            if len(list(layer.children())) > 0:
                process_model(layer)
            else:
                if pad_layer(layer):
                    modified_layers.append(name)
        
        # Second pass: modify input features of layers following modified layers
        prev_layer_name = None
        for name, layer in module.named_children():
            if prev_layer_name in modified_layers and 'down_proj' in name:
                prev_layer = module._modules[prev_layer_name]
                prev_pad_size = prev_layer.out_features - layer.in_features
                modify_next_layer(layer, prev_pad_size)
            prev_layer_name = name

    process_model(model)
    return model

model = make_model_compatible(model)

# import pdb; pdb.set_trace()

def replace_model_attention_layers(model):
    def replace_attention_layers(module):
        for name, child in module.named_children():
            if isinstance(child, LlamaSdpaAttention):
                new_child = CustomLlamaAttention(child.config, child.layer_idx, args.use_custom_k, args.use_custom_v, args.k_pruning, args.v_pruning, args.attention_mode)
                original_state_dict = child.state_dict()
                new_state_dict = new_child.state_dict()
                filtered_state_dict = {k: v for k, v in original_state_dict.items() if k in new_state_dict}
                new_child.load_state_dict(filtered_state_dict)
                new_child.load_computers(args.context_length)
                 # Ensure the new module has the same dtype and device
                device = next(child.parameters()).device
                dtype = next(child.parameters()).dtype
                new_child = new_child.to(device=device, dtype=dtype)
                setattr(module, name, new_child)
            else:
                replace_attention_layers(child)
    
    # Replace the linear layers in the copied model
    replace_attention_layers(model)
    
    return model


if args.enable_custom_attention:
    model = replace_model_attention_layers(model)
    # replace_model_attention_layers_with_LlamaAttention(model)


def replace_linear_layers_with_wrapper(model, quiet=True, shallow=False):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
            wrapped_linear = custom_layer_class.from_linear(module, shallow=shallow, num_threads=num_threads)
            setattr(model, name, wrapped_linear)
        else:
            replace_linear_layers_with_wrapper(module, quiet=quiet, shallow=shallow)
    return model

def get_safetensors_files(directory):
    return sorted([f for f in os.listdir(directory) if f.endswith('.safetensors')])

def load_split_safetensors(directory, files):
    weights = {}
    for file in files:
        file_path = os.path.join(directory, file)
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)
    return weights

if not(os.path.exists(saved_model_path)):
    load_presaved_model = False

if enable_custom_layer:
    model = replace_linear_layers_with_wrapper(model, quiet=True, shallow=load_presaved_model)

def split_weights_into_sets_for_sparse_linear(model, num_sets):
    for name, module in model.named_children():
        if enable_custom_layer and (isinstance(module, SparseLinear) or isinstance(module, QuantizedSparseLinear) or isinstance(module, AvxSparseLinear)) and name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
            module.split_weights_to_sets(num_sets)
        else:
            split_weights_into_sets_for_sparse_linear(module, num_sets)
    return model

if load_presaved_model:
    safetensors_files = get_safetensors_files(saved_model_path)
    saved_weights = load_split_safetensors(saved_model_path, safetensors_files)
    for name, param in model.named_parameters():
        if name in saved_weights:
            saved_tensor = saved_weights[name]
            if param.shape != saved_tensor.shape:
                print(f"Resizing parameter {name} from {param.shape} to {saved_tensor.shape}")
                new_param = torch.nn.Parameter(saved_tensor.clone(), requires_grad=False)
                model.get_submodule(name.rsplit('.', 1)[0]).register_parameter(name.rsplit('.', 1)[1], new_param)
            else:
                param.data.copy_(saved_tensor)
        else:
            print(f"Warning: {name} not present in saved weights")

    model.load_state_dict(model.state_dict())
else:
    if enable_custom_layer and (mode == 'sparse' or mode == 'avx_sparse' or mode == 'sparseq'):
        model = split_weights_into_sets_for_sparse_linear(model, num_threads)
        if not(os.path.exists(saved_model_path)):
            model.save_pretrained(f'{saved_model_path}')
            print(f"Saved processed model at path: {saved_model_path}")


# import pdb; pdb.set_trace()
# Attach hooks to all layers
# attach_hooks_to_all_layers(model)

model.config.eos_token_id = None

tok_latencies = []

# org_forward = model.forward
# def new_forward(self, *args, **kwargs):
#     t0 = time.time()
#     ret = org_forward.__func__(self, *args, **kwargs)
#     t1 = time.time()
#     tok_latencies.append(t1 - t0)
#     return ret

# import pdb; pdb.set_trace()
# new_forward.__signature__ = inspect.signature(model.forward)
# model.forward = types.MethodType(new_forward, model)

org_forward = model.forward
if isinstance(org_forward, functools.partial):
    # For 8-bit quantized models
    original_func = org_forward.func
    original_args = org_forward.args
    original_keywords = org_forward.keywords

    def new_forward(*args, **kwargs):
        t0 = time.time()
        ret = original_func(*args, **kwargs)
        t1 = time.time()
        tok_latencies.append(t1 - t0)
        return ret

    model.forward = functools.partial(new_forward, *original_args, **original_keywords)
else:
    # For standard models
    def new_forward(self, *args, **kwargs):
        t0 = time.time()
        ret = org_forward.__func__(self, *args, **kwargs)
        t1 = time.time()
        tok_latencies.append(t1 - t0)
        return ret

    new_forward.__signature__ = inspect.signature(model.forward)
    model.forward = types.MethodType(new_forward, model)


########################################
print(model)

# Read big_prompt.txt and use it as the first prompt
first_prompt = ""
with open("big_prompt.txt", "r") as f:
    first_prompt = f.read()

default_prompts = [
    first_prompt,
    "Alan Turing theorized that computers would one day become",
    "The quick brown fox jumps over the lazy dog",
    "In the beginning God created the heavens and the earth",
    "To be or not to be, that is the question",
    "It was the best of times, it was the worst of times",
    "I think, therefore I am",
    "All happy families are alike; each unhappy family is unhappy in its own way",
    "Call me Ishmael",
    "Once upon a time, in a land far, far away",
    "The only thing we have to fear is fear itself"
]

if args.batch_size > len(default_prompts):
    num_repeats = (args.batch_size + len(default_prompts) - 1) // len(default_prompts)
    prompts = (default_prompts * num_repeats)[:args.batch_size]
else:
    prompts = default_prompts[:args.batch_size]


tokenizer.pad_token = 'and'
input_ids = tokenizer(prompts, padding=True, truncation=True, max_length=args.context_length, return_tensors="pt").input_ids

if input_ids.shape[1] < args.context_length:
    padding = torch.full((input_ids.shape[0], args.context_length - input_ids.shape[1]), tokenizer.pad_token_id, dtype=input_ids.dtype)
    input_ids = torch.cat([input_ids, padding], dim=1)
elif input_ids.shape[1] > args.context_length:
    input_ids = input_ids[:, :args.context_length]

print('Checking if prefill is cached')
if args.use_cached_prefill:
    kv_cache_manager = KVCacheManager()
    past_key_values, new_input_ids = kv_cache_manager.load_kv_cache(args.context_length, model_id.split('/')[-1])
    # import pdb; pdb.set_trace()
    # if past_key_values is None or (args.enable_custom_attention and (model.model.layers[0].self_attn.key_computer is None or model.model.layers[0].self_attn.value_computer is None)):
    if past_key_values is None or (args.enable_custom_attention and (model.model.layers[0].self_attn.key_computer is None or model.model.layers[0].self_attn.value_computer is None)):
        print('Prefill not cached, caching now...')
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        generated_input_ids = outputs.logits.argmax(dim=-1)[:,-1:]
            # import pdb; pdb.set_trace()
        new_input_ids = torch.cat([input_ids, generated_input_ids], dim=1)
        kv_cache_manager.save_kv_cache(past_key_values, new_input_ids, args.context_length, model_id.split('/')[-1])


# import pdb; pdb.set_trace()
with torch.no_grad():
    print('Warming Up...')
    for i in range(args.num_warmup):  # Warmup
        with torch.no_grad():
            # import pdb; pdb.set_trace()
            model.generate(new_input_ids if args.use_cached_prefill else input_ids, do_sample=False, top_p=None, num_beams=1, max_new_tokens=args.num_generated_tokens, eos_token_id=None, past_key_values=past_key_values if args.use_cached_prefill else None)


    prefill_times = []
    decode_times = []
    total_times = []
    layer_type_times_list = []  # To store layer type times per model call
    print('Starting generation...')
    tok_latencies = []  # Reset to discard warmup latencies

    for name, module in model.named_modules():
        if isinstance(module, CustomLlamaAttention):
            module.print_timing_stats()
            module.reset_timing_stats()
    import pdb; pdb.set_trace()
    for i in range(args.num_iterations):
        # Reset time tracking
        for name, module in model.named_modules():
            if isinstance(module, CustomLlamaAttention):
                module.reset_cache()
                
        time_tracker.layer_type_times = defaultdict(float)
        t0 = time.time()
        
        # import pdb; pdb.set_trace()
        output_ids = model.generate(
            new_input_ids if args.use_cached_prefill else input_ids,
            do_sample=False,
            top_p=None,
            num_beams=1,
            max_new_tokens=args.num_generated_tokens,
            eos_token_id=None,
            past_key_values=past_key_values if args.use_cached_prefill else None,
        )
        t1 = time.time()
        total_times.append(t1 - t0)
        # Deep copy the layer times for this iteration
        layer_type_times_list.append(dict(time_tracker.layer_type_times))

print(f'Number of output tokens: {output_ids.size()}')
output = tokenizer.batch_decode(output_ids.cpu())

print(f'Total Times: {total_times}, Median: {median(total_times)}')

with open("total_time.txt", "w") as f:
    f.write(f'{median(total_times)}')

with open("layer_times.txt", "w") as f:
    for layer_times in layer_type_times_list:
        f.write(f'{layer_times}\n')

# Aggregate times per layer type
aggregate_layer_times = defaultdict(float)
for layer_times in layer_type_times_list:
    for layer_type, time_spent in layer_times.items():
        aggregate_layer_times[layer_type] += time_spent

# Calculate percentages
total_time_spent_in_layers = sum(aggregate_layer_times.values())
print("\nTime spent per layer type:")
for layer_type, time_spent in aggregate_layer_times.items():
    percentage = (time_spent / total_time_spent_in_layers) * 100 if total_time_spent_in_layers > 0 else 0
    print(f"{layer_type}: {time_spent:.6f} seconds ({percentage:.2f}%)")

prefill_times = tok_latencies[::args.num_generated_tokens]
decode_times = [el for i, el in enumerate(tok_latencies) if (i % args.num_generated_tokens) != 0]

# Print outputs for all prompts
# for idx, out in enumerate(output):
#     print(f"Prompt {idx+1}: {prompts[idx]}")
#     print(f"Output {idx+1}: {out}")
#     print()

print(f'Prefill Times: {prefill_times}, Median: {median(prefill_times)}')
with open("prefill_time.txt", "w") as f:
    f.write(f'{median(prefill_times)}')

print(f'Decode Times: {decode_times}, Median: {median(decode_times)}')
with open("decode_time.txt", "w") as f:
    f.write(f'{median(decode_times)}')

# Print accumulated timing statistics for all attention layers
# print("\nAttention Layer Timing Statistics:")
# for name, module in model.named_modules():
#     if isinstance(module, CustomLlamaAttention):
#         module.print_timing_stats()
