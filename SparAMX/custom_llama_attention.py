import math
from typing import List, Optional, Tuple, Union
import os.path

import torch
import torch.nn.functional as F
from torch import nn
import time


from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.cache_utils import Cache
from layer.sparse_linear import SparseLinear
from layer.dense_linear import DenseLinear
# from layer.avx_sparse_linear import AvxSparseLinear


# SAVED_FILES_DIR = "saved_kv_state_files_per_layer_amx_new/per_head"
SAVED_FILES_DIR = "saved_kv_state_files_per_layer_amx_right_keys/per_layer"
TILE_ROW_SIZE = 32

def sparsify(vals, prune_percentage):
    if prune_percentage == 0:
        return vals
    k = int(vals.numel() * (prune_percentage / 100))
    original_dtype = vals.dtype
    output_float = vals.float()
    flat_output = output_float.reshape(-1)
    threshold = torch.kthvalue(flat_output.abs(), k).values
    mask = (output_float.abs() >= threshold)
    # mask = 0
    pruned_output = (output_float * mask).to(original_dtype)

    return pruned_output

def sparsify_per_head(vals, prune_percentage):
    # Loop through the last 2 dimensions only
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            vals[i, j] = sparsify(vals[i, j], prune_percentage)
    
    return vals

def saved_file_prefix(layer_id, pruning, ctx_length, kernel):
    return f"{SAVED_FILES_DIR}/kernel_{kernel}_ctx_{ctx_length}_layer_{layer_id}_pruning_{pruning}"

class CustomLlamaAttention(LlamaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None, use_custom_k: bool = False, use_custom_v: bool = False, k_pruning: int = 50, v_pruning: int = 50, kernel_type = 'sparse', ctx_length: int = 0):
        super().__init__(config, layer_idx)
        self.key_computer = None
        self.value_computer = None
        self.use_custom_k = use_custom_k
        self.use_custom_v = use_custom_v
        self.k_pruning = k_pruning
        self.v_pruning = v_pruning
        self.cached_key_states = torch.empty(0)
        self.cached_value_states = torch.empty(0)
        if kernel_type == 'sparse':
            self.kernel = SparseLinear
        elif kernel_type == 'dense':
            self.kernel = DenseLinear
        else:
            raise ValueError("Invalid kernel type")
        # Add timing accumulators
        self.reset_timing_stats()
        print(f"Using custom k: {self.use_custom_k}, using custom v: {self.use_custom_v}, k pruning: {self.k_pruning}, v pruning: {self.v_pruning}, with kernel: {kernel_type}")


    def reset_timing_stats(self):
        """Reset accumulated timing statistics"""
        self.timing_stats = {
            'time_1': 0.0,
            'time_2': 0.0,
            'time_3': {'total': 0.0, '3_1': 0.0, '3_2': 0.0, '3_3': 0.0, '3_4': 0.0},
            'time_4': 0.0,
            'time_5': 0.0,
            'attn_weights': 0.0,
            'attn_output': 0.0,
            'total': 0.0,
            'call_count': 0
        }

    def load_computers(self, ctx_length):
        # Disabled for now. # TODO: Enable this.
        return
        if self.use_custom_k and os.path.exists(f"{saved_file_prefix(self.layer_idx, self.k_pruning, ctx_length, self.kernel)}_k.pt"):
            self.key_computer = self.kernel.load_from_file(f"{saved_file_prefix(self.layer_idx, self.k_pruning, ctx_length, self.kernel)}_k.pt")
        if self.use_custom_v and os.path.exists(f"{saved_file_prefix(self.layer_idx, self.v_pruning, ctx_length, self.kernel)}_v.pt"):
            self.value_computer = self.kernel.load_from_file(f"{saved_file_prefix(self.layer_idx, self.v_pruning, ctx_length, self.kernel)}_v.pt")


    def reset_computers(self):
        self.key_computer = None
        self.value_computer = None

    def prepare_extra_keys(self, extra_keys):
        res = extra_keys.clone()
        res = res.view(res.shape[0], res.shape[1], res.shape[2], -1, 2).transpose(2,3).reshape(extra_keys.shape[0], extra_keys.shape[1], extra_keys.shape[3], extra_keys.shape[2]).to(torch.bfloat16).contiguous()
        return res
    
    def prepare_extra_values(self, extra_values):
        # Pad the values to have full tiles
        # A full tile has TILE_SIZE rows
        # import pdb; pdb.set_trace()
        res = extra_values.clone() # TODO: Can we optimize this?
        if extra_values.size(2) % TILE_ROW_SIZE != 0:
            res = F.pad(extra_values, (0, 0, 0, TILE_ROW_SIZE - extra_values.size(2) % TILE_ROW_SIZE))
        res = res.transpose(2, 3)
        # reorder elements so that each word of the sequence is in the same row
        batch, heads, dim, seq_len = res.shape
            
        # First reshape maintaining the original sequence dimension
        res = res.view(batch, heads, dim, seq_len // 2, 2)
        
        # Transpose the sequence and dim//2 dimensions
        res = res.permute(0, 1, 3, 2, 4)
        
        # Final reshape
        res = res.reshape(batch, heads, seq_len, dim)
        
        # Convert to bfloat16 and ensure contiguous memory layout
        res = res.to(torch.bfloat16).contiguous()
        return res


    def update_cache(self, key_states, value_states):
        if key_states.shape[2] > 1:
            return key_states, value_states

        # breakpoint()
        prepared_key_states = key_states
        # Append key_states and value_states to current cache
        if self.cached_key_states.numel() == 0:
            self.cached_key_states = prepared_key_states
            # self.cached_key_states = prepared_key_states
            self.cached_value_states = value_states
        else:
            # breakpoint()
            self.cached_key_states = torch.cat((self.cached_key_states, prepared_key_states), dim=2)
            self.cached_value_states = torch.cat((self.cached_value_states, value_states), dim=2)
        
        return self.cached_key_states, self.cached_value_states
    
    def reset_cache(self):
        self.cached_key_states = torch.empty(0)
        self.cached_value_states = torch.empty(0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        # if self.layer_idx == 0:
        #     breakpoint()
        # all_function_start = time.time()
        bsz, q_len, _ = hidden_states.size()

        # past_key_value.key_cache = []
        # past_key_value.value_cache = []

        # time_1_start = time.time()
        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        # import pdb; pdb.set_trace()
        # time_1_end = time.time()

        # time_2_start = time.time()
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        # time_2_end = time.time()

        # time_3_start = time.time()
        # time_3_1_start = time.time()
        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        # time_3_1_end = time.time()
        # time_3_2_start = time.time()
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        # time_3_2_end = time.time()


        if hidden_states.shape[1] > 1:
            # We're in the prefill stage.
            key_states = sparsify(key_states, self.k_pruning)
            value_states = sparsify(value_states, self.v_pruning)
        else:
            # We're in the decode stage.
            if self.use_custom_k or self.use_custom_v:
                key_states_custom, value_states_custom = self.update_cache(key_states, value_states)
            if self.use_custom_k:
                key_states_custom = repeat_kv(key_states_custom, self.num_key_value_groups)
            if self.use_custom_v:
                value_states_custom = repeat_kv(value_states_custom, self.num_key_value_groups)
    
        # time_3_3_start = time.time()
        # import pdb; pdb.set_trace()
        if not(self.use_custom_k or self.use_custom_v) or hidden_states.shape[1] > 1:
            if past_key_value is not None:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # time_3_3_end = time.time()

        # print(f"Current key_states: {key_states.shape}, current value_states: {value_states.shape}")
        # time_3_4_start = time.time()

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
        # time_3_4_end = time.time()
        # time_3_end = time.time()
        # print("Time taken for time 3_1: ", time_3_1_end - time_3_1_start)
        # print("Time taken for time 3_2: ", time_3_2_end - time_3_2_start)
        # print("Time taken for time 3_3: ", time_3_3_end - time_3_3_start)
        # print("Time taken for time 3_4: ", time_3_4_end - time_3_4_start)

        # import pdb; pdb.set_trace()
        # attn_weights_start = time.time()
        if self.use_custom_k:
            # if self.layer_idx == 0:
            #     breakpoint()
            if self.key_computer is None:
                assert(key_states.shape[2] > 1)
                # key_states = sparsify(key_states, self.k_pruning)
                self.key_computer = self.kernel.from_batched_weights(key_states.contiguous())
                self.key_computer.save_state(f"{saved_file_prefix(self.layer_idx, self.k_pruning, q_len, self.kernel)}_k.pt")

                assert(self.cached_key_states.numel() == 0)
                attn_weights =  self.key_computer.matmul(query_states.contiguous()) / math.sqrt(self.head_dim)
            else:
                # assert(self.cached_key_states.numel() > 0) # Can't do that assert because the first time it would be empty.
                # attn_weights = self.key_computer.matmul(query_states.contiguous(), key_states.transpose(2,3), is_key=True) / math.sqrt(self.head_dim)
                if hidden_states.shape[1] > 1:
                    import pdb; pdb.set_trace()
                attn_weights = self.key_computer.matmul(query_states.contiguous(), self.prepare_extra_keys(key_states_custom), is_key=True) / math.sqrt(self.head_dim)
        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # if self.layer_idx == 0:
        #     breakpoint()
        # attn_weights_end = time.time()
        # attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # time_4_start = time.time()
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # time_4_end = time.time()
        # custom_attn_output = custom_attention_output(attn_weights, value_states)
        # import pdb; pdb.set_trace()
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # attn_otuput_start = time.time()
        if self.use_custom_v:
            if self.value_computer is None:
                assert(value_states.shape[2] > 1)
                # value_states = sparsify(value_states, self.v_pruning)
                self.value_computer = self.kernel.from_batched_weights(value_states.contiguous().transpose(2, 3))
                self.value_computer.save_state(f"{saved_file_prefix(self.layer_idx, self.v_pruning, q_len, self.kernel)}_v.pt")

                assert(self.cached_value_states.numel() == 0)
                attn_output = self.value_computer.matmul(attn_weights.contiguous())
            else:
                # Pad attn_weights to be multiple of TILE_ROW_SIZE on last dimension
                original_size = attn_weights.size(-1)
                padding_size = (TILE_ROW_SIZE - original_size % TILE_ROW_SIZE) % TILE_ROW_SIZE
                if padding_size > 0:
                    padded_attn_weights = F.pad(attn_weights, (0, padding_size))
                else:
                    padded_attn_weights = attn_weights
                    
                attn_output = self.value_computer.matmul(padded_attn_weights.contiguous(), self.prepare_extra_values(value_states_custom), is_key=False)
        else:
            # import pdb; pdb.set_trace()
            # attn_output = torch.matmul(attn_weights, value_states_2[:, :, :attn_weights.size(-1), :])
            attn_output = torch.matmul(attn_weights, value_states)
            
        # attn_output_end = time.time()
        attn_output = attn_output.to(torch.bfloat16)
        # attn_output = torch.matmul(attn_weights, value_states[:, :, :attn_weights.size(-1), :])
        # import pdb; pdb.set_trace()

        # if self.layer_idx == 0:
        #     import pdb; pdb.set_trace()

        # time_5_start = time.time()
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        # time_5_end = time.time()
        # all_function_end = time.time()
        # print(f"Time taken for attention weights: {attn_weights_end - attn_weights_start}, time taken for attn_output: {attn_output_end - attn_otuput_start}, total time taken: {all_function_end - all_function_start}")
        # print(f"Time taken for time 1: {time_1_end - time_1_start}, time taken for time 2: {time_2_end - time_2_start}, time taken for time 3: {time_3_end - time_3_start}, time taken for time 4: {time_4_end - time_4_start}, time taken for time 5: {time_5_end - time_5_start}")
        # Instead of printing, accumulate times
        # self.timing_stats['time_1'] += time_1_end - time_1_start
        # self.timing_stats['time_2'] += time_2_end - time_2_start
        # self.timing_stats['time_3']['total'] += time_3_end - time_3_start
        # self.timing_stats['time_3']['3_1'] += time_3_1_end - time_3_1_start
        # self.timing_stats['time_3']['3_2'] += time_3_2_end - time_3_2_start
        # self.timing_stats['time_3']['3_3'] += time_3_3_end - time_3_3_start
        # self.timing_stats['time_3']['3_4'] += time_3_4_end - time_3_4_start
        # self.timing_stats['time_4'] += time_4_end - time_4_start
        # self.timing_stats['time_5'] += time_5_end - time_5_start
        # self.timing_stats['attn_weights'] += attn_weights_end - attn_weights_start
        # self.timing_stats['attn_output'] += attn_output_end - attn_otuput_start
        # self.timing_stats['total'] += all_function_end - all_function_start
        # self.timing_stats['call_count'] += 1

        # if self.layer_idx % 10 == 0:
        #     print(f"Layer {self.layer_idx}: {attn_output}")
        #     import pdb; pdb.set_trace()
        return attn_output, attn_weights, past_key_value

    def print_timing_stats(self):
        """Print accumulated timing statistics"""
        if self.timing_stats['call_count'] == 0:
            return
            
        print(f"\nTiming statistics for attention layer {self.layer_idx} (averaged over {self.timing_stats['call_count']} calls):")
        for key, value in self.timing_stats.items():
            if key == 'time_3':
                print(f"Time 3 breakdown:")
                for subkey, subvalue in value.items():
                    avg = subvalue / self.timing_stats['call_count']
                    print(f"  - {subkey}: {avg:.6f}s")
            elif key != 'call_count':
                avg = value / self.timing_stats['call_count']
                print(f"{key}: {avg:.6f}s")
