import json
import torch
from safetensors.torch import save_file
from safetensors import safe_open

NUM_G_LAYERS = 9
NUM_MEM_BLOCKS = 2


def preprocess_func(data):
    new_data = {}
    transformer_weight_data = {}
    for key, weight_tensor in data.items():
        if "model.blocks" in key:
            if "linear_fc1" in key and "lora_A" not in key:
                up_proj_weight_tensor, gate_proj_weight_tensor = weight_tensor.chunk(2, dim=0)
                transformer_weight_data[key.replace("linear_fc1", "linear_fc1_up")] = up_proj_weight_tensor.clone()
                transformer_weight_data[key.replace("linear_fc1", "linear_fc1_gate")] = gate_proj_weight_tensor.clone()
            elif "lora_A" in key:
                transformer_weight_data[key.replace("linear_fc1", "linear_fc1_up")] = weight_tensor.clone()
                transformer_weight_data[key.replace("linear_fc1", "linear_fc1_gate")] = weight_tensor.clone()
            else:
                transformer_weight_data[key] = weight_tensor.clone()
        else:
            new_data[key] = weight_tensor.clone()

    for num_layer in range(NUM_G_LAYERS):
        if num_layer % NUM_MEM_BLOCKS == 0:
            id = 0
        else:
            id = 1
        cur_data = {k: v for k, v in transformer_weight_data.items() if f"model.blocks.{id}" in k}
        for k, v in cur_data.items():
            if "lora" not in k:
                new_data[k.replace(f"model.blocks.{id}", f"model.blocks.{num_layer}")] = v.clone()
            elif f"_lora_B_list.{num_layer}" in k: # lora
                lora_B = v.clone()
                lora_A = cur_data[k.replace(f"lora_B", "lora_A")]
                lora_BA = torch.matmul(lora_B, lora_A)
                new_data[k.replace(f"model.blocks.{id}", f"model.blocks.{num_layer}").replace(f"_lora_B_list.{num_layer}", "")] += lora_BA.clone()
    return new_data

def load_safetensors(filename):
    tensors = {}
    with safe_open(filename, framework="pt", device=0) as f:
        metadata = f.metadata()
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    return tensors, metadata

def save_safetensors(data, metadata, filename):
    save_file(data, filename, metadata=metadata)

new_weight_map = {}

data, metadata = load_safetensors("./Zamba2-2.7B/model-00001-of-00002.safetensors")
new_data = preprocess_func(data)
save_safetensors(new_data, metadata, "./Zamba2-2.7B/model-00001-of-00002.safetensors")
for key in new_data.keys():
    new_weight_map[key] = "model-00001-of-00002.safetensors"

data, metadata = load_safetensors("./Zamba2-2.7B/model-00002-of-00002.safetensors")
new_data = preprocess_func(data)
save_safetensors(new_data, metadata, "./Zamba2-2.7B/model-00002-of-00002.safetensors")
for key in new_data.keys():
    new_weight_map[key] = "model-00002-of-00002.safetensors"

with open("./Zamba2-2.7B/model.safetensors.index.json", "r") as f:
    data = json.load(f)
data["weight_map"] = new_weight_map
with open("./Zamba2-2.7B/model.safetensors.index.json", "w") as f:
    json.dump(data, f, indent=4)
