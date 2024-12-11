import json
import os
import random

import torch


ADAPTER_NAME = "default"


class SharedR:
    def __init__(self, r_values):
        self.r_values = r_values
        self.current_r = random.choice(r_values)

    def __call__(self):
        return self.current_r

    def update_r(self):
        self.current_r = random.choice(self.r_values)


class ElasticLoRALinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, shared_r, is_lora_A, bias=False, is_group_head=False):
        super(ElasticLoRALinear, self).__init__(in_features, out_features, bias)
        self.shared_r = shared_r
        self.is_lora_A = is_lora_A
        self.is_group_head = is_group_head
        self.eval()
        self._weight = self.weight
        self._bias = self.bias

    def init_weights(self, orig):
        assert isinstance(orig, torch.nn.Linear)
        self._weight.data.copy_(orig.weight.data)
        if orig.bias is not None:
            self._bias.data.copy_(orig.bias.data)

    @torch.no_grad
    def active_sub_adapter(self):
        if self.is_group_head:
            self.shared_r.update_r()
        return self.shared_r()

    @property
    def masked_weight(self):
        active_r = self.active_sub_adapter()
        if self.is_lora_A:
            return self._weight[:active_r, :]
        else:
            return self._weight[:, :active_r]

    @property
    def weight(self):
        return self.masked_weight if self.training else self._weight

    @property
    def bias(self):
        if self._bias is not None and self.is_lora_A and self.training:
            return self._bias[:self.shared_r()]
        return self._bias

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        state_dict = super(ElasticLoRALinear, self).state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)
        # Remove _weight and _bias from state_dict
        state_dict.pop(prefix + '_weight', None)
        state_dict.pop(prefix + '_bias', None)
        return state_dict

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Load the state_dict as usual
        super(ElasticLoRALinear, self).load_state_dict(state_dict, strict, assign)
        # Ensure _weight and _bias are correctly set
        self._weight = self.weight
        self._bias = self.bias


def make_lora_elastic(model, r_values, target_modules, share_rank_within_layer=True, config_save_dir=None):

    def replace_module(layer, named_modules, module_name, shared_r, is_lora_A, is_group_head=False):
            module = named_modules.get(module_name)
            assert module is not None and isinstance(module, torch.nn.Linear)
            elastic_lora_linear = ElasticLoRALinear(module.in_features, module.out_features, shared_r, is_lora_A, bias=module.bias is not None, is_group_head=is_group_head)
            elastic_lora_linear.init_weights(module)
            elastic_lora_linear.train()
            parent_module = layer
            *name_parts, last_part = module_name.split('.')
            for part in name_parts:
                parent_module = getattr(parent_module, part)
            setattr(parent_module, last_part, elastic_lora_linear)

    elastic_adapter_config = []
    for idx, layer in enumerate(model.base_model.model.model.layers):
        named_modules = dict(layer.named_modules())
        shared_r = SharedR(r_values)
        is_group_head = True
        elastic_adapter_config_item = {"target": [], "search_space": shared_r.r_values}
        for name, module in named_modules.items():
            if isinstance(module, torch.nn.Linear) and "lora_A" in name and any(module_name in name for module_name in target_modules):
                replace_module(layer, named_modules, name, shared_r, is_lora_A=True, is_group_head=is_group_head)
                # REMOVE ADAPTER NAME (from PEFT)
                module_sd_name = f"base_model.model.model.layers.{idx}.{name}".replace(f".{ADAPTER_NAME}", "")
                if share_rank_within_layer:
                    is_group_head = False
                    elastic_adapter_config_item["target"].append(module_sd_name)
                else:
                    elastic_adapter_config.append({"target": [module_sd_name], "search_space": shared_r.r_values})
                lora_B_name = name.replace('lora_A', 'lora_B')
                replace_module(layer, named_modules, lora_B_name, shared_r, is_lora_A=False)
        
        if share_rank_within_layer:
            elastic_adapter_config.append(elastic_adapter_config_item)

    if config_save_dir is not None:
        os.makedirs(config_save_dir, exist_ok=True)
        with open(os.path.join(config_save_dir, "elastic_adapter_config.json"), "w") as f:
            json.dump(elastic_adapter_config, f, indent=4)
