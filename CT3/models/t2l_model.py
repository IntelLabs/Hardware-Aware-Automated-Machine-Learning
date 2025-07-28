from typing import Optional

import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer
)
from hyper_llm_modulator.hyper_modulator import load_hypermod_checkpoint
from hyper_llm_modulator.utils.utils import embed_texts
from hyper_llm_modulator.utils import get_layers

from t2l_config import T2LConfig


def adjust_lora_state_dict_keys(lora_state_dict):
    """
    Adjusts LoRA state dict keys by inserting '.default' before '.weight' if not already present.
    """
    adjusted_state_dict = {}
    for key, value in lora_state_dict.items():
        if key.endswith('.weight') and not key.endswith('.default.weight'):
            new_key = key[:-len('.weight')] + '.default.weight'
        else:
            new_key = key
        adjusted_state_dict[new_key] = value
    return adjusted_state_dict


class T2LModel(torch.nn.Module):
    def __init__(self, hypernetwork_path: Optional[str] = None, model: PreTrainedModel = None, tokenizer: PreTrainedTokenizer = None) -> None:
        super(T2LModel, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        _, self.hypermod, self.model, self.tokenizer, self.emb_model, self.emb_tokenizer, self.task_desc_format_fn, self.pooling_fn = (
            load_hypermod_checkpoint(hypernetwork_path, self.device)
        )
        if model is not None:
            self.model = model
        if tokenizer is not None:
            self.tokenizer = tokenizer
        self.layer_indices = torch.tensor(range(len(get_layers(self.model))), dtype=torch.long, device=self.device)
        
    def _into_prompt(self, query) -> str:
        tokenizer = self.tokenizer
        assert hasattr(tokenizer, "apply_chat_template")
        prompt: str = tokenizer.apply_chat_template(
            conversation=[
                {
                    "role": "system",
                    "content": "",
                },
                {
                    "role": "user",
                    "content": query,
                },
            ],
            chat_template=self.tokenizer.chat_template,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt

    def forward(self, *args, **kwargs):
        return self._forward_or_generate(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self._forward_or_generate(*args, **kwargs, generate=True)

    def _forward_or_generate(self, *args, generate=False, **kwargs):
        if 'input_ids' in kwargs:
            input_ids = kwargs['input_ids']
        elif args and len(args) > 0:
            input_ids = args[0]
        else:
            raise ValueError("Neither 'input_ids' in kwargs nor a non-empty args provided")

        assert input_ids.size(0) == 1, "Only supports batch size 1."

        query = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]
        prompt = self._into_prompt(query)
        task_emb = embed_texts([prompt], self.emb_model, self.emb_tokenizer, self.task_desc_format_fn, self.pooling_fn, self.device)
        encoder_out = self.hypermod.task_encoder(task_emb)
        encoded_task_emb = encoder_out["encoded_task_emb"].detach()
        lora_sd = self.hypermod.gen_lora(self.layer_indices, encoded_task_emb)
        lora_sd = adjust_lora_state_dict_keys(lora_sd)
        self.model.load_state_dict(lora_sd, strict=False)

        # Perform the forward pass or generate
        if generate:
            output = self.model.generate(*args, **kwargs)
        else:
            output = self.model(*args, **kwargs)
        
        return output
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def load_t2l_model(t2l_config: T2LConfig, model: PreTrainedModel = None, tokenizer: PreTrainedTokenizer = None) -> T2LModel:
    t2l_model_instance = T2LModel(t2l_config.hypernetwork_path, model=model, tokenizer=tokenizer)
    return t2l_model_instance
