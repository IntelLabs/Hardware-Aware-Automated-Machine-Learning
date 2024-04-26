## NNCF Config of Shears

To enable the elastic adapter in NLS training, we employ the [BootstrapNAS](https://github.com/openvinotoolkit/nncf/tree/develop/nncf/experimental/torch/nas/bootstrapNAS) feature within [OpenVINO™ NNCF](https://github.com/openvinotoolkit/nncf), offering a range of compression algorithms tailored for optimizing neural networks.
Here is an instruction of NNCF configuration for Shears, aimed at eliminating any user doubts regarding the config and clarifying which parts are relevant to Shears.

Some explanations:

- `input_info` is used to create nncf network in Shears.
- Shears employs the `progressive_shrinking` algorithm of bootstrapNAS, details can be found in [BootstrapNAS.md](https://github.com/openvinotoolkit/nncf/blob/develop/nncf/experimental/torch/nas/bootstrapNAS/BootstrapNAS.md). 
Actually, Shears only adopts the simplest `progressive_shrinking` feature without utilizing its more intricate and advanced strategies such as multi-stage training. 
We will explore more complex training strategies of `progressive_shrinking` in the future.
- `frozen_layers_allowed` should be set to `true`, because Shears freezes the base model.
- `width` means the hidden size of the weight matrix, more precisely, it represents the low-rank size of the LoRA adapter in Shears.
- `num_bn_adaptation_samples` should be set to 0 (default is 2000), as we don't need batch norm adaption.

### Elastic low-rank

In Shears solution, the design of the low-rank search space is crucial, including the allocation of dependency groups and the design of the search space for each group. 
In our existing configurations, such as the LLaMA model, we adopt the grouping `[[Q, K, V], [Up], [Down]]` for each llama layer, with each group's search space being `[32, 24, 16]`, i.e.,

- `[Q, K, V]`: `[32, 24, 16]`
- `[Up]`: `[32, 24, 16]`
- `[Down]`: `[32, 24, 16]`

```json
"width": {
    "overwrite_groups": [
        [
            "{re}PeftModelForCausalLM/LoraModel[base_model]/LlamaForCausalLM[model]/LlamaModel[model]/ModuleList[layers]/LlamaDecoderLayer[{*}]/LlamaAttention[self_attn]/Linear[q_proj]/ModuleDict[lora_A]/NNCFLinear[default]/linear_0",
            "{re}PeftModelForCausalLM/LoraModel[base_model]/LlamaForCausalLM[model]/LlamaModel[model]/ModuleList[layers]/LlamaDecoderLayer[{*}]/LlamaAttention[self_attn]/Linear[k_proj]/ModuleDict[lora_A]/NNCFLinear[default]/linear_0",
            "{re}PeftModelForCausalLM/LoraModel[base_model]/LlamaForCausalLM[model]/LlamaModel[model]/ModuleList[layers]/LlamaDecoderLayer[{*}]/LlamaAttention[self_attn]/Linear[v_proj]/ModuleDict[lora_A]/NNCFLinear[default]/linear_0"
        ],
        [
            "{re}PeftModelForCausalLM/LoraModel[base_model]/LlamaForCausalLM[model]/LlamaModel[model]/ModuleList[layers]/LlamaDecoderLayer[{*}]/LlamaMLP[mlp]/Linear[up_proj]/ModuleDict[lora_A]/NNCFLinear[default]/linear_0"
        ],
        [
            "{re}PeftModelForCausalLM/LoraModel[base_model]/LlamaForCausalLM[model]/LlamaModel[model]/ModuleList[layers]/LlamaDecoderLayer[{*}]/LlamaMLP[mlp]/Linear[down_proj]/ModuleDict[lora_A]/NNCFLinear[default]/linear_0"
        ]
    ],
    "overwrite_groups_widths": [
        [32, 24, 16], [32, 24, 16], [32, 24, 16]
    ]
}
```

Note that the length of groups should be equal to the length of the group widths, and we only set the output hidden 
size space of LoRA-A in the config, as the input hidden size of LoRA-B will be automatically pruned according to LoRA-A.
Feel free to try your own group design and search spaces.
