# SQFT: Low-cost Model Adaptation in Low-precision Sparse Foundation Models

Official implementation of [SQFT: Low-cost Model Adaptation in Low-precision Sparse Foundation Models](https://arxiv.org/pdf/2410.03750).

This repo contains the code for **SQFT**, an end-to-end solution for low-precision sparse parameter-efficient fine-tuning of LLMs. SQFT allows for effective model manipulation in resource-constrained environments. 
Please refer to our paper for more details.

## News
- **[2024.09.24]**  **SQFT** paper has been released ([link](https://arxiv.org/pdf/2410.03750)) and **accepted at EMNLP 2024 Findings**. :books:
- **[2024.09.24]** Release the code for **SQFT**. :tada:

## Released Foundation Models 🤗

We have released several foundation models (sparse or sparse-and-quantized) for SQFT:

| Source Model                                                                      | Sparsity | Sparse Model                                                                                            | Sparse-and-Quantized Model                                                                                        |
|-----------------------------------------------------------------------------------|----------|---------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| [Mistral-7B-v0.3](https://huggingface.co/mistralai/Mistral-7B-v0.3)               | 50%      | [IntelLabs/sqft-mistral-7b-v0.3-50-base](https://huggingface.co/IntelLabs/sqft-mistral-7b-v0.3-50-base) | [IntelLabs/sqft-mistral-7b-v0.3-50-base-gptq](https://huggingface.co/IntelLabs/sqft-mistral-7b-v0.3-50-base-gptq) |
| [Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) | 50%      | [IntelLabs/sqft-phi-3-mini-4k-50-base](https://huggingface.co/IntelLabs/sqft-phi-3-mini-4k-50-base)     | [IntelLabs/sqft-phi-3-mini-4k-50-base-gptq](https://huggingface.co/IntelLabs/sqft-phi-3-mini-4k-50-base-gptq)     |
| [Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)              | 50%      | IntelLabs/sqft-llama-3-8b-50-base<sup>*</sup>                                                           | IntelLabs/sqft-llama-3-8b-50-base-gptq<sup>*</sup>                                                                |

<sup>*</sup> *Llama-3 models are currently under internal review and will be released soon.* 

[//]: # (https://huggingface.co/IntelLabs/sqft-llama-3-8b-50-base)
[//]: # (https://huggingface.co/IntelLabs/sqft-llama-3-8b-50-base-gptq)

## Setup

Follow the steps below to use **SQFT**.

```
pip install virtualenv
virtualenv sqft-env
source sqft-env/bin/activate

# install pytorch
pip install torch==2.3.1

# install dependencies
bash install.sh # if only for inference: bash install_inference.sh
```

## Experimental Settings

In our experiments, we conduct three fine-tuning settings to explore SQFT:

:hammer_and_pick: **Standard fine-tuning: GSM8K**

- Train: [GSM8K](https://huggingface.co/datasets/gsm8k) training dataset
- Evaluation: GSM8K ([lm-evaluation-harness v0.4.2](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.2))

:hammer_and_pick: **Instruction fine-tuning: Math**

- Train: 10K instruction-following math reasoning training dataset from [LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters) ([math_10k.json](https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/ft-training_set/math_10k.json))
- Evaluation: GSM8K, SVAMP, MAWPS (following the evaluation script from LLM-Adapters)

:hammer_and_pick: **Standard fine-tuning: Commonsense Reasoning**

- Train: [winogrande](https://huggingface.co/datasets/winogrande), [boolq](https://huggingface.co/datasets/google/boolq), [openbookqa](https://huggingface.co/datasets/allenai/openbookqa), [hellaswag](https://huggingface.co/datasets/Rowan/hellaswag), [piqa](https://huggingface.co/datasets/piqa), [ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) training dataset (83k)
- Evaluation: 7 commonsense reasoning datasets - BoolQ, PIQA, HellaS, WinoG, ARC-e, ARC-c and OBQA ([lm-evaluation-harness v0.4.2](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.2))


## Quick start

### Inference

The following code shows an inference example: 
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
model = AutoModelForCausalLM.from_pretrained("IntelLabs/sqft-qa-sparsepeft-mistral-7b-v0.3-50-gptq-gsm8k-heu", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("IntelLabs/sqft-qa-sparsepeft-mistral-7b-v0.3-50-gptq-gsm8k-heu")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
output = pipe("Question: Gretchen has 110 coins. There are 30 more gold coins than silver coins. How many gold coins does Gretchen have?", max_new_tokens=256)
```

### Training and Evaluation

We use Llama-3-8B + GSM8K as an example to show how to instantiate SQFT's pipelines. See 
[run_command](./run_command) for all other settings and models.

#### Sparsification

Before fine-tuning, SQFT employs a simple but effective pruning approach [Wanda](https://arxiv.org/abs/2306.11695) to 
sparsify the language model, serving as the base model (frozen) for adapter training.
Clone the Wanda repo and apply our patch:

```bath
git clone https://github.com/locuslab/wanda.git && cd wanda && git checkout 8e8fc87 && git apply ../patches/wanda-8e8fc87.patch && cd ..
```

Below is an example command for unstructured sparsifying [Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) 
with Wanda to achieve unstructured 50% sparsity.
```bash
python wanda/main.py \
    --model meta-llama/Meta-Llama-3-8B \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save wanda_out \
    --save_model <path to sparse base model>
```
- `--model`: The identifier for the model on the Hugging Face model hub or local path.
- `--sparsity_ratio`: Specifies the percentage of weights to be pruned.
- `--save_model`: Specifies the directory where the sparsified language model will be stored.

Further details can be referred to [Wanda](https://github.com/locuslab/wanda). You can skip this step and adopt our released sparse models (find them in the *Sparse Model* column of [this Table](#released-foundation-models-)). 
Note that the sparsifying step can use any other weight sparsity algorithm. 
Feel free to try other sparse approaches for the base model before training.

#### Quantization

Quantize the base model using GPTQ:
```bash
python utils/quantization.py --base_model_path <path to sparse base model> --output_dir <path to quantized sparse base model>
```

You can also skip the quantization step and adopt our released quantized models (find them in the *Sparse-and-Quantized Model* column of [this Table](#released-foundation-models-)). 

#### :hammer_and_wrench: SQFT

- Fine-tuning

```bash
python run_standard_tuning.py \
    --dataset_name gsm8k \
    --model_name_or_path <path to quantized sparse base model> \
    --do_train \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 3 \
    --learning_rate 3e-4 \
    --warmup_steps 100 \
    --optim adamw_torch \
    --fp16 \
    --output_dir <path to super-adapter> \
    --logging_steps 20 \
    --save_strategy epoch \
    --save_total_limit 2 \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --nls \
    --target_module_groups q_proj,k_proj,v_proj up_proj down_proj \
    --search_space 32,24,16 32,24,16 32,24,16 # low-rank search space for each group ([q_proj,k_proj,v_proj], [up_proj], [down_proj])
```

After completing the super-adapter training, the command to extract the heuristic sub-adapter is as follows. 
Additionally, more powerful sub-adapters can be obtained through other advanced search algorithms.

```bash
python utils/extract_sub_adapter.py \
  --adapter_model <path to super-adapter> \
  --nncf_config <path to super-adapter>/nncf_config.json \
  --subnet_version heuristic \
  --output_dir <path to sub-adapter>
```

- Evaluation

```bash
lm_eval --model hf \
    --model_args pretrained=<path to quantized sparse base model>,peft=<path to sub-adapter>,add_bos_token=True \
    --tasks gsm8k \
    --batch_size auto:4 \
    --output_path result.json
```

#### :hammer_and_wrench: SQFT + SparsePEFT

- Fine-tuning

```bash
python run_standard_tuning.py \
    --dataset_name gsm8k \
    --model_name_or_path <path to sparse base model> \
    --do_train \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 3 \
    --learning_rate 3e-4 \
    --warmup_steps 100 \
    --optim adamw_torch \
    --fp16 \
    --output_dir <path to super-adapter> \
    --logging_steps 20 \
    --save_strategy epoch \
    --save_total_limit 2 \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --nls \
    --target_module_groups q_proj,k_proj,v_proj up_proj down_proj \
    --search_space 32,24,16 32,24,16 32,24,16 \
    --sparse_adapter  # enable SparsePEFT
```

Extract the heuristic sub-adapter:

```bash
python utils/extract_sub_adapter.py --adapter_model <path to super-adapter> --nncf_config <path to super-adapter>/nncf_config.json --subnet_version heuristic --output_dir <path to sub-adapter>
```

Merge the adapter to the base model and check the sparsity of the merged model:

```bash
python utils/merge.py --base_model_path <path to sparse base model> --adapter_model_path <path to sub-adapter> --output_path <path to merged model>
python utils/check_sparsity.py --model_path <path to merged model>
```

- Evaluation

```bash
lm_eval --model hf \
    --model_args pretrained=<path to merged model>,add_bos_token=True \
    --tasks gsm8k \
    --batch_size auto:4 \
    --output_path result.json
```

#### :hammer_and_wrench: SQFT + QA-SparsePEFT

- Fine-tuning

```bash
python run_standard_tuning.py \
    --dataset_name gsm8k \
    --model_name_or_path <path to quantized sparse base model> \
    --non_quant_model_name_or_path <path to sparse base model> \
    --do_train \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 4 \
    --learning_rate 3e-4 \
    --warmup_steps 100 \
    --optim adamw_torch \
    --fp16 \
    --output_dir <path to super-adapter> \
    --logging_steps 20 \
    --save_strategy epoch \
    --save_total_limit 2 \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --nls \
    --target_module_groups q_proj,k_proj,v_proj up_proj down_proj \
    --search_space 32,24,16 32,24,16 32,24,16 \
    --sparse_adapter \
    --quantization_aware # enable quantization-aware SparsePEFT
```

Extract the heuristic sub-adapter:

```bash
python utils/extract_sub_adapter.py --adapter_model <path to super-adapter> --nncf_config <path to super-adapter>/nncf_config.json --subnet_version heuristic --output_dir <path to sub-adapter>
```

Merge the adapter to the quantized base model and check the sparsity of the merged model:

```bash
python utils/merge.py \
  --base_model_path <path to quantized sparse base model> \
  --non_quant_base_model_path <path to sparse base model> \
  --adapter_model_path <path to sub-adapter> \
  --output_path <path to merged model>

python utils/check_sparsity.py --model_path <path to merged model>
```

- Evaluation

```bash
lm_eval --model hf \
    --model_args pretrained=<path to merged model>,add_bos_token=True \
    --tasks gsm8k \
    --batch_size auto:4 \
    --output_path result.json
```

## Released Fine-tuned Models 🤗

- Mistral-7B-v0.3

| Base Model                                                                                              | Task  | Method               | Fine-tuned Model                                                                                                                              |
|---------------------------------------------------------------------------------------------------------|-------|----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| [sqft-mistral-7b-v0.3-50-base](https://huggingface.co/IntelLabs/sqft-mistral-7b-v0.3-50-base)           | GSM8K | SQFT + SparsePEFT    | [sqft-sparsepeft-mistral-7b-v0.3-50-gsm8k-heu](https://huggingface.co/IntelLabs/sqft-sparsepeft-mistral-7b-v0.3-50-gsm8k-heu)                 |
| [sqft-mistral-7b-v0.3-50-base-gptq](https://huggingface.co/IntelLabs/sqft-mistral-7b-v0.3-50-base-gptq) | GSM8K | SQFT                 | [sqft-mistral-7b-v0.3-50-gptq-gsm8k-heu-adapter](https://huggingface.co/IntelLabs/sqft-mistral-7b-v0.3-50-gptq-gsm8k-heu-adapter)             |
| [sqft-mistral-7b-v0.3-50-base-gptq](https://huggingface.co/IntelLabs/sqft-mistral-7b-v0.3-50-base-gptq) | GSM8K | SQFT + QA-SparsePEFT | [sqft-qa-sparsepeft-mistral-7b-v0.3-50-gptq-gsm8k-heu](https://huggingface.co/IntelLabs/sqft-qa-sparsepeft-mistral-7b-v0.3-50-gptq-gsm8k-heu) |
| [sqft-mistral-7b-v0.3-50-base](https://huggingface.co/IntelLabs/sqft-mistral-7b-v0.3-50-base)           | Math  | SQFT + SparsePEFT    | [sqft-sparsepeft-mistral-7b-v0.3-50-math-heu](https://huggingface.co/IntelLabs/sqft-sparsepeft-mistral-7b-v0.3-50-math-heu)                   |
| [sqft-mistral-7b-v0.3-50-base-gptq](https://huggingface.co/IntelLabs/sqft-mistral-7b-v0.3-50-base-gptq) | Math  | SQFT                 | [sqft-mistral-7b-v0.3-50-gptq-math-heu-adapter](https://huggingface.co/IntelLabs/sqft-mistral-7b-v0.3-50-gptq-math-heu-adapter)               |
| [sqft-mistral-7b-v0.3-50-base-gptq](https://huggingface.co/IntelLabs/sqft-mistral-7b-v0.3-50-base-gptq) | Math  | SQFT + QA-SparsePEFT | [sqft-qa-sparsepeft-mistral-7b-v0.3-50-gptq-math-heu](https://huggingface.co/IntelLabs/sqft-qa-sparsepeft-mistral-7b-v0.3-50-gptq-math-heu)   |

- Phi-3-mini-4k-instruct

| Base Model                                                                                          | Task | Method               | Fine-tuned Model                                                                                                                        |
|-----------------------------------------------------------------------------------------------------|------|----------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| [sqft-phi-3-mini-4k-50-base](https://huggingface.co/IntelLabs/sqft-phi-3-mini-4k-50-base)           | Math | SQFT + SparsePEFT    | [sqft-sparsepeft-phi-3-mini-4k-50-math-heu](https://huggingface.co/IntelLabs/sqft-sparsepeft-phi-3-mini-4k-50-math-heu)                 |
| [sqft-phi-3-mini-4k-50-base-gptq](https://huggingface.co/IntelLabs/sqft-phi-3-mini-4k-50-base-gptq) | Math | SQFT                 | [sqft-phi-3-mini-4k-50-gptq-math-heu-adapter](https://huggingface.co/IntelLabs/sqft-phi-3-mini-4k-50-gptq-math-heu-adapter)             |
| [sqft-phi-3-mini-4k-50-base-gptq](https://huggingface.co/IntelLabs/sqft-phi-3-mini-4k-50-base-gptq) | Math | SQFT + QA-SparsePEFT | [sqft-qa-sparsepeft-phi-3-mini-4k-50-gptq-math-heu](https://huggingface.co/IntelLabs/sqft-qa-sparsepeft-phi-3-mini-4k-50-gptq-math-heu) |
| [sqft-phi-3-mini-4k-50-base](https://huggingface.co/IntelLabs/sqft-phi-3-mini-4k-50-base)           | CS   | SQFT + SparsePEFT    | [sqft-sparsepeft-phi-3-mini-4k-50-cs-heu](https://huggingface.co/IntelLabs/sqft-sparsepeft-phi-3-mini-4k-50-cs-heu)                     |
| [sqft-phi-3-mini-4k-50-base-gptq](https://huggingface.co/IntelLabs/sqft-phi-3-mini-4k-50-base-gptq) | CS   | SQFT                 | [sqft-phi-3-mini-4k-50-gptq-cs-heu-adapter](https://huggingface.co/IntelLabs/sqft-phi-3-mini-4k-50-gptq-cs-heu-adapter)                 |
| [sqft-phi-3-mini-4k-50-base-gptq](https://huggingface.co/IntelLabs/sqft-phi-3-mini-4k-50-base-gptq) | CS   | SQFT + QA-SparsePEFT | [sqft-qa-sparsepeft-phi-3-mini-4k-50-gptq-cs-heu](https://huggingface.co/IntelLabs/sqft-qa-sparsepeft-phi-3-mini-4k-50-gptq-cs-heu)     |


- Meta-Llama-3-8B

| Base Model                          | Task    | Method                 | Fine-tuned Model                                |
|-------------------------------------|---------|------------------------|-------------------------------------------------|
| sqft-llama-3-8b-50-base<sup>*</sup> | GSM8K   | SQFT + SparsePEFT      | sqft-sparsepeft-llama-3-8b-50-gsm8k-heu         | 
| sqft-llama-3-8b-50-base-gptq        | GSM8K   | SQFT                   | sqft-llama-3-8b-50-gptq-gsm8k-heu-adapter       |
| sqft-llama-3-8b-50-base-gptq        | GSM8K   | SQFT + QA-SparsePEFT   | sqft-qa-sparsepeft-llama-3-8b-50-gptq-gsm8k-heu |

<sup>*</sup> *Llama-3 models are currently under internal review and will be released soon.* 

[//]: # (https://huggingface.co/IntelLabs/sqft-llama-3-8b-50-base)

[//]: # (https://huggingface.co/IntelLabs/sqft-llama-3-8b-50-gptq-gsm8k-heu-adapter)

[//]: # (https://huggingface.co/IntelLabs/sqft-llama-3-8b-50-base-gptq)

[//]: # (https://huggingface.co/IntelLabs/sqft-sparsepeft-llama-3-8b-50-gsm8k-heu)

[//]: # (https://huggingface.co/IntelLabs/sqft-llama-3-8b-50-base-gptq)

[//]: # (https://huggingface.co/IntelLabs/sqft-qa-sparsepeft-llama-3-8b-50-gptq-gsm8k-heu)

## Citation
If you find SQFT's code and paper helpful, please kindly cite:
```bibtex
@inproceedings{munoz-etal-2024-sqft,
    title = "{SQFT}: Low-cost Model Adaptation in Low-precision Sparse Foundation Models",
    author = "Munoz, Juan Pablo  and
      Yuan, Jinjie  and
      Jain, Nilesh",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.749",
    pages = "12817--12832",
}
```