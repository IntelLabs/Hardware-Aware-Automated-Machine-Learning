# SQFT: Low-cost Model Adaptation in Low-precision Sparse Foundation Models

Official implementation of [SQFT: Low-cost Model Adaptation in Low-precision Sparse Foundation Models](https://arxiv.org/pdf/2410.03750).

This repository contains the code for **SQFT**, an end-to-end solution for low-precision sparse parameter-efficient fine-tuning of LLMs. SQFT allows for effective model manipulation in resource-constrained environments.
Specifically, the highlights of SQFT include:

- **SparsePEFT**, an efficient and effective strategy for fine-tuning sparse models. It ensures the preservation of the base model's sparsity during merging through the use of sparse adapters.
- Introduction of quantization scenarios (sparse and quantization). **QA-SparsePEFT** built on SparsePEFT, which allows PEFT fine-tuning to achieve a single INT4 and sparse model adapted to the specific domain.
- Adopt **Neural Low-rank Adapter Search (NLS)** strategies into all pipelines and solutions. 

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
pip install torch==2.5.1

# install dependencies
bash install.sh
```

## Quick start

### Training and Evaluation

We use **Llama-3-8B + GSM8K** as an example to show how to instantiate SQFT's pipelines. 
For other settings and models, see the [legacy](./legacy) version.

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
Note that the sparsification step can use any other weight sparsification algorithm and sparsity patterns, such as the current state-of-the-art sparsity method [MaskLLM](https://vainf.github.io/maskllm-project-page/) (2:4 sparsity).
Feel free to try other sparse approaches for the base model before training.

#### Quantization

> If you do not consider low precision, you can skip this step and directly jump to [SQFT + SparsePEFT](#hammer_and_wrench-sqft--sparsepeft).

Quantize the base model using GPTQ:
```bash
python utils/quantization.py --base_model_path <path to sparse base model> --output_dir <path to quantized sparse base model>
```

You can also skip the quantization step and adopt our released quantized models (find them in the *Sparse-and-Quantized Model* column of [this Table](#released-foundation-models-)). 

#### :hammer_and_wrench: SQFT

- Fine-tuning

```bash
python run_sqft.py \
    --dataset_name gsm8k \
    --model_name_or_path <path to quantized sparse base model> \
    --do_train \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 3 \
    --learning_rate 1e-4 \
    --warmup_steps 100 \
    --lr_scheduler_type cosine \
    --optim adamw_torch \
    --fp16 \
    --output_dir <path to super-adapter> \
    --logging_steps 20 \
    --save_strategy epoch \
    --save_total_limit 2 \
    --lora_r 32 \
    --target_modules q_proj k_proj v_proj up_proj down_proj \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --nls \
    --nls_target_modules q_proj k_proj v_proj \
    --search_space 32 24 16
```

Some explanations about parameters related to the NLS strategy:

- `--nls`: determines whether to apply Neural LoRA Search (NLS) or not. When set to True, the training process will include NLS, which is a technique designed to optimize the low-rank adaptation of the model.
- `--nls_target_modules`: specifies which modules within the model will have the elastic LoRA adapter applied. The modules listed here will be the target of the NLS process.
- `--search_space`: defines the low-rank search space for NLS training. It is a list of integers that represent the different ranks to be considered during the search process.

After completing the super-adapter training, the command to extract the heuristic sub-adapter is as follows. 
Additionally, more powerful sub-adapters can be obtained through other advanced search algorithms.

```bash
python utils/extract_sub_adapter.py \
  --adapter_model <path to super-adapter> \
  --elastic_adapter_config_file <path to super-adapter>/elastic_adapter_config.json \
  --adapter_version heuristic \
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

We also provide a LoRA version (without the NLS strategy), allowing us to choose whether to enable NLS based on the actual situation.

<details>
<summary>LoRA</summary>

```bash
python run_sqft.py \
    --dataset_name gsm8k \
    --model_name_or_path <path to quantized sparse base model> \
    --do_train \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 3 \
    --learning_rate 1e-4 \
    --warmup_steps 100 \
    --lr_scheduler_type cosine \
    --optim adamw_torch \
    --fp16 \
    --output_dir <path to trained adapter> \
    --logging_steps 20 \
    --save_strategy epoch \
    --save_total_limit 2 \
    --lora_r 32 \
    --target_modules q_proj k_proj v_proj up_proj down_proj \
    --lora_alpha 64 \
    --lora_dropout 0.1 

lm_eval --model hf \
    --model_args pretrained=<path to quantized sparse base model>,peft=<path to trained adapter>,add_bos_token=True \
    --tasks gsm8k \
    --batch_size auto:4 \
    --output_path result.json
```

</details>

#### :hammer_and_wrench: SQFT + SparsePEFT

- Fine-tuning

```bash
python run_sqft.py \
    --dataset_name gsm8k \
    --model_name_or_path <path to sparse base model> \
    --do_train \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 3 \
    --learning_rate 3e-4 \
    --warmup_steps 100 \
    --lr_scheduler_type cosine \
    --optim adamw_torch \
    --fp16 \
    --output_dir <path to super-adapter> \
    --logging_steps 20 \
    --save_strategy epoch \
    --save_total_limit 2 \
    --lora_r 32 \
    --target_modules q_proj k_proj v_proj up_proj down_proj \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --nls \
    --nls_target_modules q_proj k_proj v_proj \
    --search_space 32 24 16 \
    --sparse_adapter  # enable SparsePEFT
```

Extract the heuristic sub-adapter:

```bash
python utils/extract_sub_adapter.py --adapter_model <path to super-adapter> --elastic_adapter_config_file <path to super-adapter>/elastic_adapter_config.json --adapter_version heuristic --output_dir <path to sub-adapter>
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

<details>
<summary>LoRA</summary>

```bash
python run_sqft.py \
    --dataset_name gsm8k \
    --model_name_or_path <path to sparse base model> \
    --do_train \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 3 \
    --learning_rate 3e-4 \
    --warmup_steps 100 \
    --lr_scheduler_type cosine \
    --optim adamw_torch \
    --fp16 \
    --output_dir <path to trained adapter> \
    --logging_steps 20 \
    --save_strategy epoch \
    --save_total_limit 2 \
    --lora_r 32 \
    --target_modules q_proj k_proj v_proj up_proj down_proj \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --sparse_adapter  # enable SparsePEFT

python utils/merge.py --base_model_path <path to sparse base model> --adapter_model_path <path to trained adapter> --output_path <path to merged model>
python utils/check_sparsity.py --model_path <path to merged model>

lm_eval --model hf \
    --model_args pretrained=<path to merged model>,add_bos_token=True \
    --tasks gsm8k \
    --batch_size auto:4 \
    --output_path result.json
```

</details>

#### :hammer_and_wrench: SQFT + QA-SparsePEFT

- Fine-tuning

```bash
python run_sqft.py \
    --dataset_name gsm8k \
    --model_name_or_path <path to quantized sparse base model> \
    --non_quant_model_name_or_path <path to sparse base model> \
    --do_train \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 4 \
    --learning_rate 3e-4 \
    --warmup_steps 100 \
    --lr_scheduler_type cosine \
    --optim adamw_torch \
    --fp16 \
    --output_dir <path to super-adapter> \
    --logging_steps 20 \
    --save_strategy epoch \
    --save_total_limit 2 \
    --lora_r 32 \
    --target_modules q_proj k_proj v_proj up_proj down_proj \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --nls \
    --nls_target_modules q_proj k_proj v_proj \
    --search_space 32 24 16 \
    --sparse_adapter \
    --quantization_aware # enable quantization-aware SparsePEFT
```

Extract the heuristic sub-adapter:

```bash
python utils/extract_sub_adapter.py --adapter_model <path to super-adapter> --elastic_adapter_config_file <path to super-adapter>/elastic_adapter_config.json --adapter_version heuristic --output_dir <path to sub-adapter>
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

<details>
<summary>LoRA</summary>

```bash
python run_sqft.py \
    --dataset_name gsm8k \
    --model_name_or_path <path to quantized sparse base model> \
    --non_quant_model_name_or_path <path to sparse base model> \
    --do_train \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 4 \
    --learning_rate 3e-4 \
    --warmup_steps 100 \
    --lr_scheduler_type cosine \
    --optim adamw_torch \
    --fp16 \
    --output_dir <path to trained adapter> \
    --logging_steps 20 \
    --save_strategy epoch \
    --save_total_limit 2 \
    --lora_r 32 \
    --target_modules q_proj k_proj v_proj up_proj down_proj \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --sparse_adapter \
    --quantization_aware # enable quantization-aware SparsePEFT

python utils/merge.py \
  --base_model_path <path to quantized sparse base model> \
  --non_quant_base_model_path <path to sparse base model> \
  --adapter_model_path <path to trained adapter> \
  --output_path <path to merged model>

python utils/check_sparsity.py --model_path <path to merged model>

lm_eval --model hf \
    --model_args pretrained=<path to merged model>,add_bos_token=True \
    --tasks gsm8k \
    --batch_size auto:4 \
    --output_path result.json
``````

</details>

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
If you find SQFT's code and papers helpful, please kindly cite:
```bibtex
@inproceedings{munoz-etal-2024-sqft,
    title = "{SQFT}: Low-cost Model Adaptation in Low-precision Sparse Foundation Models",
    author = "Munoz, J. Pablo  and
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
```bibtex
@inproceedings{munoz2025lowrank,
    title=Low-Rank Adapters Meet Neural Architecture Search for LLM Compression,
    author="Munoz, J. Pablo  and
      Yuan, Jinjie  and
      Jain, Nilesh",,
    booktitle={AAAI'25 workshop on CoLoRAI - Connecting Low-Rank Representations in AI},
    year={2025},
    url={https://arxiv.org/abs/2501.16372}
}
```