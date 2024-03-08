# :scissors:Shears: Fine-tuning For Efficient Models

Official implementation of [Shears: Unstructured Sparsity with Neural Low-rank Adapter Search](). :fire:

This repo proposes **Shears**, a practical and novel solution for real-world applications to generate efficient models fine-tuned for downstream specific tasks.

## News
- **[2024.03.07]**  The training and inference code for **Shears V1** are released. :tada:

## Released Models 🤗

| Name                                                                                                                 | Super-network                                                                                                       | Sparsity                                               | Train Data                                                         | Base Model
|----------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------| ------------------------------------------------------ | ------------------------------------------------------------ | ------- | 
| [shears-llama-7b-50-math-heuristic](https://huggingface.co/IntelLabs/shears-llama-7b-50-math-heuristic) | [shears-llama-7b-50-math-super](https://huggingface.co/IntelLabs/shears-llama-7b-50-math-super)             | 50%           | [Unified Math](https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/ft-training_set/math_10k.json)         | [yahma/llama-7b-hf](https://huggingface.co/yahma/llama-7b-hf)
| [shears-llama-7b-50-commonsense-heuristic](https://huggingface.co/IntelLabs/shears-llama-7b-50-commonsense-heuristic) | [shears-llama-7b-50-commonsense-super](https://huggingface.co/IntelLabs/shears-llama-7b-50-commonsense-super) | 50%                                                                               | [Unified Commonsense](https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/ft-training_set/commonsense_170k.json) | [yahma/llama-7b-hf](https://huggingface.co/yahma/llama-7b-hf)
| [shears-llama-13b-50-math-heuristic](https://huggingface.co/IntelLabs/shears-llama-13b-50-math-heuristic) | [shears-llama-13b-50-math-super](https://huggingface.co/IntelLabs/shears-llama-13b-50-math-super)             | 50%                                                                                      | [Unified Math](https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/ft-training_set/math_10k.json)      |  [yahma/llama-13b-hf](https://huggingface.co/yahma/llama-13b-hf)
| [shears-mpt-7b-50-gsm8k-heuristic](https://huggingface.co/IntelLabs/shears-mpt-7b-50-gsm8k-heuristic) | [shears-mpt-7b-50-gsm8k-super](https://huggingface.co/IntelLabs/shears-mpt-7b-50-gsm8k-super)                 | 50%                                                                                      | [GSM8K](https://huggingface.co/datasets/gsm8k)              | [mosaicml/mpt-7b](https://huggingface.co/mosaicml/mpt-7b)

## Overview

Efficiency Comparison (use LLaMA-13B as an example):
|  Method | Sparsity | Non-zero Params. | Acc. | Inference Speedup  
| :-----: | :-----: |:-----:| :-----: | :-----: |
|  [LoRA](https://github.com/microsoft/LoRA) | - | 13.0B | 51.1 | 1.00x |
| Shears | 50% | 6.7B | 50.9 | 1.84x |

By incorporating elastic LoRA adapters into the sparsified base model, Shears can fine-tune the language model without sacrificing the sparsity obtained from the original model weights, and produces sparse models with improvements or minor drops in accuracy and a fraction of the cost when compared to other approaches. The increase in sparsity can result in significant speedup when using runtimes that take advantage of these patterns. Please refer to our [paper]() for more details.

Overall, Shears has well-designed a simple yet effective, powerful and general pipeline, allowing users to easily extend it to their desired scenarios/tasks, even audio and video. Feel free to try Shears for any downstream task with any model. :relaxed:

## Setup

Here is an installation script developed from scratch for **Shears**.

```
conda create -n shears -y python=3.10
conda activate shears

# install dependencies
bash install.sh
```
Note: please ignore the whitespace issues when applying the patch when running `install.sh`.

## Quick Start

### Inference

Please download our released models from [Hugging Face](https://huggingface.co/IntelLabs) - checkout [Table](#released-models-) for model information. 
Each Shears model contains the pretrained weights of the sparsified language model and the weights of the fine-tuned adapters.
Denote the path to the downloaded model as `SHEARS_PATH`.

Below is an example to generate the instruction-following responses for some math reasoning samples:
```bash
CUDA_VISIBLE_DEVICES=$DEVICES python example.py --model_path $SHEARS_PATH
```

### Training

#### Step 1. Unstructured Sparsifying

Before fine-tuning, Shears employes a simple but effective pruning approach [Wanda](https://arxiv.org/abs/2306.11695) to sparsify the language model, serving as the base model (frozen) for adapter training.
Clone the [Wanda](https://github.com/locuslab/wanda) repo:

```bash
git clone https://github.com/locuslab/wanda.git && pushd wanda && git checkout 8e8fc87 && popd
```

Below is an example command for unstructured sparsifying LLaMA-7b with Wanda, to achieve unstructured 50% sparsity (takes about five minutes).
```bash
CUDA_VISIBLE_DEVICES=$DEVICES python wanda/main.py \
    --model yahma/llama-7b-hf \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save wanda_out \
    --save_model unstructured_sparsity_models/llama-7b-sparsity50
```
- `--model`: The identifier for the model on the Hugging Face model hub or local path.
- `--sparsity_ratio`: Specifies the percentage of weights to be pruned.
- `--save_model`: Specifies the directory where the pruned language model will be stored.

Further details can be referred to [Wanda](https://github.com/locuslab/wanda).

#### Step 2. Adapter Training

Taking the unified math reasoning training as an example, please download the 10K instruction-following [math reasoning training data](https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/ft-training_set/math_10k.json) from [LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters), and place it under `DATA_PATH`. 

Example command to train the super-adapter of the pruned LLaMA-7B using Shears:

```bash
CUDA_VISIBLE_DEVICES=$DEVICES python run_math.py \
    --dataset_path $DATA_PATH/math_10k.json \
    --model_name_or_path unstructured_sparsity_models/llama-7b-sparsity50 \
    --do_train \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3 \
    --warmup_steps 100 \
    --optim adamw_torch \
    --fp16 \
    --output_dir trained_super_adapter/unified_math/llama-7b-sparsity50-shears-math-adapter \
    --logging_steps 20 \
    --save_strategy epoch \
    --save_total_limit 2 \
    --lora \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.1 \
    --target_modules q_proj,k_proj,v_proj,up_proj,down_proj \
    --nncf_config nncf_config/unified_math/nncf_shears_llama_7b_sparsity50.json
```

`nncf_config` indicates the NNCF configuration including the search space for elastic adapters.
To implement the elastic adapter, we apply the BootstrapNAS feature supported in [OpenVINO™ NNCF](https://github.com/openvinotoolkit/nncf), which provides a suite of compression algorithms for neural networks optimization.
Note that we use the stage LR scheduler in NNCF, so the learning rate is set in the NNCF config file, instead of `learning_rate` of `TrainingArguments`.

### Evaluation

The math evaluation datasets can be downloaded from [LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters).
We place them into the directory `datasets/`.
```
git clone https://github.com/AGI-Edgerunners/LLM-Adapters.git
mv LLM-Adapters/dataset/ datasets/ 
```

Example command to evaluate the trained super-adapter (heuristic subnetwork):

```bash
CUDA_VISIBLE_DEVICES=$DEVICES python run_math.py \
    --dataset_path None \
    --model_name_or_path unstructured_sparsity_models/llama-7b-sparsity50 \
    --lora \
    --lora_weights trained_super_adapter/unified_math/llama-7b-sparsity50-shears-math-adapter \
    --do_test \
    --fp16 \
    --output_dir trained_super_adapter/unified_math/llama-7b-sparsity50-shears-math-adapter/results \
    --nncf_config nncf_config/unified_math/nncf_shears_llama_7b_sparsity50.json
```

## Reproduce Results

Please refer to `running_commands` for all commands related to reproducing the results of the paper.

## Citation
If you find our Shears code and paper helpful, please kindly cite:
```bash
@article{munoz2024shears,
  title = {Shears: Unstructured Sparsity with Neural Low-rank Adapter Search},
  author={J. Pablo Munoz and Jinjie Yuan and Nilesh Jain},
  journal={},
  year={2024}
}
```

## Acknowledgement
This work benefits from the following contributions:

- LLaMA: [https://github.com/facebookresearch/llama](https://github.com/facebookresearch/llama)
- MPT: [https://www.mosaicml.com/mpt](https://www.mosaicml.com/mpt)
- Peft: [https://github.com/huggingface/peft](https://github.com/huggingface/peft)
- LLM-Adapters: [LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters)
- NNCF: [https://github.com/openvinotoolkit/nncf](https://github.com/openvinotoolkit/nncf)
- BootstrapNAS: [https://github.com/jpablomch/bootstrapnas](https://github.com/jpablomch/bootstrapnas)
