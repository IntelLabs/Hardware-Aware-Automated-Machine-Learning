# Shears: Unstructured Sparsity with Neural Low-rank Adapter Search

PyTorch implementation of Shears.

## Setup

This repository is tested on Python* 3.11.1, PyTorch* 2.2.0 (NVidia CUDA* Toolkit 12.3)

Step 1: Create a new conda environment
```
conda create -n shears python=3.11.1
conda activate shears
```
Step 2: Install relevant packages
```
# Pytorch
pip install torch==2.2.0 -f https://download.pytorch.org/whl/torch_stable.html

mkdir third_party && cd third_party

# HuggingFace Transformers
git clone https://github.com/huggingface/transformers.git transformers && cd transformers
git checkout v4.31.0
git apply --ignore-space-change --ignore-whitespace /PATH/TO/patches/transformers-modifications-for-shears-usage.patch
pip install -e . && cd ..

# HuggingFace Peft
git clone https://github.com/huggingface/peft.git peft && cd peft
git checkout v0.5.0
git apply --ignore-space-change --ignore-whitespace /PATH/TO/patches/peft-modifications-for-shears-usage.patch
pip install -e . --no-deps && cd ..

# NNCF
git clone https://github.com/openvinotoolkit/nncf.git nncf && cd nncf
git checkout 544d5141
git apply --ignore-space-change --ignore-whitespace /PATH/TO/patches/nncf-modifications-for-shears-usage.patch
pip install -e . && cd ..

# Others
pip install datasets accelerate sentencepiece tokenizers==0.13.3
```
Note: please ignore the whitespace issues when applying the patch.

## Usage

### Model

The models we use for experiment:

- LLaMA-7B: [yahma/llama-7b](https://huggingface.co/yahma/llama-7b)
- LLaMA-13B: [yahma/llama-13b](https://huggingface.co/yahma/llama-13b)
- MPT-7B: [mosaicml/mpt-7b](https://huggingface.co/mosaicml/mpt-7b)


### Dataset Preparation

For the unified training setup, the training and test datasets can be downloaded from [LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters).
We place them into the directory `datasets/`.

```
git clone https://github.com/AGI-Edgerunners/LLM-Adapters.git && cd LLM-Adapters && git checkout 0df194e && cd ..
mv LLM-Adapters/dataset/ datasets && mv LLM-Adapters/ft-training_set/math_10k.json datasets/ && mv LLM-Adapters/ft-training_set/commonsense_170k.json datasets/ && rm -rf LLM-Adapters
```
`datasets/` contains the training data and the test datasets.
Following [LLM-Adpater](https://github.com/AGI-Edgerunners/LLM-Adapters), in the unified math setup we use `math_10k.json` for training, while in the unified commonsense setup we use `commonsense_170k.json`.
Test datasets includes:
- Math: `GSM8K`, `AQuA`, `MAWPS` and `SVAMP`.
- Commonsense: `BoolQ`, `PIQA`, `SIQA`, `HellaSwag`, `WinoGrande`, `ARC-e`, `ARC-c` and `OBQA`.

### Run

#### 1. Unstructured Sparsity Step

We apply a simple but effective pruning approach [Wanda](https://github.com/locuslab/wanda):

```bash
git clone https://github.com/locuslab/wanda.git && cd wanda && git checkout 8e8fc87 && cd ..
```

Below is an example command for unstructured pruning LLaMA-7b with Wanda, to achieve unstructured 50% sparsity.
```bash
CUDA_VISIBLE_DEVICES=${DEVICES} python wanda/main.py \
    --model yahma/llama-7b-hf \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save wanda_out \
    --save_model unstructured_sparsity_models/llama-7b-sparsity50
```
- `--model`: The identifier for the model on the Hugging Face model hub or local path.
- `--sparsity_ratio`: Denotes the percentage of weights to be pruned.
- `--sparsity_type`: Specifies the type of sparsity [`unstructured`, `2:4`, `4:8`].
- `--save`: Specifies the directory where the result will be stored.
- `--save_model`: Specifies the directory where the pruned LLM will be stored.

#### 2. Super-Adapter Training

Example command to train a super-adapter in the pruned LLaMA-7B using Shears:

```bash
CUDA_VISIBLE_DEVICES=${DEVICES} python run_math.py \
    --dataset_path datasets/math_10k.json \
    --model_name_or_path unstructured_sparsity_models/llama-7b-sparsity50 \
    --do_train \
    --do_test \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3 \
    --warmup_steps 100 \
    --optim adamw_torch \
    --fp16 \
    --output_dir trained_super_adapter/unified_math/shears-llama-7b-sparsity50 \
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

- `--lora`: the flag to enable LoRA
- `--lora_r`: LoRA attention dimension (the output size of LoRA A or the input size of LoRA B)
- `--lora_alpha`: the alpha parameter for LoRA scaling
- `--lora_dropout`: the dropout probability for LoRA layers
- `--target_modules`: the modules that will be added the LoRA adapter. 
- `--nncf_config`: the configuration of NNCF includes the search space for elastic adapters.


The command actives `--do_test`, i.e., after training it will test the heuristic sub-adapter on 4 math test datasets.
Note that we use the stage LR scheduler in NNCF, so the learning rate is set in the NNCF config file, instead of `learning_rate` of `TrainingArguments`.
Please refer to `running_commands` for all commands related to reproducing the results of the paper.

#### Reload the trained super-adapter for evaluation

Example command to reload a trained super-adapter for testing (heuristic):

```bash
CUDA_VISIBLE_DEVICES=${DEVICES} python run_math.py \
    --dataset_path None \
    --model_name_or_path unstructured_sparsity_models/llama-7b-sparsity50 \
    --do_test \
    --fp16 \
    --output_dir trained_super_adapter/unified_math/shears-llama-7b-sparsity50 \
    --lora \
    --lora_weights trained_super_adapter/shears-llama-7b-sparsity50 \
    --nncf_config nncf_config/unified_math/nncf_shears_llama_7b_sparsity50.json
```
