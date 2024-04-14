# :scissors:Shears: Fine-tuning For Efficient Models

Official implementation of [Shears: Unstructured Sparsity with Neural Low-rank Adapter Search](). :fire:

This repo contains the code for **Shears**, a practical and novel solution that generates efficient models fine-tuned for downstream-specific tasks for real-world applications. Please refer to our [paper]() for more details.

## News
- **[2024.03.07]**  Release training and inference code for **Shears V1**. :tada:

## Released Models 🤗

| Name                                                                                                                 | Super-network                                                                                                                                                                                  | Sparsity                                               | Train Data                                                         | Base Model
|----------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| ------------------------------------------------------ | ------------------------------------------------------------ | ------- | 
| [shears-llama-7b-50-math-heuristic](https://huggingface.co/IntelLabs/shears-llama-7b-50-math-heuristic) | [shears-llama-7b-50-math-super](https://huggingface.co/IntelLabs/shears-llama-7b-50-math-super) ([NNCF Config](./nncf_config/unified_math/nncf_shears_llama_7b_sparsity50.json))               | 50%           | [Unified Math](https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/ft-training_set/math_10k.json)         | [yahma/llama-7b-hf](https://huggingface.co/yahma/llama-7b-hf)
| [shears-llama-7b-50-commonsense-heuristic](https://huggingface.co/IntelLabs/shears-llama-7b-50-commonsense-heuristic) | [shears-llama-7b-50-commonsense-super](https://huggingface.co/IntelLabs/shears-llama-7b-50-commonsense-super) ([NNCF Config](./nncf_config/unified_commonsense/nncf_shears_llama_7b_sparsity50.json)) | 50%                                                                               | [Unified Commonsense](https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/ft-training_set/commonsense_170k.json) | [yahma/llama-7b-hf](https://huggingface.co/yahma/llama-7b-hf)
| [shears-llama-13b-50-math-heuristic](https://huggingface.co/IntelLabs/shears-llama-13b-50-math-heuristic) | [shears-llama-13b-50-math-super](https://huggingface.co/IntelLabs/shears-llama-13b-50-math-super)  ([NNCF Config](./nncf_config/unified_math/nncf_shears_llama_13b_sparsity50.json))           | 50%                                                                                      | [Unified Math](https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/ft-training_set/math_10k.json)      |  [yahma/llama-13b-hf](https://huggingface.co/yahma/llama-13b-hf)
| [shears-mpt-7b-50-gsm8k-heuristic](https://huggingface.co/IntelLabs/shears-mpt-7b-50-gsm8k-heuristic) | [shears-mpt-7b-50-gsm8k-super](https://huggingface.co/IntelLabs/shears-mpt-7b-50-gsm8k-super)      ([NNCF Config](./nncf_config/gsm8k/nncf_shears_mpt_7b_sparsity50.json))            | 50%                                                                                      | [GSM8K](https://huggingface.co/datasets/gsm8k)              | [mosaicml/mpt-7b](https://huggingface.co/mosaicml/mpt-7b)

## Overview

Efficiency Comparison (use LLaMA-13B as an example):

|                 Method                    | Sparsity | Non-zero Parameters | Accuracy |
|:-----------------------------------------:|:--------:|:-------------------:|:--------:|
| [LoRA](https://github.com/microsoft/LoRA) |    -     |        13.0B        |   51.1   |
|                  Shears                   |   50%    |        6.7B         |   50.9   |

By incorporating elastic LoRA adapters into a sparsified base model, Shears can fine-tune a language model without sacrificing the sparsity obtained from the original model weights. This produces sparse models with improvements or minor drops in accuracy and a fraction of the cost compared to other approaches. The increase in sparsity can result in a significant speedup when using runtimes that take advantage of these patterns. 

Overall, Shears has a well-designed, simple yet effective, powerful, and general pipeline that allows users to easily extend it to their desired scenarios/tasks, even audio and video. Feel free to try Shears for any downstream task with any model. :relaxed:

## Setup

Here is an installation script developed from scratch for **Shears**.

```
conda create -n shears -y python=3.10
conda activate shears

# install pytorch
pip install torch==2.1.2

# install dependencies
bash install.sh
```
Note: Please ignore the whitespace issues when applying the patch and running `install.sh`.

## Quick Start

### Inference

Please clone/download our released models from [HuggingFace](https://huggingface.co/IntelLabs) - checkout [Table](#released-models-) for model information, e.g.,
```bash
git lfs install
git clone https://huggingface.co/IntelLabs/shears-llama-7b-50-math-heuristic
```
Each Shears model contains the pretrained weights of the sparsified language model (`base_model/`) and the weights of the fine-tuned adapters (`adapter_model/`).
Denote the path to the downloaded model as `SHEARS_PATH`. The following code shows how to load the Shears model: 
```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model_path, adapter_model_path = f"{SHEARS_PATH}/base_model", f"{SHEARS_PATH}/adapter_model"
base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
model = PeftModel.from_pretrained(base_model, adapter_model_path)
```
Below is an example of generating the instruction-following responses for some math reasoning samples:
```bash
CUDA_VISIBLE_DEVICES=$DEVICES python example_math.py --model_path $SHEARS_PATH
```

### Training

#### Step 1. Unstructured Sparsifying

Before fine-tuning, Shears employs a simple but effective pruning approach [Wanda](https://arxiv.org/abs/2306.11695) to sparsify the language model, serving as the base model (frozen) for adapter training.
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
To implement the elastic adapter, we apply the BootstrapNAS feature supported in [OpenVINO™ NNCF](https://github.com/openvinotoolkit/nncf), which provides a suite of compression algorithms for neural network optimization.
Note that we use the stage LR scheduler in NNCF, so the learning rate is set in the NNCF config file instead of the `learning_rate` of `TrainingArguments`.

After training, the weights of the trained super-adapter will be saved in `--output_dir` directory.

### Evaluation

All evaluation datasets can be downloaded from [LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters).
Place them into the directory `datasets/`.
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
    --output_dir trained_super_adapter/unified_math/llama-7b-sparsity50-shears-math-adapter/results \
    --nncf_config nncf_config/unified_math/nncf_shears_llama_7b_sparsity50.json
```

Note that we can use the above command to test our released super-networks for result reproduction in the paper. 
Pass `${SHEARS_PATH}/base_model` to `--model_name_or_path`, `${SHEARS_PATH}/adapter_model` to `--lora_weights`, and the corresponding NNCF config to `--nncf_config` (see [Table](#released-models-)).
Note that the torch version we used in our experiments is `1.12.1+cu113`.

## Reproduce Results

Please refer to `running_commands` for all commands related to reproducing the paper's results.

- LLaMA with Math Reasoning tasks

| Model                 | Sparsity    | GSM8K | AQuA  | MAWPS | SVAMP | Average |
|-----------------------|-------------|-------|-------|-------|-------|---------|
| LLaMA-7B-LoRA         | -           | 37.5  | 18.9  | 79.0  | 52.1  | 46.9    |
| **LLaMA-7B-Shears**   | **40%**     | 36.8  | 19.7  | 83.2  | 47.7  | 46.9    |
| **LLaMA-7B-Shears**   | **50%**     | 36.1  | 22.0  | 78.6  | 44.5  | 45.3    |
| LLaMA-13B-LoRA        | -           | 47.5  | 18.5  | 83.6  | 54.6  | 51.1    |
| **LLaMA-13B-Shears**  | **40%**     | 48.3  | 21.3  | 83.2  | 55.2  | 52.0    |
| **LLaMA-13B-Shears**  | **50%**     | 45.1  | 22.0  | 83.2  | 53.3  | 50.9    |

- MPT with GSM8K

| Sparsity | 0%    | 40%  | 50%  | 60%  | 70%  |
|----------|-------|------|------|------|------|
| Accuracy | 36.1  | 35.7 | 33.4 | 30.4 | 22.8 |

- LLaMA with Commonsense Reasoning tasks

| Model                | Sparsity  | BoolQ   | PIQA   | SIQA   | HellaSwag  | WinoG  | ARC-e  | ARC-c   | OBQA   | Average  |
|----------------------|-----------|---------|--------|--------|------------|--------|--------|---------|--------|----------|
| ChatGPT              | -         | 73.1    | 85.4   | 68.5   | 78.5       | 66.1   | 89.8   | 79.9    | 74.8   | 77.0     |
| LLaMA-7B-LoRA	       | -         | 68.9    | 80.7   | 77.4   | 78.1       | 78.8   | 77.8   | 61.3    | 74.8   | 74.7     |
| **LLaMA-7B-Shears**	 | **40%**   | 67.0    | 79.9   | 76.7   | 80.1       | 78.6   | 76.9   | 62.3    | 77.8   | 74.9     |
| **LLaMA-7B-Shears**	 | **50%**   | 67.3    | 79.1   | 77.5   | 73.3       | 77.7   | 74.4   | 57.9    | 72.8   | 72.5     |

## Explore a trained super-network

To enhance exploration of the super-network trained using the Shears method, we provide an illustrative example `search/load_and_explore_supernet.ipynb`. This notebook demonstrates the direct loading of a Shears super-network and the extraction of various subnetworks from it. 
This facilitates users in applying their own search algorithms and evaluation metrics to extract subnetworks tailored to their specific requirements.

## Extract and save a subnetwork

After training, we obtain the weights of the super-adapter stored in `--output_dir` and activate different 
sub-networks using NNCF. However, once a particular sub-network we need is identified, it's no longer necessary to 
activate just that one sub-network using NNCF. Instead, what we need is a clean, pruned, and directly-loadable 
sub-network. For this purpose, we provide a function to extract/save any sub-adapter (please refer to the end of 
`search/load_and_explore_supernet.ipynb`). Below is an example to obtain the heuristic sub-adapter of a trained 
super-adapter:

```python
import os
from peft import PeftModel
from search.supernet import ShearsSuperNet
from transformers import AutoModelForCausalLM
from nncf import NNCFConfig

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH,trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL_PATH)
nncf_config = NNCFConfig.from_json(NNCF_CONFIG)
supernet = ShearsSuperNet.from_checkpoint(model, nncf_config, supernet_elasticity_path=None, supernet_weights_path=None)
supernet.activate_heuristic_subnet()
supernet.extract_and_save_active_sub_adapter(super_adapter_dir=ADAPTER_MODEL_PATH, sub_adapter_dir=os.path.join(ADAPTER_MODEL_PATH, "heuristic_adapter"))
```

We also provide a general script for subnet extraction that supports many of our methods, including Shears, 
see [extract/README.md]() for details. Moreover, we released some examples of heuristic subnetwork. 
Refer to them in [Table](#released-models-).

## Citation
If you find our Shears code and paper helpful, please kindly cite:
```bibtex
@article{munoz2024shears,
  title = {Shears: Unstructured Sparsity with Neural Low-rank Adapter Search},
  author={J. Pablo Munoz and Jinjie Yuan and Nilesh Jain},
  journal={The 2024 Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL-2024)},
  year={2024}
}
```

## Acknowledgement
This work benefits from the following repositories:

- LLaMA: [https://github.com/facebookresearch/llama](https://github.com/facebookresearch/llama)
- MPT: [https://www.mosaicml.com/mpt](https://www.mosaicml.com/mpt)
- Peft: [https://github.com/huggingface/peft](https://github.com/huggingface/peft)
- LLM-Adapters: [LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters)
- NNCF: [https://github.com/openvinotoolkit/nncf](https://github.com/openvinotoolkit/nncf)
- BootstrapNAS: [https://github.com/IntelLabs/Hardware-Aware-Automated-Machine-Learning/blob/main/BootstrapNAS](https://github.com/IntelLabs/Hardware-Aware-Automated-Machine-Learning/blob/main/BootstrapNAS)
