# Mamba-Shedder for Hymba Model

## Setup

Install the packages following: [Environment Setup](https://huggingface.co/nvidia/Hymba-1.5B-Base#step-1-environment-setup) and [Evaluation](https://huggingface.co/nvidia/Hymba-1.5B-Base#evaluation).
```
wget --header="Authorization: Bearer YOUR_HF_TOKEN" https://huggingface.co/nvidia/Hymba-1.5B-Base/resolve/main/setup.sh
bash setup.sh

git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
git fetch --all --tags
git checkout tags/v0.4.4
cd lm-evaluation-harness
pip install -e .
```

Apply the patch to the transformers to compress the Hymba model and copy `utils.py`:

```bash
MAMBA_SHEDDER_HYMBA_PATH=$PWD
git clone https://github.com/huggingface/transformers.git
pushd transformers
git checkout v4.47.0
git apply --ignore-space-change --ignore-whitespace ${MAMBA_SHEDDER_HYMBA_PATH}/patches/transformers-v4.47.0.patch
pip install -e .

pushd ${MAMBA_SHEDDER_HYMBA_PATH}
cp ../../utils.py .
```

## Run

### Evaluation before Pruning

```bash
python eval.py --model_path nvidia/Hymba-1.5B-Base
```

### Prune

Download the Hymba model and make some modifications to its modeling file:

```bash
git clone https://huggingface.co/nvidia/Hymba-1.5B-Base.git && cd Hymba-1.5B-Base && git checkout e1b7ee9
git apply --ignore-space-change --ignore-whitespace ../patches/hymba-e1b7ee9.patch && cd ..
```

An example command for [Hymba-1.5B-Base](https://huggingface.co/nvidia/Hymba-1.5B-Base) with Mamba-Shedder:

```bash
python prune.py \
  --model_path Hymba-1.5B-Base \
  --do_prune \
  --output_path <path to pruning results> \
  --num_block_pruning_steps 6 \
  --block_pruning_targets hymba_block \
  --num_width_pruning_steps 30 \
  --mlp_channel_group_size 512 \
  --weight_reorder \
  --importance_metric ppl \
  --calibration_dataset alpaca \
  --num_calibration_samples 256 \
  --do_eval
```

- `--model_path`: Specifies the path to the pre-trained Hymba model to be used for pruning.
- `--do_prune`: Indicates that the pruning process should be performed.
- `--output_path`: Defines the directory where the pruning and evaluation results will be saved.
- `--num_block_pruning_steps`: Sets the number of steps for block pruning.
- `--block_pruning_targets`: Specify the pruning targets, which can be one or more of [`self_attn`, `ssm`, `hymba_block`, `moe`]. If multiple targets are specified, separate them with commas, e.g., `self_attn,hymba_block`, which means both targets will be considered during the block pruning phase.
- `--num_width_pruning_steps`: Sets the number of steps for width pruning.
- `--mlp_channel_group_size`: Defines the number of channels for each group in MLP (MoE expert, FFN).
- `--weight_reorder`: Indicates that weight reordering should be performed in FFN.
- `--importance_metric`: Specifies the metric for calculating importance.
- `--calibration_dataset`: Defines the dataset to be used for calibration during pruning.
- `--num_calibration_samples`: Sets the number of samples to use for calibration during pruning.
- `--do_eval`: Indicates that the evaluation process should be performed after pruning.

Based on some experimental results, it is recommended to prune the Hymba Blocks only (i.e., setting `block_pruning_targets` to `hymba_block`, `num_width_pruning_steps` to 0 and unset `weight_reorder`), as it offers the best trade-off between efficiency and accuracy.

### Extract the Pruned Model

Extract the pruned model based on the optimal pruning configuration obtained from Mamba-Shedder. 
For more details, please refer to [here](./extract).

```bash
python extract/extract_hymba.py \
  --model_path Hymba-1.5B-Base \
  --weight_reorder \
  --pruned_model_config_file <path to pruning results>/pruning_config.json \
  --output_path <path to compressed model>
```

### Recovery Finetuning

After we have obtained the pruned model, we can use [Alpaca](https://huggingface.co/datasets/yahma/alpaca-cleaned) for recovery fine-tuning:

```bash
# Finetune the compressed Hymba
python recovery/finetune_hymba.py \
  --model_path <path to compressed model> \
  --do_train \
  --batch_size 4 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 3 \
  --learning_rate 3e-4 \
  --lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_target_modules in_proj,out_proj,down_proj,up_proj \
  --output_path <path to trained adapter> \
  --do_eval
```

For more details, please refer to [here](./recovery). 

## Results

All run commands and pruning results can be found in [here](./results).

## Released Pruned Models 🤗

Compressed Hymba models by Mamba-Shedder:

| Model                                                                                                         | Components Removed          | Recovery Tuning | Relative Acc. | 
|---------------------------------------------------------------------------------------------------------------|-----------------------------|-----------------|---------------|
| [Mamba-Shedder-Hymba-1.4B-Base](https://huggingface.co/IntelLabs/Mamba-Shedder-Hymba-1.4B-Base)               | 7 Hymba Blocks              | ✘               | 96.71%        |
| [Mamba-Shedder-Hymba-1.4B-Base-Alpaca](https://huggingface.co/IntelLabs/Mamba-Shedder-Hymba-1.4B-Base-Alpaca) | 7 Hymba Blocks              | ✔               | 99.84%        |

### Loading the compressed model for evaluation

```bash
python eval.py --model_path <path to compressed model>
```
