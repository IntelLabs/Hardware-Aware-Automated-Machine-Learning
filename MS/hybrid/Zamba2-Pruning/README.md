# Mamba-Shedder for Zamba2 Model

## Setup

Use the following instructions to create a virtual environment with the required dependencies.

```
# install dependencies
bash install.sh
```

## Run

### Preprocess

The purpose of this processing is to prune [Zamba2-2.7B](https://huggingface.co/Zyphra/Zamba2-2.7B) via Mamba-Shedder. 
Specifically, it aims to help calculate the actual pruning ratio for comparison of different methods and to facilitate width channel pruning.
Processing includes:

- Repeated the shared transformer blocks.
- Merged the lora adapter to the base model.
- Split `linear_fc1` to `linear_fc1_up` and `linear_fc1_gate`, in order to facilitate width pruning of `Gate` and `Up` (avoid [`torch.chunk()`](https://github.com/Zyphra/transformers_zamba2/blob/b258cb1d16a68ef3c38ac524201ce2b6e4f7f377/src/transformers/models/zamba2/modeling_zamba2.py#L842) when performs width pruning on a linear module)

```
huggingface-cli download Zyphra/Zamba2-2.7B --local-dir ./Zamba2-2.7B
python preprocess.py
```

### Evaluation before Pruning

```bash
python eval.py --model_path Zamba2-2.7B
```

### Prune

#### Mamba Block Pruning

An example command for Zamba2-2.7B with Mamba Block Pruning:

```bash
python prune.py \
  --model_path Zamba2-2.7B \
  --do_prune \
  --output_path <path to pruning results> \
  --prune_target mamba_block \
  --target_pruning_steps 10 \
  --importance_metric ppl \
  --calibration_dataset alpaca \
  --num_calibration_samples 256 \
  --do_eval
```

#### SSM Pruning

An example command for Zamba2-2.7B with SSM Pruning:

```bash
python prune.py \
  --model_path Zamba2-2.7B \
  --do_prune \
  --output_path <path to pruning results> \
  --prune_target ssm \
  --target_pruning_steps 20 \
  --importance_metric ppl \
  --calibration_dataset alpaca \
  --num_calibration_samples 256 \
  --do_eval
```

#### Mamba and Transformer Pruning

The pruning process is divided into the following three stages:

1. Mamba-2 block, MHA and MLP pruning
2. MLP channel pruning
3. SSM pruning

```bash
# Pruning steps for Mamba block, MHA and MHA pruning
target_block_pruning_steps=3
# Pruning steps for MLP channel pruning
target_width_pruning_steps=20
# Pruning steps for SSM pruning
target_ssm_pruning_steps=18

python prune_hybrid.py \
  --model_path Zamba2-2.7B \
  --output_path <path to pruning results> \
  --do_prune \
  --target_block_pruning_steps ${target_block_pruning_steps} \
  --target_width_pruning_steps ${target_width_pruning_steps} \
  --target_ssm_pruning_steps ${target_ssm_pruning_steps} \
  --mlp_channel_group_size 1024 \
  --importance_metric ppl \
  --calibration_dataset alpaca \
  --num_calibration_samples_block 256 \
  --num_calibration_samples_width 256 \
  --num_calibration_samples_ssm 256 \
  --do_eval
```

### Extract the Pruned Model

Extract the pruned model based on the optimal pruning configuration obtained from Mamba-Shedder:

```bash
python extract/extract_zamba2.py \
  --model_path Zamba2-2.7B \
  --output_path <path to compressed model> \
  --pruned_model_config_file <path to pruning result>/pruning_config.json # Or specify the config file of a pruning step from the `pruned_model_configs` folder, e.g., <path to pruning result>/pruned_model_configs/config.ssm.${eval_step}.json
```

### Recovery Fine-tuning

After we have obtained the pruned model, we can use the Alpaca dataset for recovery fine-tuning:

```bash
# Finetune the compressed Zamba-2
python recovery/finetune_zamba2.py \
  --model_path <path to compressed model> \
  --do_train \
  --batch_size 32 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 2 \
  --learning_rate 1e-4 \
  --lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_target_modules in_proj.0,out_proj \
  --output_path <path to trained adapter> \
  --do_eval
```

## Results

All run commands and pruning results can be found in [here](./results).

### Loading the compressed model for evaluation

```bash
python eval.py --model_path <path to compressed model>
```
