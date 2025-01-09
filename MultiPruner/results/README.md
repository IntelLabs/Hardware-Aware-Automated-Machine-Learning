## Run commands

Following are the commands of MultiPruner on various LLMs.

### Llama

#### Llama-3.2-3B

```bash
python run_multipruner.py \
  --model_path meta-llama/Llama-3.2-3B \
  --output_path <path to pruning results> \
  --weight_reorder \
  --do_prune \
  --target_ratio <pruning ratio> \
  --pruning_distribution 10:90:0 \
  --mlp_channel_group_size 768 \
  --attn_channel_group_size 384 \
  --importance_metric ppl \
  --calibration_dataset alpaca \
  --num_calibration_samples_block 256 \
  --num_calibration_samples_width 128 \
  --do_eval
```

#### Llama-3.1-8B

```bash
python run_multipruner.py \
  --model_path meta-llama/Llama-3.1-8B \
  --output_path <path to pruning results> \
  --weight_reorder \
  --do_prune \
  --target_ratio <pruning ratio> \
  --pruning_distribution 30:70:0 \
  --mlp_channel_group_size 1024 \
  --attn_channel_group_size 512 \
  --importance_metric ppl \
  --calibration_dataset alpaca \
  --num_calibration_samples_block 256 \
  --num_calibration_samples_width 128 \
  --do_eval
```

#### Meta-Llama-3-8B

```bash
python run_multipruner.py \
  --model_path meta-llama/Meta-Llama-3-8B \
  --output_path <path to pruning results> \
  --weight_reorder \
  --do_prune \
  --target_ratio <pruning ratio> \
  --pruning_distribution 20:80:0 \
  --mlp_channel_group_size 1024 \
  --attn_channel_group_size 512 \
  --importance_metric ppl \
  --calibration_dataset alpaca \
  --num_calibration_samples_block 256 \
  --num_calibration_samples_width 128 \
  --do_eval
```

#### Llama-2-7B

```bash
python run_multipruner.py \
  --model_path meta-llama/Llama-2-7b-hf \
  --output_path <path to pruning results> \
  --weight_reorder \
  --do_prune \
  --target_ratio <pruning ratio> \
  --pruning_distribution 44:52:4 \
  --mlp_channel_group_size 1024 \
  --attn_channel_group_size 128 \
  --importance_metric ppl \
  --calibration_dataset alpaca \
  --num_calibration_samples_block 256 \
  --num_calibration_samples_width 128 \
  --do_eval
```

#### Llama-2-13B

```bash
python run_multipruner.py \
  --model_path meta-llama/Llama-2-13b-hf \
  --output_path <path to pruning results> \
  --weight_reorder \
  --do_prune \
  --target_ratio <pruning ratio> \
  --pruning_distribution 44:52:4 \
  --mlp_channel_group_size 1024 \
  --attn_channel_group_size 128 \
  --importance_metric ppl \
  --calibration_dataset alpaca \
  --num_calibration_samples_block 256 \
  --num_calibration_samples_width 128 \
  --do_eval \
  --batch_size 16
```

### Qwen

#### Qwen2.5-7B

```bash
python run_multipruner.py \
  --model_path Qwen/Qwen2.5-7B \
  --output_path <path to pruning results> \
  --weight_reorder \
  --do_prune \
  --target_ratio <pruning ratio> \
  --pruning_distribution 50:50:0 \
  --mlp_channel_group_size 1024 \
  --attn_channel_group_size 896 \
  --importance_metric ppl \
  --calibration_dataset alpaca \
  --num_calibration_samples_block 256 \
  --num_calibration_samples_width 128 \
  --do_eval
```

#### Qwen1.5-7B

```bash
python run_multipruner.py \
  --model_path Qwen/Qwen1.5-7B \
  --output_path <path to pruning results> \
  --weight_reorder \
  --do_prune \
  --target_ratio <pruning ratio> \
  --pruning_distribution 44:52:4 \
  --mlp_channel_group_size 1024 \
  --attn_channel_group_size 128 \
  --importance_metric ppl \
  --calibration_dataset alpaca \
  --num_calibration_samples_block 256 \
  --num_calibration_samples_width 128 \
  --do_eval
```

#### Qwen1.5-14B

```bash
python run_multipruner.py \
  --model_path Qwen/Qwen1.5-14B \
  --output_path <path to pruning results> \
  --weight_reorder \
  --do_prune \
  --target_ratio <pruning ratio> \
  --pruning_distribution 44:52:4 \
  --mlp_channel_group_size 2048 \
  --attn_channel_group_size 128 \
  --importance_metric ppl \
  --calibration_dataset alpaca \
  --num_calibration_samples_block 256 \
  --num_calibration_samples_width 128 \
  --do_eval \
  --batch_size 16
```

### Baichuan

#### Baichuan2-7B-Base

To enable pruning of Query, Key, and Value, we have deconstructed the linear module `W_pack` (which combines QKV into a single linear layer) in [Baichuan2-7B-Base](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base).

```bash
python run_multipruner.py \
  --model_path <path to processed baichuan model> \
  --output_path <path to pruning results> \
  --weight_reorder \
  --do_prune \
  --target_ratio <pruning ratio> \
  --pruning_distribution 44:52:4 \
  --mlp_channel_group_size 1024 \
  --attn_channel_group_size 128 \
  --importance_metric ppl \
  --calibration_dataset alpaca \
  --num_calibration_samples_block 256 \
  --num_calibration_samples_width 256 \
  --do_eval
```

#### Baichuan2-13B-Base

```bash
python run_multipruner.py \
  --model_path <path to processed baichuan model> \
  --output_path <path to pruning results> \
  --do_prune \
  --target_ratio <pruning ratio> \
  --pruning_distribution 44:52:4 \
  --mlp_channel_group_size 1024 \
  --attn_channel_group_size 128 \
  --importance_metric ppl \
  --calibration_dataset alpaca \
  --num_calibration_samples_block 256 \
  --num_calibration_samples_width 128 \
  --do_eval \
  --batch_size 16
```

## Run commands (Recovery Tuning for Compressed Llama-2)

#### Pruning ratio of 12%

```bash
python recovery/finetune.py \
  --model_path IntelLabs/MultiPruner-Llama-2-5.9b \
  --do_train \
  --batch_size 8 \
  --gradient_accumulation_steps 4 \
  --max_steps 2000 \
  --learning_rate 1e-4 \
  --lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
  --output_path <path to trained adapter> \
  --do_eval
```

#### Pruning ratio of 15%

```bash
python recovery/finetune.py \
  --model_path IntelLabs/MultiPruner-Llama-2-5.7b \
  --do_train \
  --batch_size 8 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 2 \
  --learning_rate 1e-4 \
  --lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
  --output_path <path to trained adapter> \
  --do_eval
```

#### Pruning ratio of 18%

```bash
python recovery/finetune.py \
  --model_path IntelLabs/MultiPruner-Llama-2-5.5b \
  --do_train \
  --batch_size 8 \
  --gradient_accumulation_steps 4 \
  --max_steps 3000 \
  --learning_rate 1e-4 \
  --lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
  --output_path <path to trained adapter> \
  --do_eval
```

#### Pruning ratio of 22%

```bash
python recovery/finetune.py \
  --model_path IntelLabs/MultiPruner-Llama-2-5.3b \
  --do_train \
  --batch_size 8 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 2 \
  --learning_rate 1e-4 \
  --lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj \
  --output_path <path to trained adapter> \
  --do_eval
```

#### The impact results of Recovery Tuning

| Method                      | Pruning Ratio | Acc. (%)  | Acc. Drop  | Relative Acc. |
|-----------------------------|---------------|-----------|------------|---------------|
| Dense                       | -             | 68.96     | -          | 100.00%       |
| MultiPruner w/o finetune    | 12%           | 66.48     | -2.48%     | 96.40%        |
| **MultiPruner w finetune**  | 12%           | **68.28** | **-0.68%** | **99.01%**    |
| MultiPruner w/o finetune    | 15%           | 65.26     | -3.70%     | 94.63%        |
| **MultiPruner w finetune**  | 15%           | **67.41** | **-1.55%** | **97.75%**    |
| MultiPruner w/o finetune    | 18%           | 64.20     | -4.76%     | 93.10%        |
| **MultiPruner w finetune**  | 18%           | **66.16** | **-2.80%** | **95.94%**    |
| MultiPruner w/o finetune    | 22%           | 62.83     | -6.13%     | 91.11%        |
| **MultiPruner w finetune**  | 22%           | **64.18** | **-4.78%** | **93.07%**    |
