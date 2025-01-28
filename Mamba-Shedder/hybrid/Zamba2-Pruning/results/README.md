## Run Command (Zamba2)

Here are the commands to reproduce the main results of the paper.

### Zamba2-2.7B

#### Pruning Ratio: 10%

```bash
pruning_result_path=results/zamba2-2.7b

# Multi-granularity Pruning
python prune_hybrid.py \
  --model_path Zamba2-2.7B \
  --output_path ${pruning_result_path} \
  --do_prune \
  --target_block_pruning_steps 3 \
  --target_width_pruning_steps 20 \
  --target_ssm_pruning_steps 18 \
  --mlp_channel_group_size 1024 \
  --importance_metric ppl \
  --calibration_dataset alpaca \
  --num_calibration_samples_block 256 \
  --num_calibration_samples_width 256 \
  --num_calibration_samples_ssm 256

# Evaluation: w/o SSM Pruning
python prune_hybrid.py \
  --model_path Zamba2-2.7B \
  --output_path ${pruning_result_path} \
  --do_eval \
  --pruned_model_config_file ${pruning_result_path}/pruned_model_configs/config.mlp_width.22.json
 
# Evaluation: w/ SSM Pruning
python prune_hybrid.py \
  --model_path Zamba2-2.7B \
  --output_path ${pruning_result_path} \
  --do_eval \
  --pruned_model_config_file ${pruning_result_path}/pruned_model_configs/config.ssm.40.json
```

#### Pruning Ratio: 15%

```bash
pruning_result_path=results/zamba2-2.7b

# Multi-granularity Pruning
python prune_hybrid.py \
  --model_path Zamba2-2.7B \
  --output_path ${pruning_result_path} \
  --do_prune \
  --target_block_pruning_steps 8 \
  --target_width_pruning_steps 20 \
  --target_ssm_pruning_steps 18 \
  --mlp_channel_group_size 1024 \
  --importance_metric ppl \
  --calibration_dataset alpaca \
  --num_calibration_samples_block 256 \
  --num_calibration_samples_width 256 \
  --num_calibration_samples_ssm 256

# Evaluation: w/o SSM Pruning
python prune_hybrid.py \
  --model_path Zamba2-2.7B \
  --output_path ${pruning_result_path} \
  --do_eval \
  --pruned_model_config_file ${pruning_result_path}/pruned_model_configs/config.mlp_width.27.json
 
# Evaluation: w/ SSM Pruning
python prune_hybrid.py \
  --model_path Zamba2-2.7B \
  --output_path ${pruning_result_path} \
  --do_eval \
  --pruned_model_config_file ${pruning_result_path}/pruned_model_configs/config.ssm.45.json
```
