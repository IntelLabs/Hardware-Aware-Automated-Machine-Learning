## Run Command (Mamba and Mamba2)

Here are the commands to reproduce the main results of the paper.

### Mamba-2.8B

```bash
pruning_result_path=results/mamba-2.8b

# Mamba Block Pruning
python prune.py \
  --model_path state-spaces/mamba-2.8b \
  --prune_target mamba_block \
  --output_path ${pruning_result_path} \
  --do_prune \
  --target_pruning_steps 14 \
  --importance_metric ppl \
  --calibration_dataset alpaca \
  --num_calibration_samples 256

# Evaluation for different steps
for eval_step in 6 13; do
    python prune.py \
      --model_path state-spaces/mamba-2.8b \
      --output_path ${pruning_result_path} \
      --do_eval \
      --pruned_model_config_file ${pruning_result_path}/pruned_model_configs/config.mamba_block.${eval_step}.json
done
```

### Mamba2-2.7B

```bash
pruning_result_path=results/mamba2-2.7b

# SSM Pruning
python prune.py \
  --model_path state-spaces/mamba2-2.7b \
  --prune_target ssm \
  --output_path ${pruning_result_path} \
  --do_prune \
  --target_pruning_steps 24 \
  --importance_metric ppl \
  --calibration_dataset alpaca \
  --num_calibration_samples 256 
 
# Evaluation for different steps
for eval_step in 15 19 21 23; do
    python prune.py \
      --model_path state-spaces/mamba2-2.7b \
      --output_path ${pruning_result_path} \
      --do_eval \
      --pruned_model_config_file ${pruning_result_path}/pruned_model_configs/config.ssm.${eval_step}.json
done
```
