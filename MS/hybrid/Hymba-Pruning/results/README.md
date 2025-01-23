## Run Command (Hymba)

Here are the commands to reproduce the main results of the paper.

### Hymba-1.5B

```bash
pruning_result_path=results/hymba-1.5b-base

python prune.py \
  --model_path Hymba-1.5B-Base \
  --do_prune \
  --output_path ${pruning_result_path} \
  --num_block_pruning_steps 8 \
  --block_pruning_targets hymba_block \
  --importance_metric ppl \
  --calibration_dataset alpaca \
  --num_calibration_samples 256 \
 
for eval_step in 5 6 7; do
    python prune.py \
      --model_path Hymba-1.5B-Base \
      --output_path ${pruning_result_path} \
      --do_eval \
      --pruned_model_config_file ${pruning_result_path}/pruned_model_configs/config.hymba_block.${eval_step}.json
done
```
