## Extract the Compressed Model from MultiPruner

The final compressed model can be extracted based on the optimal pruning configuration obtained from **MultiPruner**.
Here is an example command for the compressed Llama-2-7B:

```bash
python extract/extract_model.py \
  --model_path meta-llama/Llama-2-7b-hf \
  --weight_reorder \
  --pruned_model_config_file <path to pruning result>/pruning_config.json \
  --output_path <path to compressed model>
```

- `model_path`: Path to the pre-trained model.
- `weight_reorder`: Flag to indicate whether to perform weight reordering.
- `pruned_model_config_file`: JSON file for the pruned model configuration.
- `output_path`: Directory to save the compressed model.
