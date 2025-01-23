## Extract the Compressed Model from Mamba-Shedder

The final compressed model can be extracted based on the optimal pruning configuration obtained from Mamba-Shedder.

```bash
# Hymba
python extract/extract_hymba.py \
  --model_path Hymba-1.5B-Base \
  --weight_reorder \
  --output_path .<path to pruned model> \
  --pruned_model_config_file <path to pruning result>/pruning_config.json # Or specify the config file of a pruning step from the `pruned_model_configs` folder, e.g., <path to pruning result>/pruned_model_configs/config.mlp_width.${eval_step}.json

```

- `model_path`: Path to the pre-trained model.
- `weight_reorder`: Flag to indicate whether to perform weight reorder in MLP.
- `pruned_model_config_file`: JSON file for the pruned model configuration.
- `output_path`: Directory to save the compressed model.
