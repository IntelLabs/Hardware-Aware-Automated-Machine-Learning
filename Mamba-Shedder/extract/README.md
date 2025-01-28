## Extract the Compressed Model from Mamba-Shedder

The final compressed model can be extracted based on the optimal pruning configuration obtained from Mamba-Shedder.

```bash
# Mamba-1 (Mamba Block Pruning)
python extract/extract_mamba.py \
  --model_path state-spaces/mamba-2.8b \
  --output_path <path to pruned model> \
  --pruned_model_config_file <path to pruning result>/pruning_config.json # Or specify the config file of a pruning step from the `pruned_model_configs` folder, e.g., <path to pruning result>/pruned_model_configs/config.mamba_block.${eval_step}.json
  
# Mamba-2 (SSM Pruning)
python extract/extract_mamba.py \
  --model_path state-spaces/mamba2-2.7b \ 
  --output_path <path to pruned model> \
  --pruned_model_config_file <path to pruning result>/pruning_config.json # Or specify the config file of a pruning step from the `pruned_model_configs` folder, e.g., <path to pruning result>/pruned_model_configs/config.ssm.${eval_step}.json
  ```

- `model_path`: Path to the pre-trained model.
- `pruned_model_config_file`: JSON file for the pruned model configuration.
- `output_path`: Directory to save the compressed model.
