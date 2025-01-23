### Recovery Fine-tuning after Pruning

After obtaining the pruned model ([extract](../extract)), we can finetune it to recover accuracy. 
The dataset used for finetuning is [Alpaca](https://huggingface.co/datasets/yahma/alpaca-cleaned). 
Here is an example command:

```bash
# Finetune the compressed Hymba
python finetune_hymba.py \
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

# after fine-tuning, merge the adapter to the compressed model
python merge.py \
  --base_model_path <path to compressed model> \
  --adapter_model_path <path to trained adapter> \
  --output_path <path to finetuned compressed model> \

```
