### Recovery Fine-tuning after Pruning

After obtaining the compressed model ([here](../extract)), we can finetune it to recover accuracy. 
The dataset used for finetuning is [Alpaca](https://huggingface.co/datasets/yahma/alpaca-cleaned). 
Here is an example command:

```bash
python recovery/finetune.py \
  --model_path <path to compressed model> \
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

After fine-tuning, we can merge the trained adapter into the pruned base model.

```bash
python recovery/merge.py \
    --base_model_path <path to compressed model> \
    --adapter_model_path <path to trained adapter> \
    --output_path <path to finetuned pruned model>
```
