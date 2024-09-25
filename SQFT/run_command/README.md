### Run command

Prepare the datasets from [LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters) for our math instruction tuning Setting.
```bash
git clone https://github.com/AGI-Edgerunners/LLM-Adapters.git
mv LLM-Adapters/dataset/ datasets/
mv LLM-Adapters/ft-training_set/* datasets/
```

#### Llama-3

```bash
bash run_command/llama-3-8b/sparse_quantization.sh $SPARSITY # e.g., SPARSITY=50
bash run_command/llama-3-8b/run.sh $SPARSITY
```

#### Mistral-v0.3

```bash
bash run_command/mistral-7b-v0.3/sparse_quantization.sh $SPARSITY
bash run_command/mistral-7b-v0.3/run.sh $SPARSITY $TASK
```
Supported tasks: `gsm8k` and `math`.

#### Phi-3

```bash
bash run_command/phi-3-mini-4k-instruct/sparse_quantization.sh $SPARSITY
bash run_command/phi-3-mini-4k-instruct/run.sh $SPARSITY $TASK  
```
Supported tasks: `cs` and `math`.

Note that the results presented in the paper were obtained using an older environment setup. 
Specifically, we utilized torch version `2.1.2`, transformers version `4.39.1`, and NNCF with commit ID `544d5141`. 
The training was conducted on a single Tesla V100-SXM2-32GB GPU.
