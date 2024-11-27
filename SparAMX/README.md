# SparAMX: Accelerating Compressed LLMs Token Generation on AMX-powered CPUs

Official implementation of [SparAMX: Accelerating Compressed LLMs Token Generation on AMX-powered CPUs](https://arxiv.org/pdf/...).

This repo contains the code for **SparAMX**, a set of open-source customized sparse kernels that can speed up any PyTorch model by automatically replacing all linear layers with our customized layer. Furthermore, we demonstrate for the first time the use of unstructured sparsity in the attention computation and achieving \textbf{1.14}$\times$ speedup over the current systems without compromising accuracy.

| Stock PyTorch | TLD |
|:-----------:|:-----------:|
| <img src="Videos/stock.gif" alt="Stock" width="100%"> | <img src="Videos/tld.gif" alt="TLD" width="100%"> |

# torch-custom-linear
custom implementation of linear through torch extension

### Dependency
```pip install -r requirements.txt```

### Build & Install Custom Kernel
```
python setup.py install
```

### Run Experiments Example

Please make sure you're logged in to HuggingFace through the CLI if you'll be using a private model.

You need to define the experiments you want to run in `generate_experiments.py` then run 
```
python generate_experiments.py
```

A file `experiments.csv` is generated. Modify it if needed. After that run
```
./run_experiments.sh
```

Your results will be saved inside folder `experiment_results/YYYY-MM-DD_HH-MM-SS`.


### Clean Up (Important!)
```bash
python setup.py clean 
# setup.py is scripted to remove many previous built artefacts
# This does not uninstall. 
# It may be a good idea to clean before a build, just so recompile all required codes.
```

### Uninstall
```bash
pip uninstall custom_linear
```

### Using Built Extension
```python
import torch #why we need this? torch load libc10.so
import custom_linear
dir(custom_linear) # method binded to kernel will be shown e.g. dense_linear

ic=2
oc=6
bs=3

X=torch.rand(bs, ic)
W=torch.rand(oc, ic)
b=torch.rand(oc)

custom_linear.dense_linear(X, W, b)

custom_linear.dense_linear(X, W)

custom_linear.dense_linear(X, W, None)
```

### Using Built Extension with subclass of nn.Module
see ```layer/dense_linear.py```
```python
import torch
from layer.dense_linear import DenseLinear

d=2
N=3
x = torch.rand(N, d)

one_fc_layer = DenseLinear(in_features=d, out_features=4*d, bias=True)

one_fc_layer(x)
```

### Wrapping MLP Linear layers of TinyLlama
```python llm_pipeline.py```

```
#Wrapped model
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(32000, 2048)
    (layers): ModuleList(
      (0-21): 22 x LlamaDecoderLayer(
        (self_attn): LlamaSdpaAttention(
          (q_proj): Linear(in_features=2048, out_features=2048, bias=False)
          (k_proj): Linear(in_features=2048, out_features=256, bias=False)
          (v_proj): Linear(in_features=2048, out_features=256, bias=False)
          (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): CustomLinear(in_features=2048, out_features=5632, bias=False)
          (up_proj): CustomLinear(in_features=2048, out_features=5632, bias=False)
          (down_proj): CustomLinear(in_features=5632, out_features=2048, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
    )
    (norm): LlamaRMSNorm()
  )
  (lm_head): Linear(in_features=2048, out_features=32000, bias=False)
)
```

**#TODO(Ahmed): Include bibtex.** 
