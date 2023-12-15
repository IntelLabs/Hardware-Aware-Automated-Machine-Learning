# LoNASv2 - Inference

## Setup

Step 1: Create a new conda environment
```
conda create -n lonasv2-infer python=3.10
conda activate lonasv2-infer
```
Step 2: Install relevant packages
```
# Pytorch
pip install torch==2.0.1

# HuggingFace - Transformers
git clone https://github.com/huggingface/transformers.git transformers_lonas && cd transformers_lonas
git checkout v4.31.0
git apply --ignore-space-change --ignore-whitespace /path/to/transformers-modifications-for-lonasv2-inference-usage.patch
pip install -e . && cd ..

# HuggingFace - Peft
git clone https://github.com/huggingface/peft.git peft_lonas && cd peft_lonas
git checkout v0.5.0
git apply --ignore-space-change --ignore-whitespace /path/to/peft-modifications-for-lonasv2-inference-usage.patch
pip install -e . && cd ..

# Others
pip install datasets accelerate sentencepiece tokenizers==0.13.3
```

## Usage

### Model

Two sample models are available at: 

- [llama7b-nas-heu-unmerged-75](https://huggingface.co/jpablomch/llama7b-nas-heu-unmerged-75)
- [llama7b-nas-min-unmerged-80](https://huggingface.co/jpablomch/llama7b-nas-min-unmerged-80)

- Example to load unmerged model
```python
import torch
from transformers import (
    AutoModelForCausalLM,
    LlamaTokenizer,
)
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("jpablomch/llama7b-nas-heu-unmerged-75",
    torch_dtype=torch.float16,
    device_map={"": 0},
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(
    base_model, "jpablomch/llama7b-nas-heu-unmerged-75-adapter",
    torch_dtype=torch.float16,
    device_map={"": 0}
)
... 
```

### DeepSparse for Inference Speedup

#### 1. Export to ONNX

install optimum:

```
# HuggingFace - optimum
git clone https://github.com/huggingface/optimum.git optimum_lonas && cd optimum_lonas
git checkout e840d21
git apply --ignore-space-change --ignore-whitespace /path/to/optimum-modifications-for-lonasv2-inference-usage.patch
pip install -e . && cd ..
```

```
# ONNX
pip install onnxruntime
```


- Unmerged model (e.g, LLaMA):
```python
import os
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import LlamaTokenizer

# Load a model from transformers attached adapters and export it to ONNX
pretrained_model_path = os.path.join(MODEL_PATH, "pretrained_weights")
adapter_path = os.path.join(MODEL_PATH, "adapter_weights")
ort_model = ORTModelForCausalLM.from_pretrained(pretrained_model_path, adapter_path=adapter_path, export=True)
tokenizer = LlamaTokenizer.from_pretrained(pretrained_model_path)

# Save the onnx model and tokenizer
ort_model.save_pretrained(SAVE_DIRECTORY)
tokenizer.save_pretrained(SAVE_DIRECTORY)
```

#### 2. Text Generation using DeepSparse

install DeepSparse:

```
git clone https://github.com/neuralmagic/deepsparse.git deepsparse_lonas && cd deepsparse_lonas
git checkout e2f2305
git apply --ignore-space-change --ignore-whitespace /path/to/deepsparse-modifications-for-lonasv2-inference-usage.patch
pip install -e . && cd ..
```

```python
from deepsparse import TextGeneration

pipeline = TextGeneration(model=YOUR_ONNX_PATH, sequence_length=512)
prompt = "Question: Marty has 100 centimeters of ribbon that he must cut into 4 equal parts. Each of the cut parts must be divided into 5 equal parts. How long will each final cut be?"
output = pipeline(prompt=prompt)
print(output.generations[0].text)
# check latency time
print(pipeline.timer_manager.times)
```


