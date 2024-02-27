# Shears Quantization - Internal

Implementation of Shears post-training quantization.

### Setup

- Create a new conda environment
```
conda create -n shears_quant python=3.10
conda activate shears_quant
```
- Install relevant packages
```
pip install torch==1.12.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers==4.37.2 openvino==2023.3.0 nncf==2.8.1 onnx==1.14.1
pip install --no-deps peft==0.5.0

# optimum-intel
git clone https://github.com/huggingface/optimum-intel.git && cd optimum-intel
git checkout v1.15.0
git apply --ignore-space-change --ignore-whitespace /PATH/TO/optimum-intel-modifications-for-shears-quantization-usage.patch
python -m pip install "optimum-intel[openvino]" . && cd ..
```

### Run

Shears model example: `/data1/jinjieyu/shears_examples/llama7b-sparsity40-shears/`

#### Obtain quantized models (openvino, INT8_ASYM):

- LLaMA-7B
```bash
python export_ov_compressed_model.py \
    --model_name_or_path yahma/llama-7b-hf \
    --output_dir llama7b-ov-compressed
```

- Unstructured sparsified LLaMA-7B
```bash
python export_ov_compressed_model.py \
    --model_name_or_path llama7b-sparsity40-shears/pretrained_weights \
    --output_dir llama7b-sparsity40-ov-compressed
```

- LLaMA-7B + Shears
```bash
python export_ov_compressed_model.py \
    --model_name_or_path llama7b-sparsity40-shears/pretrained_weights \
    --adapter_path llama7b-sparsity40-shears/adapter_weights/maximal_subnetwork \
    --output_dir llama7b-sparsity40-shears-ov-compressed
```

#### Use OV benchmark app to measure the performance

- LLaMA-7B (INT8_ASYM)

```bash
benchmark_app -m llama7b-ov-compressed/openvino_model.xml -data_shape input_ids[1,1],attention_mask[1,512],position_ids[1,512],past_key_values.0.value[1,32,511,128],past_key_values.0.key[1,32,511,128],past_key_values.1.value[1,32,511,128],past_key_values.1.key[1,32,511,128],past_key_values.2.value[1,32,511,128],past_key_values.2.key[1,32,511,128],past_key_values.3.value[1,32,511,128],past_key_values.3.key[1,32,511,128],past_key_values.4.value[1,32,511,128],past_key_values.4.key[1,32,511,128],past_key_values.5.value[1,32,511,128],past_key_values.5.key[1,32,511,128],past_key_values.6.value[1,32,511,128],past_key_values.6.key[1,32,511,128],past_key_values.7.value[1,32,511,128],past_key_values.7.key[1,32,511,128],past_key_values.8.value[1,32,511,128],past_key_values.8.key[1,32,511,128],past_key_values.9.value[1,32,511,128],past_key_values.9.key[1,32,511,128],past_key_values.10.value[1,32,511,128],past_key_values.10.key[1,32,511,128],past_key_values.11.value[1,32,511,128],past_key_values.11.key[1,32,511,128],past_key_values.12.value[1,32,511,128],past_key_values.12.key[1,32,511,128],past_key_values.13.value[1,32,511,128],past_key_values.13.key[1,32,511,128],past_key_values.14.value[1,32,511,128],past_key_values.14.key[1,32,511,128],past_key_values.15.value[1,32,511,128],past_key_values.15.key[1,32,511,128],past_key_values.16.value[1,32,511,128],past_key_values.16.key[1,32,511,128],past_key_values.17.value[1,32,511,128],past_key_values.17.key[1,32,511,128],past_key_values.18.value[1,32,511,128],past_key_values.18.key[1,32,511,128],past_key_values.19.value[1,32,511,128],past_key_values.19.key[1,32,511,128],past_key_values.20.value[1,32,511,128],past_key_values.20.key[1,32,511,128],past_key_values.21.value[1,32,511,128],past_key_values.21.key[1,32,511,128],past_key_values.22.value[1,32,511,128],past_key_values.22.key[1,32,511,128],past_key_values.23.value[1,32,511,128],past_key_values.23.key[1,32,511,128],past_key_values.24.value[1,32,511,128],past_key_values.24.key[1,32,511,128],past_key_values.25.value[1,32,511,128],past_key_values.25.key[1,32,511,128],past_key_values.26.value[1,32,511,128],past_key_values.26.key[1,32,511,128],past_key_values.27.value[1,32,511,128],past_key_values.27.key[1,32,511,128],past_key_values.28.value[1,32,511,128],past_key_values.28.key[1,32,511,128],past_key_values.29.value[1,32,511,128],past_key_values.29.key[1,32,511,128],past_key_values.30.value[1,32,511,128],past_key_values.30.key[1,32,511,128],past_key_values.31.value[1,32,511,128],past_key_values.31.key[1,32,511,128] -t 120 -hint latency -infer_precision f32
```

- Unstructured sparsified LLaMA-7B (INT8_ASYM)
```bash
benchmark_app -m llama7b-sparsity40-ov-compressed/openvino_model.xml -data_shape input_ids[1,1],attention_mask[1,512],position_ids[1,512],past_key_values.0.value[1,32,511,128],past_key_values.0.key[1,32,511,128],past_key_values.1.value[1,32,511,128],past_key_values.1.key[1,32,511,128],past_key_values.2.value[1,32,511,128],past_key_values.2.key[1,32,511,128],past_key_values.3.value[1,32,511,128],past_key_values.3.key[1,32,511,128],past_key_values.4.value[1,32,511,128],past_key_values.4.key[1,32,511,128],past_key_values.5.value[1,32,511,128],past_key_values.5.key[1,32,511,128],past_key_values.6.value[1,32,511,128],past_key_values.6.key[1,32,511,128],past_key_values.7.value[1,32,511,128],past_key_values.7.key[1,32,511,128],past_key_values.8.value[1,32,511,128],past_key_values.8.key[1,32,511,128],past_key_values.9.value[1,32,511,128],past_key_values.9.key[1,32,511,128],past_key_values.10.value[1,32,511,128],past_key_values.10.key[1,32,511,128],past_key_values.11.value[1,32,511,128],past_key_values.11.key[1,32,511,128],past_key_values.12.value[1,32,511,128],past_key_values.12.key[1,32,511,128],past_key_values.13.value[1,32,511,128],past_key_values.13.key[1,32,511,128],past_key_values.14.value[1,32,511,128],past_key_values.14.key[1,32,511,128],past_key_values.15.value[1,32,511,128],past_key_values.15.key[1,32,511,128],past_key_values.16.value[1,32,511,128],past_key_values.16.key[1,32,511,128],past_key_values.17.value[1,32,511,128],past_key_values.17.key[1,32,511,128],past_key_values.18.value[1,32,511,128],past_key_values.18.key[1,32,511,128],past_key_values.19.value[1,32,511,128],past_key_values.19.key[1,32,511,128],past_key_values.20.value[1,32,511,128],past_key_values.20.key[1,32,511,128],past_key_values.21.value[1,32,511,128],past_key_values.21.key[1,32,511,128],past_key_values.22.value[1,32,511,128],past_key_values.22.key[1,32,511,128],past_key_values.23.value[1,32,511,128],past_key_values.23.key[1,32,511,128],past_key_values.24.value[1,32,511,128],past_key_values.24.key[1,32,511,128],past_key_values.25.value[1,32,511,128],past_key_values.25.key[1,32,511,128],past_key_values.26.value[1,32,511,128],past_key_values.26.key[1,32,511,128],past_key_values.27.value[1,32,511,128],past_key_values.27.key[1,32,511,128],past_key_values.28.value[1,32,511,128],past_key_values.28.key[1,32,511,128],past_key_values.29.value[1,32,511,128],past_key_values.29.key[1,32,511,128],past_key_values.30.value[1,32,511,128],past_key_values.30.key[1,32,511,128],past_key_values.31.value[1,32,511,128],past_key_values.31.key[1,32,511,128] -t 120 -hint latency -infer_precision f32 -load_config ./tld0.6.json
```

- LLaMA-7B + Shears (INT8_ASYM)
```bash
benchmark_app -m llama7b-sparsity40-shears-ov-compressed/openvino_model.xml -data_shape input_ids[1,1],attention_mask[1,512],position_ids[1,512],past_key_values.0.value[1,32,511,128],past_key_values.0.key[1,32,511,128],past_key_values.1.value[1,32,511,128],past_key_values.1.key[1,32,511,128],past_key_values.2.value[1,32,511,128],past_key_values.2.key[1,32,511,128],past_key_values.3.value[1,32,511,128],past_key_values.3.key[1,32,511,128],past_key_values.4.value[1,32,511,128],past_key_values.4.key[1,32,511,128],past_key_values.5.value[1,32,511,128],past_key_values.5.key[1,32,511,128],past_key_values.6.value[1,32,511,128],past_key_values.6.key[1,32,511,128],past_key_values.7.value[1,32,511,128],past_key_values.7.key[1,32,511,128],past_key_values.8.value[1,32,511,128],past_key_values.8.key[1,32,511,128],past_key_values.9.value[1,32,511,128],past_key_values.9.key[1,32,511,128],past_key_values.10.value[1,32,511,128],past_key_values.10.key[1,32,511,128],past_key_values.11.value[1,32,511,128],past_key_values.11.key[1,32,511,128],past_key_values.12.value[1,32,511,128],past_key_values.12.key[1,32,511,128],past_key_values.13.value[1,32,511,128],past_key_values.13.key[1,32,511,128],past_key_values.14.value[1,32,511,128],past_key_values.14.key[1,32,511,128],past_key_values.15.value[1,32,511,128],past_key_values.15.key[1,32,511,128],past_key_values.16.value[1,32,511,128],past_key_values.16.key[1,32,511,128],past_key_values.17.value[1,32,511,128],past_key_values.17.key[1,32,511,128],past_key_values.18.value[1,32,511,128],past_key_values.18.key[1,32,511,128],past_key_values.19.value[1,32,511,128],past_key_values.19.key[1,32,511,128],past_key_values.20.value[1,32,511,128],past_key_values.20.key[1,32,511,128],past_key_values.21.value[1,32,511,128],past_key_values.21.key[1,32,511,128],past_key_values.22.value[1,32,511,128],past_key_values.22.key[1,32,511,128],past_key_values.23.value[1,32,511,128],past_key_values.23.key[1,32,511,128],past_key_values.24.value[1,32,511,128],past_key_values.24.key[1,32,511,128],past_key_values.25.value[1,32,511,128],past_key_values.25.key[1,32,511,128],past_key_values.26.value[1,32,511,128],past_key_values.26.key[1,32,511,128],past_key_values.27.value[1,32,511,128],past_key_values.27.key[1,32,511,128],past_key_values.28.value[1,32,511,128],past_key_values.28.key[1,32,511,128],past_key_values.29.value[1,32,511,128],past_key_values.29.key[1,32,511,128],past_key_values.30.value[1,32,511,128],past_key_values.30.key[1,32,511,128],past_key_values.31.value[1,32,511,128],past_key_values.31.key[1,32,511,128] -t 120 -hint latency -infer_precision f32 -load_config ./tld0.6.json
```

Results tested on `DGX`:

```bash
# LLaMA-7B (INT8_ASYM)
[ INFO ] Execution Devices:['CPU']
[ INFO ] Count:            851 iterations
[ INFO ] Duration:         120293.70 ms
[ INFO ] Latency:
[ INFO ]    Median:        292.72 ms
[ INFO ]    Average:       281.18 ms
[ INFO ]    Min:           163.17 ms
[ INFO ]    Max:           1343.84 ms
[ INFO ] Throughput:   7.07 FPS

# Unstructured sparsified LLaMA-7B (INT8_ASYM)
[ INFO ] Execution Devices:['CPU']
[ INFO ] Count:            884 iterations
[ INFO ] Duration:         120413.71 ms
[ INFO ] Latency:
[ INFO ]    Median:        267.13 ms
[ INFO ]    Average:       270.57 ms
[ INFO ]    Min:           155.35 ms
[ INFO ]    Max:           1203.65 ms
[ INFO ] Throughput:   7.34 FPS

# LLaMA-7B + Shears (INT8_ASYM)
[ INFO ] Execution Devices:['CPU']
[ INFO ] Count:            817 iterations
[ INFO ] Duration:         120251.26 ms
[ INFO ] Latency:
[ INFO ]    Median:        275.53 ms
[ INFO ]    Average:       292.85 ms
[ INFO ]    Min:           186.68 ms
[ INFO ]    Max:           1119.82 ms
[ INFO ] Throughput:   6.79 FPS
```
