# EFTNAS

## Env Setup

### 1. NNCF
```
git clone https://github.com/openvinotoolkit/nncf.git nncf_eftnas
cd nncf_eftnas
git checkout 415c3c4d
# Apply Patch 
git apply path/to/nncf.patch
pip install -e .
```

### 2. Hugging Face - Transformers
```
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout v4.29.1
git apply /path/to/transformers.patch
pip install -e .
```

## EFTNAS Results
eftnas_results
- BERT: {Subnet version}_{task}
    - Subnetwork version: S1 for EFTNAS-S1, S2 for EFTNAS-S2
    - task: datasets in GLUE benchmark, SQuADv1, SQuADv2
- ViT: ViT

## EFTNAS Configs
Configs are in the folder `eftnas_configs`.
- BERT: {Subnet version}_{task}.json
    - Subnetwork version: S1 for EFTNAS-S1, S2 for EFTNAS-S2
    - task: datasets in GLUE benchmark, SQuADv1, SQuADv2
- ViT: ViT.json

### Config Explanation (Extra parameters in the nncf config)
- **movement_sparsity_total_epochs**: Used for movement sparsity. This overwrites `num_train_epochs` in the following scripts. It is similar to `epoch_lr` in BootstrapNAS. The real training epochs depends on `warmup_end_epoch` in movement_sparsity params.
- **model_name_or_path**: Overwrite `model_name_or_path` in the following scripts.
- **kd_teacher_model**: Teacher model.
- **reorg_cache_model**: Cache model of trained importance weight.

## Reproduce Results

** In the following scripts, nncf_config.json will overwrite some hyper-parameters.
For example, `model_name_or_path` and `num_train_epochs`.

### Train BERT - GLUE
```
cd /path/to/transformers
CUDA_VISIBLE_DEVICES=${DEVICES}  python examples/pytorch/text-classification/run_glue.py \
    --model_name_or_path bert-base-uncased \
    --task_name $TASK_NAME \
    --nncf_config ${CONFIG} \
    --output_dir ${OUT_DIR} \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --do_search \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 128 \
    --learning_rate 2e-5 \
    --num_train_epochs 5 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 1 \
    --seed 42 \
    --fp16
```
- TASK_NAME: mnli, qqp, qnli, cola, rte, sst2, mrpc
- CONFIG: Select one from `eftnas_configs`.
- OUT_DIR: Your output dir

### Train BERT - SQuADv1.1

```
cd /path/to/transformers
CUDA_VISIBLE_DEVICES=${DEVICES} python examples/pytorch/question-answering/run_qa.py \
    --model_name_or_path bert-base-uncased \
    --do_train \
    --do_eval \
    --do_search \
    --dataset_name squad \
    --learning_rate 3e-5 \
    --per_gpu_train_batch_size 16 \
    --per_gpu_eval_batch_size 128 \
    --output_dir ${OUT_DIR} \
    --max_seq_length 384 \
    --doc_stride 128 \
    --nncf_config ${CONFIG} \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --metric_for_best_model f1 \
    --overwrite_output_dir \
    --save_total_limit 1 \
    --num_train_epochs 8 \
    --fp16
```
- CONFIG: Select one from `eftnas_configs`.
- OUT_DIR: Your output dir

### Train BERT - SQuADv2.0

```
CUDA_VISIBLE_DEVICES=${DEVICES} python examples/pytorch/question-answering/run_qa.py \
    --model_name_or_path bert-base-uncased \
    --do_train \
    --do_eval \
    --do_search \
    --dataset_name squad_v2 \
    --learning_rate 3e-5 \
    --per_gpu_train_batch_size 16 \
    --per_gpu_eval_batch_size 128 \
    --version_2_with_negative \
    --output_dir ${OUT_DIR} \
    --max_seq_length 384 \
    --doc_stride 128 \
    --nncf_config ${CONFIG} \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --metric_for_best_model f1 \
    --kd_teacher_model deepset/bert-base-uncased-squad2 \
    --ddp_find_unused_parameters True \
    --overwrite_output_dir \
    --save_total_limit 1 \
    --num_train_epochs 8 \
    --fp16
```
- CONFIG: Select one from `eftnas_configs`.
- OUT_DIR: Your output dir

### Train ViT

#### Train ViT - importance mask

```
cd /path/to/nncf/examples/experimental/torch/classification/
CUDA_VISIBLE_DEVICES=${DEVICES} python get_importance_mask.py \
    --mode train \
    --config eftnas_configs/ViT.json \
    --data ${IMAGENET_DIR} \
    --log-dir ${OUT_DIR}
```

- IMAGENET_DIR: path to imagenet dataset. 
- OUT_DIR: Your output dir.

#### Train ViT - NAS part

```
cd /path/to/nncf/examples/experimental/torch/classification/
CUDA_VISIBLE_DEVICES=${DEVICES} torchrun --nproc_per_node=2 --master_port ${MASTER_PORT} bootstrap_nas.py \
    --mode train \
    --config aaai_configs/ViT.json \
    --data ${IMAGENET_DIR} \
    --log-dir $OUT_DIR \
    --importance-mask-weight ${IMPORTANCE_MASK_WEIGHT}
```

- IMAGENET_DIR: path to imagenet dataset. 
- MASTER_PORT: master port
- OUT_DIR: Your output dir.
- IMPORTANCE_MASK_WEIGHT: ../eftnas_vit/model_best.pth)