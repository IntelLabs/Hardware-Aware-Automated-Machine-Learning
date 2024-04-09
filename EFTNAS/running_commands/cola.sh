#!/bin/bash

EFTNAS_PATH=$PWD
DEVICES=0

# Model: BERT-base
# Dataset: cola

cd transformers_eftnas
CUDA_VISIBLE_DEVICES=${DEVICES} python examples/pytorch/text-classification/run_glue.py \
    --model_name_or_path bert-base-uncased \
    --task_name cola \
    --nncf_config ${EFTNAS_PATH}/eftnas_configs/nncf_eftnas_s1_bert_base_cola.json \
    --output_dir ${EFTNAS_PATH}/results/trained_models/eftnas-bert-base-cola/movement_sparsity \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 128 \
    --learning_rate 2e-5 \
    --num_train_epochs 30 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 1 \
    --seed 42 \
    --fp16 \
    --only_generate_importance_weight True
cd ..

CUDA_VISIBLE_DEVICES=${DEVICES} python eftnas_search_space/generate_eftnas_search_space.py \
  --source_config eftnas_configs/nncf_eftnas_s1_bert_base_cola.json \
  --model_name_or_path bert-base-uncased \
  --importance_weight_dir trained_models/eftnas-bert-base-cola/movement_sparsity \
  --target_config results/generated_configs/nncf_eftnas_s1_bert_base_cola.json

cd transformers_eftnas
CUDA_VISIBLE_DEVICES=${DEVICES} python examples/pytorch/text-classification/run_glue.py \
    --model_name_or_path bert-base-uncased \
    --kd_teacher_model ModelTC/bert-base-uncased-cola \
    --reorg_cache_model ${EFTNAS_PATH}/results/trained_models/eftnas-bert-base-cola/movement_sparsity/pytorch_model.bin \
    --task_name cola \
    --nncf_config ${EFTNAS_PATH}/results/generated_configs/nncf_eftnas_s1_bert_base_cola.json \
    --output_dir ${EFTNAS_PATH}/results/trained_models/eftnas-bert-base-cola \
    --do_train \
    --do_eval \
    --do_search \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 128 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 1 \
    --seed 42 \
    --fp16
