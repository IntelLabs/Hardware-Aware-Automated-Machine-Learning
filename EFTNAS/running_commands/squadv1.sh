#!/bin/bash

EFTNAS_PATH=$PWD
DEVICES=0

# Model: BERT-base
# Dataset: squadv1

cd transformers_eftnas
CUDA_VISIBLE_DEVICES=${DEVICES} python examples/pytorch/question-answering/run_qa.py \
    --model_name_or_path bert-base-uncased \
    --do_train \
    --do_eval \
    --dataset_name squad \
    --learning_rate 2e-5 \
    --per_gpu_train_batch_size 16 \
    --per_gpu_eval_batch_size 128 \
    --output_dir ${EFTNAS_PATH}/results/trained_models/eftnas-bert-base-squadv1/movement_sparsity \
    --max_seq_length 384 \
    --doc_stride 128 \
    --nncf_config ${EFTNAS_PATH}/eftnas_configs/nncf_eftnas_s1_bert_base_squadv1.json \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --metric_for_best_model f1 \
    --overwrite_output_dir \
    --save_total_limit 1 \
    --num_train_epochs 8 \
    --fp16 \
    --only_generate_importance_weight True

CUDA_VISIBLE_DEVICES=${DEVICES} python eftnas_search_space/generate_eftnas_search_space.py \
  --source_config eftnas_configs/nncf_eftnas_s1_bert_base_squadv1.json \
  --model_name_or_path bert-base-uncased \
  --importance_weight_dir trained_models/eftnas-bert-base-squadv1/movement_sparsity \
  --target_config results/generated_configs/nncf_eftnas_s1_bert_base_squadv1.json

cd transformers_eftnas
CUDA_VISIBLE_DEVICES=${DEVICES} python examples/pytorch/question-answering/run_qa.py \
    --model_name_or_path bert-base-uncased \
    --kd_teacher_model csarron/bert-base-uncased-squad-v1 \
    --reorg_cache_model ${EFTNAS_PATH}/results/trained_models/eftnas-bert-base-squadv1/movement_sparsity/pytorch_model.bin \
    --do_train \
    --do_eval \
    --do_search \
    --dataset_name squad \
    --learning_rate 3e-5 \
    --per_gpu_train_batch_size 16 \
    --per_gpu_eval_batch_size 128 \
    --output_dir ${EFTNAS_PATH}/results/trained_models/eftnas-bert-base-squadv1 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --nncf_config ${EFTNAS_PATH}/results/generated_configs/nncf_eftnas_s1_bert_base_squadv1.json \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --metric_for_best_model f1 \
    --save_total_limit 1 \
    --num_train_epochs 8 \
    --fp16
