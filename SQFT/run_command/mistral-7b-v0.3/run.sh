#!/bin/bash
set -x
SPARSITY=$1
TASK=$2

# Shears (https://arxiv.org/abs/2404.10934)
# -------------------------------------------------------------------------------
BASE_MODEL_PATH=IntelLabs/sqft-mistral-7b-v0.3-${SPARSITY}-base
ADAPTER_MODEL_PATH=trained_adapters/shears-mistral-7b-v0.3-${SPARSITY}-${TASK}-adapter
NNCF_CONFIG=nncf_config/sqft_sparsepeft/nncf_mistralv0.3.json
SEARCH_SPACE=32,28,24,20,16

if [ "$TASK" = "gsm8k" ]; then
  # train
  python run_standard_tuning.py --dataset_name gsm8k --model_name_or_path ${BASE_MODEL_PATH} --do_train --per_device_train_batch_size 16 --gradient_accumulation_steps 1 --num_train_epochs 3 --learning_rate 3e-4 --warmup_steps 100 --optim adamw_torch --fp16 --output_dir ${ADAPTER_MODEL_PATH} --logging_steps 20 --save_strategy epoch --save_total_limit 1 --lora --lora_r 32 --lora_alpha 64 --lora_dropout 0.1 --target_modules q_proj,k_proj,v_proj,up_proj,down_proj --nncf_config ${NNCF_CONFIG} --search_space ${SEARCH_SPACE}

  # extract heuristic adapter
  HEU_ADAPTER_MODEL_PATH=${ADAPTER_MODEL_PATH}/heuristic
  python utils/extract_sub_adapter.py --base_model ${BASE_MODEL_PATH} --adapter_model ${ADAPTER_MODEL_PATH} --nncf_config ${NNCF_CONFIG} --search_space ${SEARCH_SPACE} --subnet_version heuristic --output_dir ${HEU_ADAPTER_MODEL_PATH}

  # evaluation
  OUTPUT_DIR=${HEU_ADAPTER_MODEL_PATH}/result.json
  rm ${OUTPUT_DIR}
  lm_eval --model hf --model_args pretrained=${BASE_MODEL_PATH},peft=${HEU_ADAPTER_MODEL_PATH},add_bos_token=True,trust_remote_code=True --tasks gsm8k --batch_size auto:4 --output_path ${OUTPUT_DIR}

elif [ "$TASK" = "math" ]; then
  # train
  python run_instruction_tuning.py --dataset_path datasets/math_10k.json --model_name_or_path ${BASE_MODEL_PATH} --do_train --per_device_train_batch_size 16 --gradient_accumulation_steps 1 --num_train_epochs 3 --learning_rate 3e-4 --warmup_steps 100 --optim adamw_torch --fp16 --output_dir ${ADAPTER_MODEL_PATH} --logging_steps 20 --save_strategy epoch --save_total_limit 1 --lora --lora_r 32 --lora_alpha 64 --lora_dropout 0.1 --target_modules q_proj,k_proj,v_proj,up_proj,down_proj --nncf_config ${NNCF_CONFIG} --search_space ${SEARCH_SPACE}

  # extract heuristic adapter
  HEU_ADAPTER_MODEL_PATH=${ADAPTER_MODEL_PATH}/heuristic
  python utils/extract_sub_adapter.py --base_model ${BASE_MODEL_PATH} --adapter_model ${ADAPTER_MODEL_PATH} --nncf_config ${NNCF_CONFIG} --search_space ${SEARCH_SPACE} --subnet_version heuristic --output_dir ${HEU_ADAPTER_MODEL_PATH}

  # evaluation
  OUTPUT_DIR=${HEU_ADAPTER_MODEL_PATH}/eval_results
  rm -rf ${OUTPUT_DIR}
  python eval/evaluate_math.py --base_model_path ${BASE_MODEL_PATH} --adapter_model_path ${HEU_ADAPTER_MODEL_PATH} --output_dir ${OUTPUT_DIR}

# -------------------------------------------------------------------------------


# SQFT + SparsePEFT
# -------------------------------------------------------------------------------
BASE_MODEL_PATH=IntelLabs/sqft-mistral-7b-v0.3-${SPARSITY}-base
ADAPTER_MODEL_PATH=trained_adapters/sqft-sparsepeft-mistral-7b-v0.3-${SPARSITY}-${TASK}-adapter
NNCF_CONFIG=nncf_config/sqft_sparsepeft/nncf_mistralv0.3.json
SEARCH_SPACE=32,28,24,20,16

if [ "$TASK" = "gsm8k" ]; then
  # train
  python run_standard_tuning.py --dataset_name gsm8k --model_name_or_path ${BASE_MODEL_PATH} --sparse_adapter --do_train --per_device_train_batch_size 16 --gradient_accumulation_steps 1 --num_train_epochs 3 --learning_rate 3e-4 --warmup_steps 100 --optim adamw_torch --fp16 --output_dir ${ADAPTER_MODEL_PATH} --logging_steps 20 --save_strategy epoch --save_total_limit 1 --lora --lora_r 32 --lora_alpha 64 --lora_dropout 0.1 --target_modules q_proj,k_proj,v_proj,up_proj,down_proj --nncf_config ${NNCF_CONFIG} --search_space ${SEARCH_SPACE}

  # extract heuristic adapter
  HEU_ADAPTER_MODEL_PATH=${ADAPTER_MODEL_PATH}/heuristic
  python utils/extract_sub_adapter.py --base_model ${BASE_MODEL_PATH} --adapter_model ${ADAPTER_MODEL_PATH} --nncf_config ${NNCF_CONFIG} --search_space ${SEARCH_SPACE} --subnet_version heuristic --output_dir ${HEU_ADAPTER_MODEL_PATH}

  # merge
  MERGED_MODEL_PATH=${HEU_ADAPTER_MODEL_PATH}/merged
  python utils/merge.py --base_model_path ${BASE_MODEL_PATH} --adapter_model_path ${HEU_ADAPTER_MODEL_PATH} --output_path ${MERGED_MODEL_PATH}

  # check the sparsity of merged model
  python utils/check_sparsity.py --model_path ${MERGED_MODEL_PATH}

  # evaluation
  OUTPUT_DIR=${MERGED_MODEL_PATH}/result.json
  rm ${OUTPUT_DIR}
  lm_eval --model hf --model_args pretrained=${MERGED_MODEL_PATH},add_bos_token=True,trust_remote_code=True --tasks gsm8k --batch_size auto:4 --output_path ${OUTPUT_DIR}

elif [ "$TASK" = "math" ]; then
  # train
  python run_instruction_tuning.py --dataset_path datasets/math_10k.json --model_name_or_path ${BASE_MODEL_PATH} --sparse_adapter --do_train --per_device_train_batch_size 16 --gradient_accumulation_steps 1 --num_train_epochs 3 --learning_rate 3e-4 --warmup_steps 100 --optim adamw_torch --fp16 --output_dir ${ADAPTER_MODEL_PATH} --logging_steps 20 --save_strategy epoch --save_total_limit 1 --lora --lora_r 32 --lora_alpha 64 --lora_dropout 0.1 --target_modules q_proj,k_proj,v_proj,up_proj,down_proj --nncf_config ${NNCF_CONFIG} --search_space ${SEARCH_SPACE}

  # extract heuristic adapter
  HEU_ADAPTER_MODEL_PATH=${ADAPTER_MODEL_PATH}/heuristic
  python utils/extract_sub_adapter.py --base_model ${BASE_MODEL_PATH} --adapter_model ${ADAPTER_MODEL_PATH} --nncf_config ${NNCF_CONFIG} --search_space ${SEARCH_SPACE} --subnet_version heuristic --output_dir ${HEU_ADAPTER_MODEL_PATH}

  # merge
  MERGED_MODEL_PATH=${HEU_ADAPTER_MODEL_PATH}/merged
  python utils/merge.py --base_model_path ${BASE_MODEL_PATH} --adapter_model_path ${HEU_ADAPTER_MODEL_PATH} --output_path ${MERGED_MODEL_PATH}

  # check the sparsity of merged model
  python utils/check_sparsity.py --model_path ${MERGED_MODEL_PATH}

  # evaluation
  OUTPUT_DIR=${MERGED_MODEL_PATH}/eval_results
  rm -rf ${OUTPUT_DIR}
  python eval/evaluate_math.py --base_model_path ${MERGED_MODEL_PATH} --output_dir ${OUTPUT_DIR}

# -------------------------------------------------------------------------------


# SQFT
# -------------------------------------------------------------------------------
BASE_MODEL_PATH=IntelLabs/sqft-mistral-7b-v0.3-${SPARSITY}-base-gptq
ADAPTER_MODEL_PATH=trained_adapters/sqft-mistral-7b-v0.3-${SPARSITY}-gptq-${TASK}-adapter
NNCF_CONFIG=nncf_config/sqft/nncf_mistralv0.3.json
SEARCH_SPACE=32,28,24,20,16

if [ "$TASK" = "gsm8k" ]; then
  # train
  python run_standard_tuning.py --dataset_name gsm8k --model_name_or_path ${BASE_MODEL_PATH} --do_train --per_device_train_batch_size 16 --gradient_accumulation_steps 1 --num_train_epochs 3 --learning_rate 3e-4 --warmup_steps 100 --optim adamw_torch --fp16 --output_dir ${ADAPTER_MODEL_PATH} --logging_steps 20 --save_strategy epoch --save_total_limit 1 --lora --lora_r 32 --lora_alpha 64 --lora_dropout 0.1 --target_modules q_proj,k_proj,v_proj,up_proj,down_proj --nncf_config ${NNCF_CONFIG} --search_space ${SEARCH_SPACE}

  # extract heuristic adapter
  HEU_ADAPTER_MODEL_PATH=${ADAPTER_MODEL_PATH}/heuristic
  python utils/extract_sub_adapter.py --base_model ${BASE_MODEL_PATH} --adapter_model ${ADAPTER_MODEL_PATH} --nncf_config ${NNCF_CONFIG} --search_space ${SEARCH_SPACE} --subnet_version heuristic --output_dir ${HEU_ADAPTER_MODEL_PATH}

  # evaluation
  OUTPUT_DIR=${HEU_ADAPTER_MODEL_PATH}/result.json
  rm ${OUTPUT_DIR}
  lm_eval --model hf --model_args pretrained=${BASE_MODEL_PATH},peft=${HEU_ADAPTER_MODEL_PATH},add_bos_token=True,trust_remote_code=True --tasks gsm8k --batch_size auto:4 --output_path ${OUTPUT_DIR}

elif [ "$TASK" = "math" ]; then
  # train
  python run_instruction_tuning.py --dataset_path datasets/math_10k.json --model_name_or_path ${BASE_MODEL_PATH} --do_train --per_device_train_batch_size 16 --gradient_accumulation_steps 1 --num_train_epochs 3 --learning_rate 3e-4 --warmup_steps 100 --optim adamw_torch --fp16 --output_dir ${ADAPTER_MODEL_PATH} --logging_steps 20 --save_strategy epoch --save_total_limit 1 --lora --lora_r 32 --lora_alpha 64 --lora_dropout 0.1 --target_modules q_proj,k_proj,v_proj,up_proj,down_proj --nncf_config ${NNCF_CONFIG} --search_space ${SEARCH_SPACE}

  # extract heuristic adapter
  HEU_ADAPTER_MODEL_PATH=${ADAPTER_MODEL_PATH}/heuristic
  python utils/extract_sub_adapter.py --base_model ${BASE_MODEL_PATH} --adapter_model ${ADAPTER_MODEL_PATH} --nncf_config ${NNCF_CONFIG} --search_space ${SEARCH_SPACE} --subnet_version heuristic --output_dir ${HEU_ADAPTER_MODEL_PATH}

  # evaluation
  OUTPUT_DIR=${HEU_ADAPTER_MODEL_PATH}/eval_results
  rm -rf ${OUTPUT_DIR}
  python eval/evaluate_math.py --base_model_path ${BASE_MODEL_PATH} --adapter_model_path ${HEU_ADAPTER_MODEL_PATH} --output_dir ${OUTPUT_DIR}

# -------------------------------------------------------------------------------


# SQFT + QA-SparsePEFT
# -------------------------------------------------------------------------------
BASE_MODEL_PATH=IntelLabs/sqft-mistral-7b-v0.3-${SPARSITY}-base-gptq
NON_QUANT_BASE_MODEL_PATH=IntelLabs/sqft-mistral-7b-v0.3-${SPARSITY}-base
ADAPTER_MODEL_PATH=trained_adapters/sqft-qa-sparsepeft-mistral-7b-v0.3-${SPARSITY}-gptq-${TASK}-adapter
NNCF_CONFIG=nncf_config/sqft_qa_sparsepeft/nncf_mistralv0.3.json
SEARCH_SPACE=32,28,24,20,16

if [ "$TASK" = "gsm8k" ]; then
  # train
  python run_standard_tuning.py --dataset_name gsm8k --model_name_or_path ${BASE_MODEL_PATH} --non_quant_model_name_or_path ${NON_QUANT_BASE_MODEL_PATH} --quantization_aware --sparse_adapter --do_train --per_device_train_batch_size 16 --gradient_accumulation_steps 1 --num_train_epochs 4 --learning_rate 3e-4 --warmup_steps 100 --optim adamw_torch --fp16 --output_dir ${ADAPTER_MODEL_PATH} --logging_steps 20 --save_strategy epoch --save_total_limit 1 --lora --lora_r 32 --lora_alpha 64 --lora_dropout 0.1 --target_modules q_proj,k_proj,v_proj,up_proj,down_proj --nncf_config ${NNCF_CONFIG} --search_space ${SEARCH_SPACE}

  # extract heuristic adapter
  HEU_ADAPTER_MODEL_PATH=${ADAPTER_MODEL_PATH}/heuristic
  python utils/extract_sub_adapter.py --base_model ${BASE_MODEL_PATH} --adapter_model ${ADAPTER_MODEL_PATH} --nncf_config ${NNCF_CONFIG} --search_space ${SEARCH_SPACE} --subnet_version heuristic --output_dir ${HEU_ADAPTER_MODEL_PATH}

  # merge
  MERGED_MODEL_PATH=${HEU_ADAPTER_MODEL_PATH}/merged
  python utils/merge.py --base_model_path ${BASE_MODEL_PATH} --non_quant_base_model_path ${NON_QUANT_BASE_MODEL_PATH} --adapter_model_path ${HEU_ADAPTER_MODEL_PATH} --output_path ${MERGED_MODEL_PATH}

  # check the sparsity of merged model
  python utils/check_sparsity.py --model_path ${MERGED_MODEL_PATH}

  # evaluation
  OUTPUT_DIR=${MERGED_MODEL_PATH}/result.json
  rm ${OUTPUT_DIR}
  lm_eval --model hf --model_args pretrained=${MERGED_MODEL_PATH},add_bos_token=True,trust_remote_code=True --tasks gsm8k --batch_size auto:4 --output_path ${OUTPUT_DIR}

elif [ "$TASK" = "math" ]; then
  # train
  python run_instruction_tuning.py --dataset_path datasets/math_10k.json --model_name_or_path ${BASE_MODEL_PATH} --non_quant_model_name_or_path ${NON_QUANT_BASE_MODEL_PATH} --quantization_aware --sparse_adapter --do_train --per_device_train_batch_size 16 --gradient_accumulation_steps 1 --num_train_epochs 4 --learning_rate 3e-4 --warmup_steps 100 --optim adamw_torch --fp16 --output_dir ${ADAPTER_MODEL_PATH} --logging_steps 20 --save_strategy epoch --save_total_limit 1 --lora --lora_r 32 --lora_alpha 64 --lora_dropout 0.1 --target_modules q_proj,k_proj,v_proj,up_proj,down_proj --nncf_config ${NNCF_CONFIG} --search_space ${SEARCH_SPACE}

  # extract heuristic adapter
  HEU_ADAPTER_MODEL_PATH=${ADAPTER_MODEL_PATH}/heuristic
  python utils/extract_sub_adapter.py --base_model ${BASE_MODEL_PATH} --adapter_model ${ADAPTER_MODEL_PATH} --nncf_config ${NNCF_CONFIG} --search_space ${SEARCH_SPACE} --subnet_version heuristic --output_dir ${HEU_ADAPTER_MODEL_PATH}

  # merge
  MERGED_MODEL_PATH=${HEU_ADAPTER_MODEL_PATH}/merged
  python utils/merge.py --base_model_path ${BASE_MODEL_PATH} --non_quant_base_model_path ${NON_QUANT_BASE_MODEL_PATH} --adapter_model_path ${HEU_ADAPTER_MODEL_PATH} --output_path ${MERGED_MODEL_PATH}

  # check the sparsity of merged model
  python utils/check_sparsity.py --model_path ${MERGED_MODEL_PATH}

  # evaluation
  OUTPUT_DIR=${MERGED_MODEL_PATH}/eval_results
  rm -rf ${OUTPUT_DIR}
  python eval/evaluate_math.py --base_model_path ${MERGED_MODEL_PATH} --output_dir ${OUTPUT_DIR}
# -------------------------------------------------------------------------------