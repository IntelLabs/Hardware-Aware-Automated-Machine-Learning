DEVICES=$1
TRANSFORMER_PATH=$2
CONFIG=$3
OUT_DIR=$4

CUDA_VISIBLE_DEVICES=${DEVICES}  python $(realpath ${TRANSFORMER_PATH})/examples/pytorch/text-classification/run_glue.py \
    --model_name_or_path bert-base-uncased \
    --task_name cola \
    --nncf_config ${CONFIG} \
    --output_dir ${OUT_DIR} \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 128 \
    --learning_rate 2e-5 \
    --num_train_epochs 5 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 1 \
    --seed 42 \
    --fp16 \
    --only_generate_importance_weight True