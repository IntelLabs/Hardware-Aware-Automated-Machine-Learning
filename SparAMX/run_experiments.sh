#!/bin/bash

# Input and output CSV files
input_csv="experiments.csv"
time=$(date +"%Y-%m-%d_%H-%M-%S")
dir="experiment_results/$time"
mkdir -p $dir
output_csv="$dir/results.csv"
prefill_time_file="prefill_time.txt"
decode_time_file="decode_time.txt"
total_time_file="total_time.txt"
layer_times_file="layer_times.txt"
num_repeats=1
# Read the header and write it to the output CSV
header=$(head -n 1 "$input_csv")
echo "$header" > "$output_csv"

# Process the CSV file line by line, skipping the header
tail -n +2 "$input_csv" | tr -d '\r' | while IFS=, read -r model_id saved_model_path context_length num_generated_tokens mode num_groups num_threads core batch_size enable_custom_attention use_custom_k use_custom_v k_pruning_percentage v_pruning_percentage; do

  for ((i=1; i<=$num_repeats; i++)); do
    rm -f $prefill_time_file
    rm -f $decode_time_file

    if [ "$mode" = "avx_sparse" ]; then
        sed -i '/# CHANGE BELOW FOR GROUP SIZE/!b;n;cNUM_GROUPS = '"$num_groups" layer/avx_sparse_linear.py
        sed -i '/\/\/ CHANGE BELOW FOR GROUP SIZE/!b;n;c#define NUM_COL_GROUPS          '"$num_groups" csrc/avx_sparse_linear.cpp
        python setup.py install
    fi

    # construct new variable core_vals = "0-($core-1)"
    core_vals="0-$(($core-1))"

    enable_custom_attention_str=""
    if [ "$enable_custom_attention" = "True" ]; then
        enable_custom_attention_str="--enable_custom_attention"
    fi

    use_custom_k_str=""
    if [ "$use_custom_k" = "True" ]; then
        use_custom_k_str="--use_custom_k"
    fi

    use_custom_v_str=""
    if [ "$use_custom_v" = "True" ]; then
        use_custom_v_str="--use_custom_v"
    fi

    # Run the experiment
    numactl --cpunodebind 0 --membind 0 --physcpubind=$core_vals python llm_pipeline.py --model_id $model_id --saved_model_path $saved_model_path --context_length $context_length --num_generated_tokens $num_generated_tokens --mode $mode --num_threads $num_threads --batch_size $batch_size $enable_custom_attention_str $use_custom_k_str $use_custom_v_str --k_pruning $k_pruning_percentage --v_pruning $v_pruning_percentage

    # Read the result from the result file
    time_prefill=$(<$prefill_time_file)
    time_decode=$(<$decode_time_file)
    total_time=$(<$total_time_file)
    # layer_times=$(<$layer_times_file)

    # Construct the new CSV line with the result
    new_line="$model_id,$saved_model_path,$context_length,$num_generated_tokens,$mode,$num_groups,$num_threads,$core,$batch_size,$enable_custom_attention,$use_custom_k,$use_custom_v,$k_pruning_percentage,$v_pruning_percentage,$time_prefill,$time_decode,$total_time"
    # new_line="$model_id,$saved_model_path,$context_length,$num_generated_tokens,$mode,$num_groups,$num_threads,$core,$batch_size,$enable_custom_attention,$use_custom_k,$use_custom_v,$k_pruning_percentage,$v_pruning_percentage,$time_prefill,$time_decode,$total_time,$layer_times"
    # new_line="$model_id,$saved_model_path,$context_length,$num_generated_tokens,$mode,$num_groups,$num_threads,$core,$batch_size,$time_prefill,$time_decode"

    # Append the new line to the output CSV
    echo "$new_line" >> "$output_csv"
  done
done
