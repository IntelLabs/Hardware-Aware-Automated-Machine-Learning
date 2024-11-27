import csv

# Example arrays (you can replace these with your actual data)
num_threads = [32]
cores = [32]
# modes = ['denseq']
modes = ['stock']
# modes = ['sparse']
# attention_modes = ['sparse', 'dense']
attention_modes = ['sparse']
sparsity_percentages = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
# modes = ['sparseq']
context_lengths = [16384]
generation_lengths = [8]
# batch_sizes = [16, 32, 64, 128, 256, 512, 1024]
batch_sizes = [1]
# model_ids = ['meta-llama/Meta-Llama-3-8B', 'IntelLabs/sqft-sparsepeft-llama-3-8b-50-gsm8k']
# model_ids = ['mgoin/TinyStories-1M-ds']
# model_ids = ['jpablomch/sqft-sparsepeft-llama-3-8b-50-gsm8k-heu']
# model_ids = ['sparsified_models/llama_2_7b_wanda_50_unstructured']
# model_ids = ['sparsified_models/llama_2_7b_wanda_50_unstructured']
# model_ids = ['jpablomch/PA-sqft-sparsepeft-llama-3-8b-20-gsm8k-heu', 'jpablomch/PA-sqft-sparsepeft-llama-3-8b-30-gsm8k-heu']
model_ids = ['jpablomch/sqft-sparsepeft-llama-3-8b-50-gsm8k-heu']
# saved_model_paths = [f'processed_shears_llama-3-8b-0-sparsity_{num_threads}_threads',f'processed_shears_llama-3-8b-50-sparsity_{num_threads}_threads']
# saved_model_paths = [f'processed_shears_llama-3-8b-50-sparsity_num_threads_threads']
# saved_model_paths = [f'processed_TinyStories-1M-ds_num_threads_threads']
num_groups = [32]
use_custom_attention = True
use_custom_k = [True]
use_custom_v = [True]
v_pruning_percentages = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
k_pruning_percentages = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
# use_custom_attention = False
# use_custom_k = [False]
# use_custom_v = [False]
# k_pruning_percentages = [0]
# v_pruning_percentages = [0]
int8 = False

# Define the CSV file path
csv_file = "experiments_attention.csv"

# Create a CSV file and write the header
with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Model ID", "Saved Model Path", "Context Length", "Number of generated Tokens", "Mode", "Number of Column Groups", "Number of Threads", "Number of Cores", "Batch Size", "Use Custom Attention", "Use Custom K", "Use Custom V", "K Pruning Percentage", "V Pruning Percentage", "Int8?" ,"Attention Mode", "Prefill Latency", "Decode Latency", "Total Latency", "Layer Times"])

    # Loop through the arrays and write the data
    for model_id in model_ids:
        for mode in modes:
            placeholder_saved_model_path = f"processed_models/{mode}/{model_id.split('/')[-1]}_num_threads_threads"
            if int8:
                placeholder_saved_model_path = f"processed_models/{mode}/int8/{model_id.split('/')[-1]}_num_threads_threads"
            for generation_length in generation_lengths:
                for context_length in context_lengths:
                    for core in cores:
                        for batch_size in batch_sizes:
                            for use_custom_k_val in use_custom_k:
                                for use_custom_v_val in use_custom_v:
                                    for num_thread in num_threads:
                                        if num_thread != core:
                                            continue
                                        saved_model_path = placeholder_saved_model_path.replace('_num_threads', f'_{num_thread}')
                                        for num_group in num_groups:
                                            original_model_path = f'{saved_model_path}'
                                            if mode == 'avx_sparse':
                                                saved_model_path = f'{saved_model_path}_{num_group}_groups'
                                            for k_pruning in k_pruning_percentages:
                                                for v_pruning in v_pruning_percentages:
                                                    for attention_mode in attention_modes:
                                                        writer.writerow([model_id, saved_model_path, context_length, generation_length, mode, num_group, num_thread, core, batch_size, use_custom_attention, use_custom_k_val, use_custom_v_val, k_pruning, v_pruning, int8, attention_mode])
                                                    if not(use_custom_k_val):
                                                        break
                                                if not(use_custom_v_val):
                                                    break
                                            saved_model_path = original_model_path
                                            if mode != 'avx_sparse':
                                                break
                                        if mode == 'stock':
                                            break

print(f"CSV file '{csv_file}' has been generated with the data.")
