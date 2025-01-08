import csv

# Example arrays (you can replace these with your actual data)
num_threads = [32]
cores = [32]
# modes = ['avx_sparse', 'sparse', 'stock']
modes = ['stock']
context_lengths = [512, 1024, 2048, 4096, 8192, 16384]
generation_lengths = [2]
batch_sizes = [1]
model_ids = ['meta-llama/Meta-Llama-3-8B']

num_groups = [32]
use_custom_attention = True
use_custom_k = [True, False]
use_custom_v = [True, False]
k_pruning_percentages = [50, 60, 70, 80]
v_pruning_percentages = [50, 60, 70, 80]

# Define the CSV file path
csv_file = "experiments.csv"

# Create a CSV file and write the header
with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Model ID", "Saved Model Path", "Context Length", "Number of generated Tokens", "Mode", "Number of Column Groups", "Number of Threads", "Number of Cores", "Batch Size", "Use Custom Attention", "Use Custom K", "Use Custom V", "K Pruning Percentage", "V Pruning Percentage", "Prefill Latency", "Decode Latency", "Total Latency", "Layer Times"])

    # Loop through the arrays and write the data
    for model_id in model_ids:
        for mode in modes:
            placeholder_saved_model_path = f"processed_models/{mode}/{model_id.split('/')[-1]}_num_threads_threads"
            for generation_length in generation_lengths:
                for context_length in context_lengths:
                    for core in cores:
                        for batch_size in batch_sizes:
                            for use_custom_k_val in use_custom_k:
                                for use_custom_v_val in use_custom_v:
                                    for num_thread in num_threads:
                                        saved_model_path = placeholder_saved_model_path.replace('_num_threads', f'_{num_thread}')
                                        for num_group in num_groups:
                                            original_model_path = f'{saved_model_path}'
                                            if mode == 'avx_sparse':
                                                saved_model_path = f'{saved_model_path}_{num_group}_groups'
                                            for k_pruning in k_pruning_percentages:
                                                for v_pruning in v_pruning_percentages:
                                                    writer.writerow([model_id, saved_model_path, context_length, generation_length, mode, num_group, num_thread, core, batch_size, use_custom_attention, use_custom_k_val, use_custom_v_val, k_pruning, v_pruning])
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
