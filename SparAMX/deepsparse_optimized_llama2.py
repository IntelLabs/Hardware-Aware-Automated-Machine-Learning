# Example Run: numactl --cpunodebind 0 --membind 0 --physcpubind=0-31 python deepsparse_optimized_llama2.py --batch_size 32 --num_generation_tokens 128 --warmup_iterations 5 --num_iterations 5
import time
import numpy as np
import argparse
from deepsparse import TextGeneration

# def benchmark(batch_size, num_generation_tokens, input_size, num_iterations, warmup_iterations):
def benchmark(batch_size, num_generation_tokens, num_iterations, warmup_iterations):
    # Initialize the pipeline outside of timing
    pipeline = TextGeneration(model="hf:neuralmagic/Llama2-7b-chat-pruned50-quant-ds")

    # Prepare batch inputs (use realistic prompts of specified input size)
    batch_inputs = ["" for _ in range(batch_size)]

    # Warm-up phase
    for _ in range(warmup_iterations):
        _ = pipeline(batch_inputs, max_new_tokens=num_generation_tokens)

    # Benchmarking
    timings = []
    import pdb; pdb.set_trace()
    for _ in range(num_iterations):
        start_time = time.time()
        out = pipeline(batch_inputs, max_new_tokens=num_generation_tokens)
        end_time = time.time()
        timings.append(end_time - start_time)

    # Compute statistics
    mean_time = np.mean(timings)
    std_time = np.std(timings)
    median_time = np.median(timings)
    throughput = (batch_size * num_generation_tokens) / mean_time  # Tokens per second
    # Output results
    print(f"Mean inference time per batch: {mean_time:.4f} seconds")
    print(f"Standard deviation: {std_time:.4f} seconds")
    print(f"Median inference time per batch: {median_time:.4f} seconds")
    print(f"Throughput: {throughput:.2f} tokens per second")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark DeepSparse Text Generation Pipeline")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for the benchmark")
    parser.add_argument("--num_generation_tokens", type=int, default=1, help="Number of tokens to generate")
    # parser.add_argument("--input_size", type=int, default=1, help="Size of the input prompt (number of characters)")
    parser.add_argument("--num_iterations", type=int, default=100, help="Number of iterations for benchmarking")
    parser.add_argument("--warmup_iterations", type=int, default=10, help="Number of warm-up iterations")

    args = parser.parse_args()
    # Run the benchmark
    benchmark(
        batch_size=args.batch_size,
        num_generation_tokens=args.num_generation_tokens,
        # input_size=args.input_size,
        num_iterations=args.num_iterations,
        warmup_iterations=args.warmup_iterations
    )
