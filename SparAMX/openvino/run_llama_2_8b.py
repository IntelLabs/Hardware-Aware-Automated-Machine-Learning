import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from optimum.intel import OVModelForCausalLM
from optimum.intel.openvino import OVQuantizer
import openvino as ov
import logging
import time
import numpy as np


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/llama-2-7b-chat-hf")

# Load the model
model = AutoModelForCausalLM.from_pretrained("meta-llama/llama-2-7b-chat-hf")


# Convert to OpenVINO format
ov_model = OVModelForCausalLM.from_pretrained(
    model_id="meta-llama/llama-2-7b-chat-hf",
    export=True,
    load_in_8bit=False
)

# Save the quantized model
ov_model.save_pretrained("llama2-16bit-ov")

# Function to generate text
def generate_text(prompt, max_length=8):
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate
    outputs = ov_model.generate(
        inputs.input_ids,
        max_length=max_length,
        num_beams=1,
        temperature=0.7,
        # no_repeat_ngram_size=2
    )
    
    # Decode and return the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def measure_generation_speed(ov_model, tokenizer, prompt, num_runs=5, max_new_tokens=100):
    logger.info(f"Measuring generation speed over {num_runs} runs...")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids + torch.zeros((32,1), dtype = input_ids.dtype)
    # import pdb; pdb.set_trace()
    input_length = len(input_ids[0])
    
    generation_times = []
    tokens_generated = []
    token_times = []
    
    # Warm-up run
    logger.info("Performing warm-up run...")
    _ = ov_model.generate(input_ids, max_new_tokens=max_new_tokens)
    
    for run in range(num_runs):
        logger.info(f"Run {run + 1}/{num_runs}")
        
        # Time the generation
        start_time = time.time()
        outputs = ov_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            num_beams=1,  # Use greedy decoding for consistent timing
            temperature=0.7,
            return_dict_in_generate=True,
            output_scores=True
        )
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        output_length = len(outputs.sequences[0]) - input_length
        avg_time_per_token = total_time / output_length
        
        generation_times.append(total_time)
        tokens_generated.append(output_length)
        token_times.append(avg_time_per_token)
        
        generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        
        logger.info(f"Generated {output_length} new tokens in {total_time:.2f} seconds")
        logger.info(f"Average time per token: {avg_time_per_token:.4f} seconds")
    
    # Calculate aggregate statistics
    mean_generation_time = np.mean(generation_times)
    mean_tokens = np.mean(tokens_generated)
    mean_token_time = np.mean(token_times)
    std_token_time = np.std(token_times)
    
    stats = {
        'mean_generation_time': mean_generation_time,
        'mean_tokens_generated': mean_tokens,
        'mean_time_per_token': mean_token_time,
        'std_time_per_token': std_token_time,
        'individual_runs': list(zip(generation_times, tokens_generated, token_times))
    }
    
    return stats, generated_text

def print_stats(stats):
    logger.info("\nGeneration Statistics:")
    logger.info(f"Average generation time: {stats['mean_generation_time']:.2f} seconds")
    logger.info(f"Average tokens generated: {stats['mean_tokens_generated']:.1f}")
    logger.info(f"Average time per token: {stats['mean_time_per_token']*1000:.2f} ms")
    logger.info(f"Standard deviation of time per token: {stats['std_time_per_token']*1000:.2f} ms")
    
    logger.info("\nIndividual runs (total time, tokens, ms per token):")
    for i, (time, tokens, token_time) in enumerate(stats['individual_runs'], 1):
        logger.info(f"Run {i}: {time:.2f}s, {tokens} tokens, {token_time*1000:.2f} ms/token")


# Example usage
if __name__ == "__main__":
    prompt = ""
    stats, generated_text = measure_generation_speed(ov_model, tokenizer, prompt)
    print(f"Input: {prompt}")
    print("\nGenerated Text:")
    print(generated_text)
    print("\nTiming Statistics:")
    print_stats(stats)