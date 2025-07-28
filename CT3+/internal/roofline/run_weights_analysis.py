import torch
from transformers import (
    AutoModel,
    Qwen2_5_VLForConditionalGeneration,
    Gemma3ForConditionalGeneration,
)
import matplotlib.pyplot as plt
import numpy as np


def to_billion(n):
    return n / 1e9


def analyze_model_params(model, mapping, model_name):
    params_count = {k: 0 for k in mapping}
    total_count = 0
    for name, param in model.named_parameters():
        matched = False
        for part, prefix in mapping.items():
            if name.startswith(prefix):
                params_count[part] += param.numel()
                matched = True
                break
        if not matched:
            raise AssertionError(f"Unrecognized parameter name: {name}")
        total_count += param.numel()
    print(f"\n[{model_name}]")
    for part, count in params_count.items():
        print(f"Total {part} params: {count} ({to_billion(count):.3f}B)")
    print(f"Total params: {total_count} ({to_billion(total_count):.3f}B)")
    # Return parameter counts in the order of mapping keys, plus total
    return [to_billion(params_count[k]) for k in mapping] + [to_billion(total_count)]


def plot_weights_histogram(model_labels, components, weights, filename="model_weights_histogram.pdf"):
    # Define custom RGB colors for bars
    rgb_colors = [
        (142/255, 207/255, 201/255),   # Projector
        (255/255, 190/255, 122/255),   # ViT
        (250/255, 127/255, 111/255),   # LLM
        (0.6, 0.6, 0.6),             # Total (gray)
    ]
    bar_width = 0.18
    x = np.arange(len(model_labels))

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Draw bars for each component and total
    for i, component in enumerate(components + ["Total"]):
        bars = plt.bar(
            x + i * bar_width - 1.5 * bar_width,
            weights[:, i],
            width=bar_width,
            color=rgb_colors[i],
            label=component
        )
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.3f}",
                ha='center',
                va='bottom',
                fontsize=10
            )

    # Remove all spines (borders)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Add horizontal grid lines for y-axis
    ax.yaxis.grid(True, color='#cccccc', linestyle='-', linewidth=1, alpha=0.7)
    ax.set_axisbelow(True)

    plt.ylabel("Billions of Weights", fontsize=12)
    plt.title("Parameter Distribution of Model Components", fontsize=18, loc='left')
    plt.xticks(x, model_labels, fontsize=11)
    plt.xlabel("")  # Remove x-axis label

    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.00),
        ncol=4,
        frameon=False,
        fontsize=14
    )

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"\nHistogram saved as {filename}")

if __name__ == "__main__":
    qwen_mapping = {
        "Projector": "model.visual.merger",
        "ViT": "model.visual",
        "LLM": "model.language_model",
    }
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    qwen_weights = analyze_model_params(qwen_model, qwen_mapping, "Qwen2.5-VL-3B")

    internvl_mapping = {
        "Projector": "mlp",
        "ViT": "vision_model",
        "LLM": "language_model",
    }
    internvl_model = AutoModel.from_pretrained(
        "OpenGVLab/InternVL3-2B",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
    ).eval().cuda()
    internvl_weights = analyze_model_params(internvl_model, internvl_mapping, "InternVL3-2B")

    gemma_mapping = {
        "Projector": "model.multi_modal_projector",
        "ViT": "model.vision_tower",
        "LLM": "model.language_model",
    }
    gemma_model = Gemma3ForConditionalGeneration.from_pretrained(
        "google/gemma-3-4b-it", 
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
    ).eval()
    gemma_weights = analyze_model_params(gemma_model, gemma_mapping, "Gemma3-4B")

    # Prepare labels with model size
    model_labels = ["Qwen2.5-VL-3B", "InternVL3-2B", "Gemma3-4B"]
    components = list(qwen_mapping.keys())
    weights = np.array([qwen_weights, internvl_weights, gemma_weights])
    plot_weights_histogram(model_labels, components, weights)
