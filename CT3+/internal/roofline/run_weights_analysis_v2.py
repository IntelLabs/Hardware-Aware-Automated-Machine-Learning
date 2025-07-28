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
    extra_keys = ["ViT-MLP", "ViT-Attn", "LLM-MLP", "LLM-Attn"]
    extra_count = {k: 0 for k in extra_keys}
    total_count = 0
    for name, param in model.named_parameters():
        matched = False
        for part, rule in mapping.items():
            if isinstance(rule, tuple):
                prefix, keyword = rule
                if name.startswith(prefix) and keyword in name:
                    params_count[part] += param.numel()
                    matched = True
                    break
            else:
                if name.startswith(rule):
                    params_count[part] += param.numel()
                    matched = True
                    break
        for base, prefix in [("ViT", mapping.get("ViT")), ("LLM", mapping.get("LLM"))]:
            if prefix and name.startswith(prefix):
                if "mlp" in name and "visual.merger" not in name: # just for Qwen2.5-VL
                    extra_count[f"{base}-MLP"] += param.numel()
                if "attn" in name:
                    extra_count[f"{base}-Attn"] += param.numel()
        if not matched:
            raise AssertionError(f"Unrecognized parameter name: {name}")
        total_count += param.numel()
    print(f"\n[{model_name}]")
    for part, count in params_count.items():
        print(f"Total {part} params: {count} ({to_billion(count):.3f}B)")
    for part, count in extra_count.items():
        print(f"  {part} params: {count} ({to_billion(count):.3f}B)")
    print(f"Total params: {total_count} ({to_billion(total_count):.3f}B)")
    return (
        [to_billion(params_count[k]) for k in mapping] +
        [to_billion(extra_count[k]) for k in extra_keys] +
        [to_billion(total_count)]
    )


def plot_weights_histogram(model_labels, components, extra_keys, weights, filename="model_weights_histogram.pdf"):
    rgb_colors = [
        (142/255, 207/255, 201/255),   # Projector
        (255/255, 190/255, 122/255),   # ViT
        (250/255, 127/255, 111/255),   # LLM
        (0.6, 0.6, 0.6),               # Total (gray)
        (130/255, 176/255, 210/255),   # ViT-MLP
        (190/255, 184/255, 220/255),   # ViT-Attn
        (231/255, 218/255, 210/255),   # LLM-MLP
        (99/255, 227/255, 152/255),    # LLM-Attn
    ]
    bar_labels = [
        "Projector", "ViT", "ViT-MLP", "ViT-Attn",
        "LLM", "LLM-MLP", "LLM-Attn", "Total"
    ]
    color_order = [0, 1, 4, 5, 2, 6, 7, 3]
    bar_colors = [rgb_colors[i] for i in color_order]

    reorder_idx = [0, 1, 3, 4, 2, 5, 6, 7]
    weights = weights[:, reorder_idx]

    bar_width = 0.10
    x = np.arange(len(model_labels))

    plt.figure(figsize=(16, 6))
    ax = plt.gca()

    for i, (component, color) in enumerate(zip(bar_labels, bar_colors)):
        bars = plt.bar(
            x + i * bar_width - (len(bar_labels))/2 * bar_width,
            weights[:, i],
            width=bar_width,
            color=color,
            label=component
        )
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.3f}",
                ha='center',
                va='bottom',
                fontsize=9
            )

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.yaxis.grid(True, color='#cccccc', linestyle='-', linewidth=1, alpha=0.7)
    ax.set_axisbelow(True)

    plt.ylabel("Billions of Weights", fontsize=12)
    plt.title("Parameter Distribution of Model Components", fontsize=18, pad=15, loc='left')
    plt.xticks(x, model_labels, fontsize=11)
    plt.xlabel("")

    plt.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.05),
        ncol=8,
        frameon=False,
        fontsize=13
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
    internvl_mapping = {
        "Projector": "mlp",
        "ViT": "vision_model",
        "LLM": "language_model",
    }
    gemma_mapping = {
        "Projector": "model.multi_modal_projector",
        "ViT": "model.vision_tower",
        "LLM": "model.language_model",
    }

    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    qwen_weights = analyze_model_params(qwen_model, qwen_mapping, "Qwen2.5-VL-3B")

    internvl_model = AutoModel.from_pretrained(
        "OpenGVLab/InternVL3-2B",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
    ).eval().cuda()
    internvl_weights = analyze_model_params(internvl_model, internvl_mapping, "InternVL3-2B")

    gemma_model = Gemma3ForConditionalGeneration.from_pretrained(
        "google/gemma-3-4b-it", 
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
    ).eval()
    gemma_weights = analyze_model_params(gemma_model, gemma_mapping, "Gemma3-4B")

    model_labels = ["Qwen2.5-VL-3B", "InternVL3-2B", "Gemma3-4B"]
    components = list(qwen_mapping.keys())
    extra_keys = ["ViT-MLP", "ViT-Attn", "LLM-MLP", "LLM-Attn"]
    weights = np.array([qwen_weights, internvl_weights, gemma_weights])
    plot_weights_histogram(model_labels, components, extra_keys, weights)
