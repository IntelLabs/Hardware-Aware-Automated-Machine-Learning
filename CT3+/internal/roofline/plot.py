import matplotlib.pyplot as plt

batch_sizes = [1, 2, 4, 8, 16]
qwen2_5_vl_3b_instruct_prefill = [210.42308, 420.8461201, 841.6922004, 1683.384361, 3366.768682]
gemma_3_4b_it_prefill = [191.3870388, 382.7739911, 765.5478958, 1531.095705, 3062.191324]

qwen2_5_vl_3b_instruct_decode = [6.575879282, 13.15159929, 26.3030393, 52.60591932, 105.2116794]
gemma_3_4b_it_decode = [1.805061746, 3.61003784, 7.21999003, 14.43989441, 28.87970317]

plt.figure(figsize=(8, 6))

# Qwen2.5-VL-3B-Instruct
plt.plot(batch_sizes, qwen2_5_vl_3b_instruct_prefill, marker='o', color='tab:blue', label='Qwen2.5-VL-3B-Instruct Prefill', linewidth=2)
plt.plot(batch_sizes, qwen2_5_vl_3b_instruct_decode, marker='o', color='tab:blue', label='Qwen2.5-VL-3B-Instruct Decode', linewidth=2, linestyle='--')

# Gemma-3-4B-it
plt.plot(batch_sizes, gemma_3_4b_it_prefill, marker='o', color='tab:orange', label='Gemma-3-4B-it Prefill', linewidth=2)
plt.plot(batch_sizes, gemma_3_4b_it_decode, marker='o', color='tab:orange', label='Gemma-3-4B-it Decode', linewidth=2, linestyle='--')

for x, y in zip(batch_sizes[2:], qwen2_5_vl_3b_instruct_prefill[2:]):
    plt.text(x, y, f'{y:.0f}', fontsize=12, fontweight='bold', ha='left', va='bottom', color='tab:blue')
for x, y in zip(batch_sizes[2:], gemma_3_4b_it_prefill[2:]):
    plt.text(x, y, f'{y:.0f}', fontsize=12, fontweight='bold', ha='right', va='top', color='tab:orange')

for x, y in zip(batch_sizes[2:], qwen2_5_vl_3b_instruct_decode[2:]):
    plt.text(x, y, f'{y:.1f}', fontsize=12, fontweight='bold', ha='left', va='bottom', color='tab:blue')
for x, y in zip(batch_sizes[2:], gemma_3_4b_it_decode[2:]):
    plt.text(x, y, f'{y:.1f}', fontsize=12, fontweight='bold', ha='right', va='top', color='tab:orange')

plt.xlabel('Batch size', fontsize=16, fontweight='bold')
plt.ylabel('FLOPs / Weights', fontsize=16, fontweight='bold')
plt.title('Arithmetic Intensity: Qwen2.5-VL-3B-Instruct vs. Gemma-3-4B-it', fontsize=14, fontweight='bold', pad=45)

plt.legend(fontsize=13, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.13), ncol=2)

plt.grid(True)

for spine in plt.gca().spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.savefig('flops_weights_plot.pdf', format='pdf')
