#include <immintrin.h>
#include <torch/extension.h>
#include <torch/nn/functional/linear.h>
#include <optional>
#include <torch/torch.h>
#include <cstdint>
#include <iostream>

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <stdbool.h>
#include <thread>
#include <omp.h>

#define _OPENMP
#include <ATen/ParallelOpenMP.h>
#include <cstdlib>
// uncomment to disable assert()
// #define NDEBUG
// #include <cassert>

namespace F = torch::nn::functional;

#define MAX_OUT_ROWS            1
#define MAX_INNER_DIM           2
#define MAX_OUT_COLS            16
// CHANGE BELOW FOR GROUP SIZE
#define NUM_COL_GROUPS          32
// #define VERBOSE
// #define TIMEDEBUG
#define CUSTOM_CLOCKS_PER_SEC 1
#include <time.h>
constexpr size_t CACHE_LINE_SIZE = 64;
constexpr size_t ELEMENT_SIZE = 2;  // 2 bytes per element

#define METADATA_CHUNK_SIZE 16  // Number of metadata items to load at once


class AlignedMemory {
public:
    AlignedMemory(size_t size, size_t alignment) {
        if (posix_memalign(&ptr_, alignment, size) != 0) {
            throw std::runtime_error("Memory allocation failed");
        }
    }

    ~AlignedMemory() {
        free(ptr_);
    }

    void* get() const { return ptr_; }

    // Prevent copying
    AlignedMemory(const AlignedMemory&) = delete;
    AlignedMemory& operator=(const AlignedMemory&) = delete;

private:
    void* ptr_;
};

#pragma pack(push, 1)
alignas(64) union Vec512iUnion {
    __m512i vec;
    uint32_t arr[16];
};
#pragma pack(pop)


__m512i parallel_prefix_sum_32(__m512i v) {
    // Shift right by 1 int32 and add
    __m512i sum = _mm512_add_epi32(v, _mm512_alignr_epi32(v, _mm512_setzero_si512(), 15));
    
    // Shift right by 2 int32s and add
    sum = _mm512_add_epi32(sum, _mm512_alignr_epi32(sum, _mm512_setzero_si512(), 14));
    
    // Shift right by 4 int32s and add
    sum = _mm512_add_epi32(sum, _mm512_alignr_epi32(sum, _mm512_setzero_si512(), 12));
    
    // Shift right by 8 int32s and add
    return _mm512_add_epi32(sum, _mm512_alignr_epi32(sum, _mm512_setzero_si512(), 8));
}

torch::Tensor sparse_linear_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight_metadata,
    const torch::Tensor& weight_values,
    const torch::Tensor& weight_values_is,
    const int            out_cols,
    const int            weight_values_start = 0,
    const c10::optional<torch::Tensor>& bias = c10::nullopt) {

    #ifdef TIMEDEBUG
      clock_t start, end;
    #endif

    #ifdef TIMEDEBUG
      start = clock();
    #endif

    const int out_rows = input.sizes()[0] * input.sizes()[1];
    const int inner_dim = input.sizes()[2];
    const int res_size = (out_rows * out_cols);
    std::unique_ptr<AlignedMemory> res = std::make_unique<AlignedMemory>(res_size * 4, CACHE_LINE_SIZE);

    const int tile_out_rows = std::min(out_rows, MAX_OUT_ROWS);
    const int tile_inner_dim = std::min(inner_dim, MAX_INNER_DIM);
    // TODO: Support for smaller inner_dim.
    // assert(tile_inner_dim == MAX_INNER_DIM);
    const int tile_out_cols = std::min(out_cols, MAX_OUT_COLS);

    const int num_inner_threads = weight_values_is.sizes()[0];

    const int start_ir = 0, end_ir = tile_inner_dim / 2;
    const int start_t = 0, end_t = num_inner_threads;
    const int num_col_tiles = (out_cols + MAX_OUT_COLS - 1) / MAX_OUT_COLS;
    const int col_tiles_per_thread = (num_col_tiles + num_inner_threads - 1) / num_inner_threads;

    #ifdef TIMEDEBUG
      end = clock();
      std::cout << "Time taken to initialize variables: " << ((double)(end - start)) / CUSTOM_CLOCKS_PER_SEC << "\n";
    #endif

    const int num_chunks = (out_rows + MAX_OUT_ROWS - 1) / MAX_OUT_ROWS;
    int start_r = 0, end_r = num_chunks;

    for (int out_row_i = start_r; out_row_i < end_r; out_row_i++) {
        at::parallel_for(0, num_inner_threads, 0, [&](int start_t, int end_t) {
            for (int t_id = start_t; t_id < end_t; t_id++) {
                int cur_weight_i = weight_values_is[t_id].item<int>();

                const int start_c = t_id * col_tiles_per_thread;
                const int end_c = std::min((t_id + 1) * col_tiles_per_thread, num_col_tiles);

                // Vec512iUnion metadata;

                int weight_index_bias = t_id * col_tiles_per_thread * tile_out_cols * inner_dim / 32;

                for (int out_col_i = start_c; out_col_i < end_c; out_col_i+= NUM_COL_GROUPS) {
                    // std::cout << "out_col_i: " << out_col_i << "\n";
                    #ifdef TIMEDEBUG
                      start = clock();
                    #endif

                    #ifdef TIMEDEBUG
                      end = clock();
                      std::cout << "Time taken to init variables: " << ((double)(end - start)) / CUSTOM_CLOCKS_PER_SEC << "\n";
                    #endif

                    const int start_id = 0,  end_id = (inner_dim + tile_inner_dim - 1) / tile_inner_dim;
                    // __m512 acc0 = _mm512_setzero_ps();
                    const int num_col_groups = std::min(NUM_COL_GROUPS, end_c - out_col_i);
                    // std::cout << "num_col_groups: " << num_col_groups << "\n";
                    __m512 acc[num_col_groups];
                    for (int i = 0; i < num_col_groups; i++) {
                        acc[i] = _mm512_setzero_ps();
                    }

                    Vec512iUnion metadata[2], popcnt[2];

                    const int items_to_load = std::min(num_col_groups, METADATA_CHUNK_SIZE);
                    const int items_to_load2 = num_col_groups - METADATA_CHUNK_SIZE;

                    metadata[0].vec = _mm512_maskz_loadu_epi32((1ULL << items_to_load) - 1,
                      (unsigned int *)weight_metadata.data_ptr() + weight_index_bias);

                    for (int inner_dim_i = start_id; inner_dim_i < end_id; inner_dim_i++) {
                        #ifdef TIMEDEBUG
                          start = clock();
                        #endif

                        // TODO: Can we optimize this to do one load of a set of inputs and use them when needed?
                        __m512bh input_reg = (__m512bh)_mm512_set1_epi32(*(int*)((short*)input.data_ptr() + out_row_i * inner_dim + inner_dim_i * tile_inner_dim));
                        // _mm_prefetch (((short*)input.data_ptr() + out_row_i * inner_dim + (inner_dim_i+1) * tile_inner_dim), _MM_HINT_T1);

                        // __m256 acc1 = _mm256_setzero_ps();
                        // __m256 acc2 = _mm256_setzero_ps();
                        // __m256 acc3 = _mm256_setzero_ps();

                        // the / 32 is because each metadata item has 32 weights.
                        // if (grouped_col_tiles) {
                        //     weight_index_bias = (tile_inner_dim * (inner_dim_i + out_col_i * ((inner_dim + tile_inner_dim - 1) / tile_inner_dim / 2)));
                        // } else {
                        //     weight_index_bias = (tile_inner_dim / 2 * ((inner_dim_i) + (out_col_i) * ((inner_dim + tile_inner_dim - 1) / tile_inner_dim)));
                        // }

                        // TODO: This assumes that num_col_groups is always <= METADATA_CHUNK_SIZE.

                        if (num_col_groups > METADATA_CHUNK_SIZE) {
                            metadata[1].vec = _mm512_maskz_loadu_epi32((1ULL << items_to_load2) - 1,
                                        (unsigned int *)weight_metadata.data_ptr() + weight_index_bias + METADATA_CHUNK_SIZE);
                        } else {
                          metadata[0].vec = _mm512_maskz_loadu_epi32((1ULL << items_to_load) - 1,
                            (unsigned int *)weight_metadata.data_ptr() + weight_index_bias);
                        }
                        // _mm_prefetch ((unsigned int *)weight_metadata.data_ptr() + weight_index_bias + num_col_groups, _MM_HINT_T0);
                        popcnt[0].vec = parallel_prefix_sum_32(_mm512_popcnt_epi32(metadata[0].vec));
                        // metadata.vec = _mm512_loadu_si512((unsigned int *)weight_metadata.data_ptr() + weight_index_bias);

                        acc[0] = _mm512_dpbf16_ps(acc[0], (__m512bh)_mm512_maskz_expandloadu_epi16(metadata[0].arr[0], (short *)weight_values.data_ptr() + weight_values_start + cur_weight_i), input_reg);
                        // cur_weight_i+= __builtin_popcount(metadata.arr[0]);
                        // #pragma unroll(15)
                        for (int col_group_i = 1; col_group_i < items_to_load; col_group_i++) {
                            // metadata.vec =  _mm512_loadu_si512((unsigned int *)weight_metadata.data_ptr() + weight_index_bias);
                            // unsigned int weight_metadata_item = metadata.arr[col_group_i];
                            // std::cout << "weight_metadata_item: " << weight_metadata_item << "\n";
                            // unsigned int weight_metadata_item = weight_metadata[weight_index_bias].item<unsigned int>();
                            // 32 weights, each one is 16 bits.
                            // cur_weight_i += __builtin_popcount(weight_metadata_item);
                            // 32 inputs, each one is 16 bits.
                            // __m512bh input_reg = (__m512bh)_mm512_loadu_ps((short *) input.data_ptr() + out_row_i * inner_dim + inner_dim_i * tile_inner_dim);
                            acc[col_group_i] = _mm512_dpbf16_ps(acc[col_group_i], (__m512bh)_mm512_maskz_expandloadu_epi16(metadata[0].arr[col_group_i], (short *)weight_values.data_ptr() + weight_values_start + cur_weight_i + popcnt[0].arr[col_group_i - 1]), input_reg);
                            // cur_weight_i+= __builtin_popcount(metadata.arr[col_group_i]);
                        }
                        cur_weight_i+= popcnt[0].arr[items_to_load - 1];

                        if (num_col_groups > METADATA_CHUNK_SIZE) {
                            if (inner_dim_i < end_id - 1) {
                                metadata[0].vec = _mm512_maskz_loadu_epi32((1ULL << items_to_load) - 1,
                                            (unsigned int *)weight_metadata.data_ptr() + weight_index_bias + num_col_groups);
                            }
                            popcnt[1].vec = parallel_prefix_sum_32(_mm512_popcnt_epi32(metadata[1].vec));

                            acc[METADATA_CHUNK_SIZE] = _mm512_dpbf16_ps(acc[METADATA_CHUNK_SIZE], (__m512bh)_mm512_maskz_expandloadu_epi16(metadata[1].arr[0], (short *)weight_values.data_ptr() + weight_values_start + cur_weight_i), input_reg);
                            // cur_weight_i+= __builtin_popcount(metadata.arr[0]);
                            for (int col_group_i = 1; col_group_i < items_to_load2; col_group_i++) {
                                acc[METADATA_CHUNK_SIZE + col_group_i] = _mm512_dpbf16_ps(acc[METADATA_CHUNK_SIZE + col_group_i], (__m512bh)_mm512_maskz_expandloadu_epi16(metadata[1].arr[col_group_i], (short *)weight_values.data_ptr() + weight_values_start + cur_weight_i + popcnt[1].arr[col_group_i - 1]), input_reg);
                                // cur_weight_i+= __builtin_popcount(metadata.arr[col_group_i]);
                            }
                          cur_weight_i+= popcnt[1].arr[items_to_load2 - 1];
                        }

                        // if (col_group_i == METADATA_CHUNK_SIZE) {
                        //   items_to_load = std::min(num_col_groups - METADATA_CHUNK_SIZE, METADATA_CHUNK_SIZE);
                        //   metadata.vec = _mm512_maskz_loadu_epi32((1ULL << items_to_load) - 1,
                        //             (unsigned int *)weight_metadata.data_ptr() + weight_index_bias + METADATA_CHUNK_SIZE);
                        //   popcnt.vec = parallel_prefix_sum_32(_mm512_popcnt_epi32(metadata.vec));
                        // }


                        weight_index_bias += num_col_groups;

                        // for (int row = start_ir; row < end_ir; row++) {
                        //     unsigned int weight_metadata_item = metadata.arr[row];
                        //     __m512bh weight_reg = (__m512bh)_mm512_maskz_expandloadu_epi16(weight_metadata_item, (short *)weight_values.data_ptr() + cur_weight_i);
                        //     cur_weight_i += __builtin_popcount(weight_metadata_item);

                        //     // TODO: Need to revisit this to make sure we're getting the correct input values.
                        //     acc0 = _mm512_dpbf16_ps(acc0, weight_reg, (__m512bh)_mm512_loadu_ps((unsigned int *)input.data_ptr() + 2 * ((out_row_i + row) * tile_out_rows * inner_dim + inner_dim_i * tile_inner_dim + out_col_i * 16)));

                        //     // if (extra_col) {
                        //     //     weight_float = _mm256_cvtph_ps(_mm256_extracti128_si256(weight_reg, 1));
                        //     //     acc1 = _mm256_fmadd_ps(input_reg, weight_float, acc1);
                        //     // }

                        //     // if (extra_row) {
                        //     //     input_reg = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(input.data_ptr() + 2 * ((out_row_i + 1) * tile_out_rows * inner_dim + inner_dim_i * tile_inner_dim + row * 8))));
                        //     //     weight_float = _mm256_cvtph_ps(_mm256_castsi256_si128(weight_reg));
                        //     //     acc2 = _mm256_fmadd_ps(input_reg, weight_float, acc2);

                        //     //     if (extra_col) {
                        //     //         weight_float = _mm256_cvtph_ps(_mm256_extracti128_si256(weight_reg, 1));
                        //     //         acc3 = _mm256_fmadd_ps(input_reg, weight_float, acc3);
                        //     //     }
                        //     // }
                        // }

                        // if (extra_col)
                        //     _mm256_storeu_ps((float*)res->get() + out_row_i * tile_out_rows * out_cols + (out_col_i+1) * tile_out_cols, acc1);
                        // if (extra_row)
                        //     _mm256_storeu_ps((float*)res->get() + (out_row_i+1) * tile_out_rows * out_cols + out_col_i * tile_out_cols, acc2);
                        // if (extra_col && extra_row)
                        //     _mm256_storeu_ps((float*)res->get() + (out_row_i+1) * tile_out_rows * out_cols + (out_col_i+1) * tile_out_cols, acc3);

                        #ifdef TIMEDEBUG
                          end = clock();
                          std::cout << "Time taken to perform computation and store: " << ((double)(end - start)) / CUSTOM_CLOCKS_PER_SEC << "\n";
                        #endif
                    }

                    // Store acc0 in the correct position in res.
                    for (int col_group_i = 0; col_group_i < num_col_groups; col_group_i++) {
                        _mm512_storeu_ps((float*)res->get() + out_row_i * out_cols + out_col_i * tile_out_cols + col_group_i * 16, acc[col_group_i]);
                    }
                }
            }
        });
    }


    auto t_res = torch::from_blob(res->get(), {input.sizes()[0], input.sizes()[1], out_cols}, torch::kFloat32).to(torch::kBFloat16);
    return t_res;
}


torch::Tensor sparse_matmul(
    const torch::Tensor& input,
    const torch::Tensor& weight_metadata,
    const torch::Tensor& weight_values,
    const torch::Tensor& weight_values_bs,
    const torch::Tensor& weight_values_is,
    const int            out_cols,
    const c10::optional<torch::Tensor>& bias = c10::nullopt) {
      const int batch_size = input.sizes()[0];
      const int num_heads = input.sizes()[1];
      auto out = torch::empty({batch_size, num_heads, input.sizes()[2], out_cols}, torch::kBFloat16);
      for (int batch_i = 0; batch_i < batch_size; batch_i++) {
        at::parallel_for(0, num_heads, 0, [&](int start_h, int end_h) {
        for (int head_i = start_h; head_i < end_h; head_i++) {
          out[batch_i][head_i] = sparse_linear_forward(input[batch_i][head_i].unsqueeze(0), weight_metadata[batch_i][head_i], weight_values, weight_values_is[batch_i][head_i], out_cols, weight_values_bs[batch_i][head_i].item<int>(), bias).squeeze(0);
        }
        }); // Head parallelization.
      }

      return out;
    }


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sparse_linear_forward, "sparse linear forward",
  pybind11::arg("input"), pybind11::arg("weight_metadata"), pybind11::arg("weight_values"), pybind11::arg("weight_values_is"), pybind11::arg("out_cols"), pybind11::arg("weight_values_start") = 0, pybind11::arg("bias") = nullptr);


  m.def("matmul", &sparse_matmul, "sparse matmul",
  pybind11::arg("input"), pybind11::arg("weight_metadata"), pybind11::arg("weight_values"), pybind11::arg("weight_values_bs") ,pybind11::arg("weight_values_is"), pybind11::arg("out_cols"), pybind11::arg("bias") = nullptr);
}
