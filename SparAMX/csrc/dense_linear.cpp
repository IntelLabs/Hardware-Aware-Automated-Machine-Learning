#include <torch/extension.h>
#include <torch/nn/functional/linear.h>
#include <optional>
#include <torch/torch.h>
#include <cstdint>
#include <iostream>

#include <immintrin.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <stdbool.h>
#include <thread>
#define _OPENMP
#include <ATen/ParallelOpenMP.h>

namespace F = torch::nn::functional;

#define ARCH_REQ_XCOMP_PERM     0x1023
#define XFEATURE_XTILEDATA      18
#define MAX_OUT_ROWS            16
#define MAX_INNER_DIM           32
#define MAX_OUT_COLS            16
// #define VERBOSE

//Define tile config data structure 
typedef struct __tile_config
{
  uint8_t palette_id;
  uint8_t start_row;
  // Updated to match the data here: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=loadconfig&ig_expand=6875,6866,6875,6877
  // Specifically, using 8 tiles instead of 16 like the article assumes.
  uint8_t reserved_0[14];
  uint16_t colsb[8]; 
  uint8_t reserved_1[16];
  uint8_t rows[8];
  uint8_t reserved_2[8];
} __tilecfg;

/* Initialize tile config */
static void init_tile_config (__tilecfg *tileinfo, int out_rows, int inner_dim, int out_cols)
{
  tileinfo->palette_id = 1;
  tileinfo->start_row = 0;

  // Destination tiles (For the output)
  // Each item in the result is 32 bits ==> 4 Bytes
  for (int i = 0; i < 4; i++) {
    tileinfo->colsb[i] = out_cols * 4;
    tileinfo->rows[i] =  out_rows;
  }



  // SRC 1 (Matrix A) tile
  tileinfo->colsb[4] = inner_dim * 2;
  tileinfo->rows[4] =  out_rows;

  tileinfo->colsb[5] = inner_dim * 2;
  tileinfo->rows[5] =  out_rows;

  // SRC 2 (Matrix B) tile
  // This tile is tricky. Each 2 consecutive items are considered to belong to the same column.
  tileinfo->colsb[6] = out_cols * 2 * 2; 
  tileinfo->rows[6] =  inner_dim / 2;

  tileinfo->colsb[7] = out_cols * 2 * 2; 
  tileinfo->rows[7] =  inner_dim / 2;

  _tile_loadconfig (tileinfo);
}

/* Set_tiledata_use() - Invoke syscall to set ARCH_SET_STATE_USE */
static bool set_tiledata_use()
{
   if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) 
   {
      #ifdef VERBOSE
        std::cout << ("\n Fail to do XFEATURE_XTILEDATA \n\n");
      #endif
      return false;
   }
   else
   {
      #ifdef VERBOSE
        std::cout << ("\n TILE DATA USE SET - OK \n\n");
      #endif
      return true;
   }

   return true;
}


// intent is to create an alternative kernel for linear layer by using torch C API
// just to demonstrate the capability of torch custom op extension
// we will not implement backward since we focus of inference
// we can also reuse parent implementation if we inherit nn.Linear

// implementation below is adapted from
// https://github.com/pytorch/pytorch/blob/main/torch/csrc/api/include/torch/nn/functional/linear.h

void dense_linear_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    torch::Tensor&       output,
    const c10::optional<torch::Tensor>& bias = c10::nullopt) {
  if (true) {

  float* res = output.data_ptr<float>();
   __tilecfg tile_data = {0};
  const int out_rows = input.sizes()[0];
  const int inner_dim = input.sizes()[1];
  const int out_cols = weight.sizes()[0];

  const int tile_out_rows = std::min(out_rows, MAX_OUT_ROWS);
  const int tile_inner_dim = std::min(inner_dim, MAX_INNER_DIM);
  const int tile_out_cols = std::min(out_cols, MAX_OUT_COLS);

   // Request permission to linux kernel to run AMX 
   if (!set_tiledata_use())
      exit(-1);

    #ifdef VERBOSE
      std::cout << "Loading Tile Configuration.\n";
    #endif
   // Load tile configuration 
  //  init_tile_config (&tile_data, tile_out_rows, tile_inner_dim, tile_out_cols);

    const int64_t num_chunks = (out_rows + MAX_OUT_ROWS - 1) / MAX_OUT_ROWS;
    at::parallel_for(0, num_chunks, 0, [&](int64_t start_r, int64_t end_r) {
      for (int64_t out_row_i = start_r; out_row_i < end_r; out_row_i+=2) {
        const bool extra_row = (out_row_i + 1 < end_r);
        const int64_t num_chunks_2 = (out_cols + MAX_OUT_COLS - 1) / MAX_OUT_COLS;
        at::parallel_for(0, num_chunks_2, 0, [&](int64_t start_c, int64_t end_c) {
          // std::cout << "Thread ID: " << omp_get_thread_num() << std::endl;
          for (int64_t out_col_i = start_c; out_col_i < end_c; out_col_i+=2) {
            init_tile_config (&tile_data, tile_out_rows, tile_inner_dim, tile_out_cols);
            #ifdef VERBOSE
              std::cout << "Computing Tile Pair: (" << out_row_i << "," << out_col_i << ")\n";
              std::cout << ("Loading res into AMX registers\n");
            #endif

              const bool extra_col = (out_col_i + 1 < end_c);

              // _tile_loadd (0, res + out_row_i * MAX_OUT_ROWS * out_cols + out_col_i * MAX_OUT_COLS, (out_cols) * 4);
              // if (enable_second) {
              //   _tile_loadd (3, res + out_row_i * MAX_OUT_ROWS * out_cols + (out_col_i+1) * MAX_OUT_COLS, (out_cols) * 4);
              // }
            #ifdef VERBOSE
              std::cout << "Result Before: \n";

              for (int i = 0; i < out_rows; i++) {
                  for (int j = 0; j < out_cols; j++) {
                      std::cout << " " << (float) * (res + i * out_cols + j);
                  }
                  std::cout << "\n";
              }
            #endif

              // TODO: Would parallelizing this make things faster? We'd need to add results after all threads finish.
              for (int inner_dim_i = 0; inner_dim_i < (inner_dim + MAX_INNER_DIM - 1) / MAX_INNER_DIM; inner_dim_i++) {
                #ifdef VERBOSE
                  std::cout << "Computing Tile: (" << out_row_i << "," << out_col_i << ")->" << inner_dim_i << "\n";

                  std::cout << ("Loading Elements into AMX registers\n");

                  // Load tile rows from memory
                  std::cout << ("Loading src1 into AMX registers\n");
                #endif

                  _tile_loadd (4, input.data_ptr() + 2 * (out_row_i * tile_out_rows * inner_dim + inner_dim_i * tile_inner_dim), inner_dim * 2);

                  if (extra_row) {
                    _tile_loadd (5, input.data_ptr() + 2 * ((out_row_i + 1) * tile_out_rows * inner_dim + inner_dim_i * tile_inner_dim), inner_dim * 2);
                  }
                #ifdef VERBOSE
                  std::cout << ("Loading src2 into AMX registers\n");
                #endif
                  _tile_loadd (6, weight.data_ptr() + 2 * (inner_dim_i * tile_inner_dim / 2 * out_cols * 2 + out_col_i * tile_out_cols * 2), out_cols * 2 * 2 );

                  if (extra_col) {
                  _tile_loadd (7, weight.data_ptr() + 2 * (inner_dim_i * tile_inner_dim / 2 * out_cols * 2 + (out_col_i+1) * tile_out_cols * 2), out_cols * 2 * 2 );
                  }
                #ifdef VERBOSE
                  std::cout << ("Performing the computation\n");
                #endif

                  // Compute dot-product of bytes in tiles
                  _tile_dpbf16ps (0, 4, 6);
                  _tile_stored (0, res + out_row_i * tile_out_rows * out_cols + out_col_i * tile_out_cols, (out_cols) * 4);
                  if (extra_col) {
                    _tile_dpbf16ps (1, 4, 7);
                    _tile_stored (1, res + out_row_i * tile_out_rows * out_cols + (out_col_i + 1) * tile_out_cols, (out_cols) * 4);
                  }
                  if (extra_row) {
                    _tile_dpbf16ps (2, 5, 6);
                    _tile_stored (2, res + (out_row_i + 1) * tile_out_rows * out_cols + out_col_i * tile_out_cols, (out_cols) * 4);              
                  }
                  if (extra_col && extra_row) {
                    _tile_dpbf16ps (3, 5, 7);
                    _tile_stored (3, res + (out_row_i + 1) * tile_out_rows * out_cols + (out_col_i+ 1) * tile_out_cols, (out_cols) * 4);
                  }

              }
              #ifdef VERBOSE
                std::cout << ("Storing the result in memory\n");
              #endif
              // Store the tile data to memory

              #ifdef VERBOSE
                std::cout << "Result After: \n";

                for (int i = 0; i < out_rows; i++) {
                    for (int j = 0; j < out_cols; j++) {
                        std::cout << " " << (float) * (res + i * out_cols + j);
                    }
                    std::cout << "\n";
                }
              #endif
        }
      }
        );}
    });

   // Release the tile configuration to return to the init state, 
   // which releases all storage it currently holds
   _tile_release ();
    // fused op is marginally faster
    // auto t_res = torch::from_blob(res, {out_rows,out_cols}, torch::kFloat32).to(torch::kBFloat16);
    // return t_res;
  }
}

torch::Tensor dense_matmul(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias = c10::nullopt) {
      const int batch_size = input.sizes()[0];
      const int num_heads = input.sizes()[1];
      auto out = torch::empty({batch_size, num_heads, input.sizes()[2], weight.sizes()[2]}, torch::TensorOptions().dtype(torch::kFloat32).memory_format(torch::MemoryFormat::Contiguous));      
      float* out_ptr = static_cast<float*>(out.data_ptr());

      for (int batch_i = 0; batch_i < batch_size; batch_i++) {
        at::parallel_for(0, num_heads, 0, [&](int start_h, int end_h) {
        for (int head_i = start_h; head_i < end_h; head_i++) {
          float* current_out_ptr = out_ptr + 
                (batch_i * num_heads * input.sizes()[2] * weight.sizes()[2] + 
                head_i * input.sizes()[2] * weight.sizes()[2]);
          auto out_tensor = torch::from_blob(current_out_ptr, {input.sizes()[2], weight.sizes()[2]}, torch::kFloat32);
          dense_linear_forward(input[batch_i][head_i], weight[batch_i][head_i], out_tensor, bias);
        }
        }); // Head parallelization.
      }

      return out.to(torch::kBFloat16);
    }


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &dense_linear_forward, "dense linear forward",
  pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("res"), pybind11::arg("bias") = nullptr);

  m.def("matmul", &dense_matmul, "dense matmul",
  pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("bias") = nullptr);
}