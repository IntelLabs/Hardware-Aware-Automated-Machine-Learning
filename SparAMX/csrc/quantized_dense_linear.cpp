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
#include <omp.h>

#define _OPENMP
#include <ATen/ParallelOpenMP.h>
#include <cstdlib>
namespace F = torch::nn::functional;

#define ARCH_REQ_XCOMP_PERM     0x1023
#define XFEATURE_XTILEDATA      18
#define MAX_OUT_ROWS            16 // How many rows you can process in one tile.
#define MAX_INNER_DIM           64 // How many inner dimensions you can process in one tile. 4 per row and there are 16 rows.
#define MAX_OUT_COLS            16 // How many columns you can process in one tile. Always 16 because we have 64 bytes divided into 4 bytes per column regardless of the datatype.
#define NUM_INT_IN_CACHE_LINE   16 // Number of integers in a cache line.
// #define VERBOSE
// #define TIMEDEBUG
#define CUSTOM_CLOCKS_PER_SEC 1
#include <time.h> 
constexpr size_t CACHE_LINE_SIZE = 64;
constexpr size_t ELEMENT_SIZE = 1;  // 1 byte per element


// #define TIME_OPERATION(name, code_block) \
//     { \
//         auto start = clock(); \
//         int cnt = 0; \
//         for (int i = 0; i < 1; i++) { \
//             code_block \
//             cnt++; \
//         } \
//         auto end = clock(); \
//         std::cout << cnt << " Clocks taken by " << name << ": " << ((double)(end - start)) / CUSTOM_CLOCKS_PER_SEC << "\n"; \
//     }

#define TIME_OPERATION(name, code_block) { code_block }

#include <chrono>

using namespace std::chrono;


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
  tileinfo->colsb[4] = inner_dim * 1; // 1 byte per element in case of INT8.
  tileinfo->rows[4] =  out_rows;

  tileinfo->colsb[5] = inner_dim * 1;
  tileinfo->rows[5] =  out_rows;

  // SRC 2 (Matrix B) tile
  // This tile is tricky. Each 4 consecutive items are considered to belong to the same column so we have 4 elements per col. Each element takes only 1 byte.
  tileinfo->colsb[6] = out_cols * 4 * 1; 
  tileinfo->rows[6] =  inner_dim / 4;

  tileinfo->colsb[7] = out_cols * 4 * 1; 
  tileinfo->rows[7] =  inner_dim / 4;

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

#pragma pack(push, 1)
alignas(64) union Vec512iUnion {
    __m512i vec;
    uint64_t arr[8];
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

__m512i parallel_prefix_sum_64(__m512i v) {
    // With 64-bit integers, we have 8 elements total in a 512-bit vector
    // instead of 16 elements with 32-bit integers
    
    // Shift right by 1 int64 and add
    __m512i sum = _mm512_add_epi64(v, _mm512_alignr_epi64(v, _mm512_setzero_si512(), 7));
    
    // Shift right by 2 int64s and add
    sum = _mm512_add_epi64(sum, _mm512_alignr_epi64(sum, _mm512_setzero_si512(), 6));
    
    // Shift right by 4 int64s and add
    return _mm512_add_epi64(sum, _mm512_alignr_epi64(sum, _mm512_setzero_si512(), 4));
}


// intent is to create an alternative kernel for linear layer by using torch C API
// just to demonstrate the capability of torch custom op extension
// we will not implement backward since we focus of inference
// we can also reuse parent implementation if we inherit nn.Linear

// implementation below is adapted from
// https://github.com/pytorch/pytorch/blob/main/torch/csrc/api/include/torch/nn/functional/linear.h

torch::Tensor quantized_dense_linear_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias = c10::nullopt) {

    #ifdef TIMEDEBUG
      clock_t start, end;
    #endif

    #ifdef TIMEDEBUG
      start = clock();
    #endif
  const int out_rows = input.sizes()[0] * input.sizes()[1];
  const int inner_dim = input.sizes()[2];
  const int out_cols = weight.sizes()[0];
  const int res_size = (out_rows * out_cols);
  // std::cout << "Sometimes I reach here.";
  std::unique_ptr<AlignedMemory> res = std::make_unique<AlignedMemory>(res_size * 4, CACHE_LINE_SIZE);
  // float * res = new float[res_size]();

  //  int weight_index;
  // std::cout << "out_rows: " << out_rows << ", inner_dim: " << inner_dim << ", out_cols: " << out_cols << "\n";
   
  // const int start_t = 0, end_t = num_inner_threads;

  //  short weight[MAX_OUT_ROWS * MAX_INNER_DIM] = {0};


   // Request permission to linux kernel to run AMX 
   if (!set_tiledata_use())
      exit(-1);

    #ifdef TIMEDEBUG
      end = clock();
      std::cout << "Time taken to initialize variables and tile: " << ((double)(end - start)) / CUSTOM_CLOCKS_PER_SEC << "\n";
    #endif
    #ifdef VERBOSE
      std::cout << "Loading Tile Configuration.\n";
    #endif

   // Load tile configuration 
  //  init_tile_config (&tile_data, tile_out_rows, tile_inner_dim, tile_out_cols);

    const int num_chunks = (out_rows + MAX_OUT_ROWS - 1) / MAX_OUT_ROWS;
    int start_r = 0, end_r = num_chunks;
    // at::parallel_for(0, num_chunks, 0, [&](int start_r, int end_r) {
      for (int out_row_i = start_r; out_row_i < end_r; out_row_i+=2) {

            const int tile_out_rows = std::min(out_rows, MAX_OUT_ROWS);
            const int tile_inner_dim = std::min(inner_dim, MAX_INNER_DIM);
            const int tile_out_cols = std::min(out_cols, MAX_OUT_COLS);

            const int start_ir = 0, end_ir = tile_inner_dim / 4;
            const int num_col_tiles = (out_cols + MAX_OUT_COLS - 1) / MAX_OUT_COLS;

            __tilecfg tile_data = {0};
        const int64_t num_chunks_2 = (out_cols + MAX_OUT_COLS - 1) / MAX_OUT_COLS;
        // int start_c = 0, end_c = num_chunks_2;
        at::parallel_for(0, num_chunks_2, 0, [&](int start_c, int end_c) {
          // std::cout << "Thread ID: " << omp_get_thread_num() << std::endl; 
          for (int out_col_i = start_c; out_col_i < end_c; out_col_i+=2) {

            init_tile_config (&tile_data, tile_out_rows, tile_inner_dim, tile_out_cols);
            #ifdef TIMEDEBUG
              start = clock();
            #endif
            
            const bool extra_row = (out_row_i + 1 < end_r), extra_col = (out_col_i + 1 < end_c);
            // const bool extra_row = false, extra_col = false;
              #ifdef TIMEDEBUG
                end = clock();
                std::cout << "Time taken to init tile config: " << ((double)(end - start)) / CUSTOM_CLOCKS_PER_SEC << "\n";
              #endif
            #ifdef VERBOSE
              std::cout << "Computing Tile Pair: (" << out_row_i << "," << out_col_i << ")\n";
              std::cout << ("Loading res into AMX registers\n");
            #endif

            #ifdef TIMEDEBUG
              start = clock();
            #endif
              // _tile_zero(0);
              // _tile_zero(1);
              // _tile_zero(2);
              // _tile_zero(3);
              // _tile_loadd (0, res + out_row_i * MAX_OUT_ROWS * out_cols + out_col_i * MAX_OUT_COLS, (out_cols) * 4);
              #ifdef TIMEDEBUG
                end = clock();
                std::cout << "Time taken to load res tile into AMX registers: " << ((double)(end - start)) / CUSTOM_CLOCKS_PER_SEC << "\n";
              #endif

            #ifdef VERBOSE
              std::cout << "Result Before: \n";

              for (int i = 0; i < out_rows; i++) {
                  for (int j = 0; j < out_cols; j++) {
                      std::cout << " " << (float) * (res + i * out_cols + j);
                  }
                  std::cout << "\n";
              }
            #endif
              // Parallelize on inner dimension.
              const int start_id = 0, end_id = (inner_dim + tile_inner_dim - 1) / tile_inner_dim;
              // at::parallel_for(0, (inner_dim + tile_inner_dim - 1) / tile_inner_dim, 0, [&](int start_id, int end_id) {
              for (int inner_dim_i = start_id; inner_dim_i < end_id; inner_dim_i++) {
                  int weight_index_bias, weight_index_bias2;

                #ifdef VERBOSE
                    std::cout << "Computing Tile: (" << out_row_i << "," << out_col_i << ")->" << inner_dim_i << "\n";
                    std::cout << ("Computing Weight\n");
                #endif
                  #ifdef TIMEDEBUG
                    start = clock();
                  #endif
                  // weight.fill_(0); // around 10X slower than using memset.
                  // memset(weight.data_ptr(), 0x00, MAX_OUT_ROWS * MAX_INNER_DIM * 2);

                  // TODO: The implementation now assumes that there are always full tiles. Need to handle cases where the matrix
                  // is smaller than the tile size.

                  // if (weight_metadata[inner_dim_i * tile_inner_dim / 2 + i][out_col_i * tile_out_cols * 2 + j].item<bool>()) {
                  // ((unsigned char *) void_ptr)[index];
                  // torch::Tensor weight = torch::empty(MAX_OUT_ROWS * MAX_INNER_DIM, torch::dtype(torch::kBFloat16));
                    //  int cur_weight_i = 0; // TODO: wrong. will be inaccurate.
                  // std::cout << "Weight Metadata Sizes 1: " << weight_metadata.sizes()[1] << "\n";
                  // #pragma omp parallel for num_threads(2)
                  // parallelize fetching of rows.
                  #ifdef TIMEDEBUG
                    start = clock();
                  #endif
                  TIME_OPERATION("Loading Input 4", {
                    _tile_loadd (4, input.data_ptr() + 1 * (out_row_i * tile_out_rows * inner_dim + inner_dim_i * tile_inner_dim), inner_dim * 1);
                  })
                  // Tried tile_stream_loadd but no noticeable difference in performance.
                  #ifdef TIMEDEBUG
                    end = clock();
                    std::cout << "Time taken to load input tile: " << ((double)(end - start)) / CUSTOM_CLOCKS_PER_SEC << "\n";
                  #endif



                  TIME_OPERATION("Loading Weight 6", {
                    _tile_loadd (6, weight.data_ptr() + 1 * (inner_dim_i * tile_inner_dim / 4 * out_cols * 4 + out_col_i * tile_out_cols * 4), out_cols * 4 * 1 );
                  });

                  // Create a memory array to hold values inside weight, store them and print the contents of the that array
                  char weight_arr[16 * 64];
                  _tile_stored (6, weight_arr, (tile_out_cols) * 4);

                  // std::cout << "Weight Array 6: \n";
                  // for (int i = 0; i < 16; i++) {
                  //   for (int j = 0; j < 64; j++) {
                  //     std::cout << (int)weight_arr[i * 64 + j] << " ";
                  //   }
                  //   std::cout << "\n";
                  // }


                   TIME_OPERATION("Computing 0", {
                    _tile_dpbssd (0, 4, 6);
                    });


                  // std::cout << "Arrived Here" << std::endl;

                  if (extra_col) {                   

                    TIME_OPERATION("Loading Weight 7", {
                      _tile_loadd (7, weight.data_ptr() + 1 * (inner_dim_i * tile_inner_dim / 4 * out_cols * 4 + (out_col_i+1) * tile_out_cols * 4), out_cols * 4 * 1 );
                    });

                    _tile_stored (7, weight_arr, (tile_out_cols) * 4);

                    // std::cout << "Weight Array 7: \n";
                    // for (int i = 0; i < 16; i++) {
                    //   for (int j = 0; j < 64; j++) {
                    //     std::cout << (int)weight_arr[i * 64 + j] << " ";
                    //   }
                    //   std::cout << "\n";
                    // }

                    // _mm_prefetch(input.data_ptr() + 2 * ((out_row_i + 1) * tile_out_rows * inner_dim + inner_dim_i * tile_inner_dim), _MM_HINT_T0);
                    TIME_OPERATION("Computing 1", {
                      _tile_dpbssd (1, 4, 7);
                    });
                  }

                  if (extra_row) {
                    TIME_OPERATION("Loading Input 5", {
                      _tile_loadd (5, input.data_ptr() + 1 * ((out_row_i + 1) * tile_out_rows * inner_dim + inner_dim_i * tile_inner_dim), inner_dim * 1);
                    });

                    TIME_OPERATION("Computing 2", {
                      _tile_dpbssd (2, 5, 6);
                    });
                  }

                  #ifdef TIMEDEBUG
                  start = clock();
                  #endif
                  // Compute dot-product of bytes in tiles 

                  if (extra_col && extra_row)
                  {
                    TIME_OPERATION("Computing 3", {
                      _tile_dpbssd (3, 5, 7);
                    });
                  }

                  #ifdef TIMEDEBUG
                    end = clock();
                    std::cout << "Time taken to perform the computation of the tile: " << ((double)(end - start)) / CUSTOM_CLOCKS_PER_SEC << "\n";
                  #endif
                  

                  #ifdef TIMEDEBUG
                    end = clock();
                    std::cout << "Time taken to load weights into AMX registers: " << ((double)(end - start)) / CUSTOM_CLOCKS_PER_SEC << "\n";
                  #endif
                #ifdef VERBOSE
                  std::cout << ("Performing the computation\n");
                #endif     
              }
              
              // }); // Parallelize on inner dimension.
              #ifdef VERBOSE
                std::cout << ("Storing the result in memory\n");
              #endif

              #ifdef TIMEDEBUG
                start = clock();
              #endif
              // Store the tile data to memory
              _tile_stored (0, (int*)res->get() + out_row_i * tile_out_rows * out_cols + out_col_i * tile_out_cols, (out_cols) * 4);
              if (extra_col)
                _tile_stored (1, (int*)res->get() + out_row_i * tile_out_rows * out_cols + (out_col_i+1) * tile_out_cols, (out_cols) * 4);
              if (extra_row)
                _tile_stored (2, (int*)res->get() + (out_row_i+1) * tile_out_rows * out_cols + out_col_i * tile_out_cols, (out_cols) * 4);
              if (extra_col && extra_row)
                _tile_stored (3, (int*)res->get() + (out_row_i+1) * tile_out_rows * out_cols + (out_col_i+1) * tile_out_cols, (out_cols) * 4);

              #ifdef TIMEDEBUG
                    end = clock();
                    std::cout << "Time taken to store the tile: " << ((double)(end - start)) / CUSTOM_CLOCKS_PER_SEC << "\n";
                  #endif  
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
      // }
        // );
        });
  }
    // }
    // });

   // Release the tile configuration to return to the init state, 
   // which releases all storage it currently holds
   _tile_release ();
    // fused op is marginally faster
    auto t_res = torch::from_blob(res->get(), {input.sizes()[0], input.sizes()[1], out_cols}, torch::kInt32).to(torch::kBFloat16);
    // std::cout << "Performed one forward pass\n";
    return t_res;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &quantized_dense_linear_forward, "quantized dense linear forward",
  pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("bias") = nullptr);
}