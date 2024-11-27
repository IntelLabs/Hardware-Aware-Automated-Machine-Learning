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
#define MAX_OUT_ROWS            16
#define MAX_INNER_DIM           32
#define MAX_OUT_COLS            16
#define NUM_INT_IN_CACHE_LINE   16
// #define VERBOSE
// #define TIMEDEBUG
#define CUSTOM_CLOCKS_PER_SEC 1
#include <time.h> 
constexpr size_t CACHE_LINE_SIZE = 64;
constexpr size_t ELEMENT_SIZE = 2;  // 2 bytes per element


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

  // std::cout << "Out rows: " << out_rows << ", Inner Dim: " << inner_dim << ", Out Cols: " << out_cols << "\n";

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

// intent is to create an alternative kernel for linear layer by using torch C API
// just to demonstrate the capability of torch custom op extension
// we will not implement backward since we focus of inference
// we can also reuse parent implementation if we inherit nn.Linear

// implementation below is adapted from
// https://github.com/pytorch/pytorch/blob/main/torch/csrc/api/include/torch/nn/functional/linear.h

torch::Tensor sparse_linear_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight_metadata,
    const torch::Tensor& weight_values,
    const torch::Tensor& weight_values_is,
    const int            out_cols,
    float*              output_ptr,
    const int            weight_values_start = 0,
    const c10::optional<torch::Tensor>& bias = c10::nullopt) {

    // return  torch::empty({batch_size, num_heads, input.sizes()[2], out_cols}, torch::TensorOptions().dtype(torch::kBFloat16).memory_format(torch::MemoryFormat::Contiguous));

    constexpr size_t MAX_WEIGHT_SIZE = ELEMENT_SIZE * MAX_OUT_COLS * MAX_INNER_DIM; // 2 * 16 * 32 = 1024 bytes

    #ifdef TIMEDEBUG
      clock_t start, end;
    #endif

    #ifdef TIMEDEBUG
      start = clock();
    #endif
  // const int out_rows = input.sizes()[0] * input.sizes()[1];
  // const int inner_dim = input.sizes()[2];
  const int out_rows = input.sizes()[0];
  const int inner_dim = input.sizes()[1];
  // int out_cols = (weight_metadata.sizes()[1] + 1) / 2;
  // std::cout << "Out Cols: " << out_cols << "\n";
  // const int res_size = (out_rows * out_cols);
  // std::unique_ptr<AlignedMemory> res = std::make_unique<AlignedMemory>(res_size * 4, CACHE_LINE_SIZE);

  //  int weight_index;
   
  const int num_inner_threads = weight_values_is.sizes()[0];
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
        continue;
        // weight_index = 0;
        const int start_t = 0, end_t = num_inner_threads;
        at::parallel_for(0, num_inner_threads, 1, [&](int start_t, int end_t) {
          // #pragma omp parallel for num_threads(num_inner_threads) schedule(static) 
          // int cur_weight_i = 0;
          // #pragma omp parallel for num_threads(1)
          for (int t_id = start_t; t_id < end_t; t_id++) {
            continue;

            const int tile_out_rows = std::min(out_rows, MAX_OUT_ROWS);
            const int tile_inner_dim = std::min(inner_dim, MAX_INNER_DIM);
            const int tile_out_cols = std::min(out_cols, MAX_OUT_COLS);

            const bool grouped_col_tiles = ((out_cols > MAX_OUT_COLS) && (out_cols * 2 / num_inner_threads) > (MAX_OUT_COLS * 2));

            const int start_ir = 0, end_ir = tile_inner_dim / 2;
            const int num_col_tiles = (out_cols + MAX_OUT_COLS - 1) / MAX_OUT_COLS;
            const int col_tiles_per_thread = (num_col_tiles + num_inner_threads - 1) / num_inner_threads;
  
            int cur_weight_i = weight_values_is[t_id].item<int>();
            int cur_weight_i_2;

                  
            // std::cout << "from C: i: " << t_id << ", cur_weight_i: " << cur_weight_i << "\n";
            size_t total_size = ELEMENT_SIZE * tile_out_cols * tile_inner_dim;
            // std::unique_ptr<AlignedMemory> weight = std::make_unique<AlignedMemory>(total_size, CACHE_LINE_SIZE);
            alignas(CACHE_LINE_SIZE) uint8_t weight_buffer[MAX_WEIGHT_SIZE];
            alignas(CACHE_LINE_SIZE) uint8_t weight_buffer2[MAX_WEIGHT_SIZE];

            // torch::Tensor weight = torch::empty(
            //     {tile_out_cols * tile_inner_dim},
            //     torch::TensorOptions()
            //         .dtype(torch::kBFloat16)
            //         .memory_format(torch::MemoryFormat::Contiguous)
            //         .aligned(CACHE_LINE_SIZE)
            // );
            const int start_c = t_id * col_tiles_per_thread;
            const int end_c = std::min((t_id + 1) * col_tiles_per_thread, num_col_tiles);
            __tilecfg tile_data = {0};
            init_tile_config (&tile_data, tile_out_rows, tile_inner_dim, tile_out_cols);
            Vec512iUnion metadata[2], popcnt[2];
        // at::parallel_for(0, num_chunks_2, 0, [&](int start_c, int end_c) {
          // std::cout << "Thread ID: " << omp_get_thread_num() << std::endl; 
          for (int out_col_i = start_c; out_col_i < end_c; out_col_i+=2) {
            
             int weight_index_bias;
              if (grouped_col_tiles) {
                // The index equation would be different as each 2 col tiles are placed consecutively.
                weight_index_bias = (tile_inner_dim * out_col_i * ((inner_dim + tile_inner_dim - 1) / tile_inner_dim / 2));
              } else {
                weight_index_bias = (tile_inner_dim / 2 * (out_col_i) * ((inner_dim + tile_inner_dim - 1) / tile_inner_dim));
              }

            #ifdef TIMEDEBUG
              start = clock();
            #endif
            
            const bool extra_row = (out_row_i + 1 < end_r), extra_col = (out_col_i + 1 < end_c);
              #ifdef TIMEDEBUG
                end = clock();
                std::cout << "Time taken to init tile config: " << ((double)(end - start)) / CUSTOM_CLOCKS_PER_SEC << "\n";
              #endif
            #ifdef VERBOSE
              std::cout << "Computing Tile Pair: (" << out_row_i << "," << out_col_i << ")\n";
              std::cout << ("Loading res into AMX registers\n");
            #endif

            // _mm_prefetch ((unsigned int *)weight_metadata.data_ptr() + weight_index_bias, _MM_HINT_T0);
            // _mm_prefetch ((unsigned int *)weight_metadata.data_ptr() + weight_index_bias + tile_inner_dim / 2, _MM_HINT_T0);

            #ifdef TIMEDEBUG
              start = clock();
            #endif
              _tile_zero(0);
              _tile_zero(1);
              _tile_zero(2);
              _tile_zero(3);
              // _tile_loadd (0, res + out_row_i * MAX_OUT_ROWS * out_cols + out_col_i * MAX_OUT_COLS, (out_cols) * 4);
              #ifdef TIMEDEBUG
                end = clock();
                std::cout << "Time taken to load res tile into AMX registers: " << ((double)(end - start)) / CUSTOM_CLOCKS_PER_SEC << "\n";
              #endif

            #ifdef VERBOSE
              std::cout << "Result Before: \n";

              for (int i = 0; i < out_rows; i++) {
                  for (int j = 0; j < out_cols; j++) {
                      std::cout << " " << (float) * (output_ptr + i * out_cols + j);
                  }
                  std::cout << "\n";
              }
            #endif
              // Parallelize on inner dimension.
              const int start_id = 0, end_id = (inner_dim + tile_inner_dim - 1) / tile_inner_dim;
              // at::parallel_for(0, (inner_dim + tile_inner_dim - 1) / tile_inner_dim, 0, [&](int start_id, int end_id) {
              for (int inner_dim_i = start_id; inner_dim_i < end_id; inner_dim_i++) {

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
                    _tile_loadd (4, input.data_ptr() + 2 * (out_row_i * tile_out_rows * inner_dim + inner_dim_i * tile_inner_dim), inner_dim * 2);
                  })

                  if (extra_row) {
                    TIME_OPERATION("Loading Input 5", {
                      _tile_loadd (5, input.data_ptr() + 2 * ((out_row_i + 1) * tile_out_rows * inner_dim + inner_dim_i * tile_inner_dim), inner_dim * 2);
                    });
                  }
                  // _mm_prefetch (input.data_ptr() + 2 * ((out_row_i) * tile_out_rows * inner_dim + (inner_dim_i+1) * tile_inner_dim), _MM_HINT_T0);
                  // _mm_prefetch (input.data_ptr() + 2 * ((out_row_i + 1) * tile_out_rows * inner_dim + (inner_dim_i+1) * tile_inner_dim), _MM_HINT_T0);
                  // Tried tile_stream_loadd but no noticeable difference in performance.
                  #ifdef TIMEDEBUG
                    end = clock();
                    std::cout << "Time taken to load input tile: " << ((double)(end - start)) / CUSTOM_CLOCKS_PER_SEC << "\n";
                  #endif
                  
                  // TIME_OPERATION("Constructing Weight 6", {
                  // _mm_prefetch ((short *)weight_values.data_ptr() + cur_weight_i, _MM_HINT_T2);
                    metadata[0].vec =  _mm512_load_si512((unsigned int *)weight_metadata.data_ptr() + weight_index_bias);
                    metadata[1].vec =  _mm512_load_si512((unsigned int *)weight_metadata.data_ptr() + weight_index_bias + tile_inner_dim / 2);
                    _mm_prefetch ((unsigned int *)weight_metadata.data_ptr() + weight_index_bias + tile_inner_dim, _MM_HINT_T0);
                    _mm_prefetch ((unsigned int *)weight_metadata.data_ptr() + weight_index_bias + tile_inner_dim + tile_inner_dim / 2, _MM_HINT_T0);
                    popcnt[0].vec = parallel_prefix_sum_32(_mm512_popcnt_epi32(metadata[0].vec));
                    popcnt[1].vec = parallel_prefix_sum_32(_mm512_popcnt_epi32(metadata[1].vec));
                    cur_weight_i_2 = cur_weight_i + popcnt[0].arr[end_ir - 1];


  
                    // std::cout << "Arrived after vectorization\n";
                    // Print all values in metadata and popcnt
                    // for (int i = 0; i < 16; i++) {
                    //   std::cout << "Metadata: " << std::bitset<32>(metadata.arr[i]) << ", Popcnt: " << popcnt.arr[i] << "\n";
                    // }
                    // at::parallel_for(start_ir, end_ir, 0, [&](int start_ir, int end_ir) {

                    // Perform the first row computation:
                    _mm512_store_si512(weight_buffer, _mm512_maskz_expandloadu_epi16(metadata[0].arr[start_ir], (short *)weight_values.data_ptr() + weight_values_start + cur_weight_i));


                    #pragma unroll(15)
                    for (register int row = 1; row < 16; row++) {
                      // if (row + 1 < end_ir) {
                      //     _mm_prefetch((__m512i *)(weight->get() + (row + 1) * tile_out_cols * 2 * 2), _MM_HINT_T0);
                      // }

                      _mm512_store_si512(weight_buffer + row * tile_out_cols * 2 * 2, _mm512_maskz_expandloadu_epi16(metadata[0].arr[row], (short *)weight_values.data_ptr() + weight_values_start + cur_weight_i + popcnt[0].arr[row - 1]));

                    }

                    // #pragma unroll(2)
                    // for (int i = 0; i < 2; i++) {
                      // _mm_prefetch ((short *)weight_values.data_ptr() + weight_values_start + cur_weight_i_2 + popcnt[1].arr[15], _MM_HINT_T0);
                    // }

                    // });
                    cur_weight_i = cur_weight_i_2;
                  // });

                  TIME_OPERATION("Loading Weight 6", {
                    _tile_loadd (6, weight_buffer, tile_out_cols * 2 * 2);
                  });

                  TIME_OPERATION("Computing 0", {
                  _tile_dpbf16ps (0, 4, 6);
                  });

                  if (extra_row) {
                    TIME_OPERATION("Computing 2", {
                      _tile_dpbf16ps (2, 5, 6);
                    });
                  }

                  if (extra_col) {
                    // weight_index_bias = (tile_inner_dim / 2 * ((inner_dim_i) + (out_col_i+1) * ((inner_dim + tile_inner_dim - 1) / tile_inner_dim)));

                    // TIME_OPERATION("Constructing Weight 7", {
                      // _mm_prefetch ((short *)weight_values.data_ptr() + cur_weight_i, _MM_HINT_T2);
                      // popcnt.vec = _mm512_popcnt_epi32(metadata.vec);

                    _mm512_store_si512(weight_buffer2, _mm512_maskz_expandloadu_epi16(metadata[1].arr[start_ir], (short *)weight_values.data_ptr() + weight_values_start + cur_weight_i_2));

                      #pragma unroll(15)
                      for (register int row = 1; row < 16; row++) {
                          _mm512_store_si512(weight_buffer2 + row * tile_out_cols * 2 * 2, _mm512_maskz_expandloadu_epi16(metadata[1].arr[row], (short *)weight_values.data_ptr() + weight_values_start + cur_weight_i_2 + popcnt[1].arr[row - 1]));
                      }

                      cur_weight_i+= popcnt[1].arr[end_ir - 1];

                    // });

                    TIME_OPERATION("Loading Weight 7", {
                      _tile_loadd (7, weight_buffer2, tile_out_cols * 2 * 2);
                    });
                    // _mm_prefetch(input.data_ptr() + 2 * ((out_row_i + 1) * tile_out_rows * inner_dim + inner_dim_i * tile_inner_dim), _MM_HINT_T0);
                    TIME_OPERATION("Computing 1", {
                      _tile_dpbf16ps (1, 4, 7);
                    });
                  }
                  
                  #ifdef TIMEDEBUG
                  start = clock();
                  #endif
                  // Compute dot-product of bytes in tiles 

                  if (extra_col && extra_row)
                  {
                    TIME_OPERATION("Computing 3", {
                      _tile_dpbf16ps (3, 5, 7);
                    });
                  }

                  if (grouped_col_tiles) {
                    // The index equation would be different as each 2 col tiles are placed consecutively.
                    weight_index_bias += tile_inner_dim;
                  } else {
                    weight_index_bias += tile_inner_dim / 2;
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
              _tile_stored (0, output_ptr + out_row_i * tile_out_rows * out_cols + out_col_i * tile_out_cols, (out_cols) * 4);
              if (extra_col)
                _tile_stored (1, output_ptr + out_row_i * tile_out_rows * out_cols + (out_col_i+1) * tile_out_cols, (out_cols) * 4);
              if (extra_row)
                _tile_stored (2, output_ptr + (out_row_i+1) * tile_out_rows * out_cols + out_col_i * tile_out_cols, (out_cols) * 4);
              if (extra_col && extra_row)
                _tile_stored (3, output_ptr + (out_row_i+1) * tile_out_rows * out_cols + (out_col_i+1) * tile_out_cols, (out_cols) * 4);

              #ifdef TIMEDEBUG
                    end = clock();
                    std::cout << "Time taken to store the tile: " << ((double)(end - start)) / CUSTOM_CLOCKS_PER_SEC << "\n";
                  #endif  
              #ifdef VERBOSE
                std::cout << "Result After: \n";

                for (int i = 0; i < out_rows; i++) {
                    for (int j = 0; j < out_cols; j++) {
                        std::cout << " " << (float) * (output_ptr + i * out_cols + j);
                    }
                    std::cout << "\n";
                }
              #endif
        }
      // }
        // );
  }
        }); // Number of threads.
    }
    // });

   // Release the tile configuration to return to the init state, 
   // which releases all storage it currently holds
   _tile_release ();
    // fused op is marginally faster
    // return torch::from_blob(res->get(), {input.sizes()[0], input.sizes()[1], out_cols}, torch::kFloat32).to(torch::kBFloat16);;
    return input; // Return dummy tensor since we need to return something
}

torch::Tensor sparse_matmul(
    const torch::Tensor& input,
    const torch::Tensor& weight_metadata,
    const torch::Tensor& weight_values,
    const torch::Tensor& weight_values_bs,
    const torch::Tensor& weight_values_is,
    const int            out_cols,
    const c10::optional<torch::Tensor>& bias = c10::nullopt) {

      if (!set_tiledata_use())
        exit(-1);
      
      const int batch_size = input.sizes()[0];
      const int num_heads = input.sizes()[1];
      // const int out_rows = input.sizes()[2];
      // const int inner_dim = input.sizes()[3];

      //  int weight_index;
      
      // const int num_inner_threads = weight_values_is.sizes()[2];
      // const int num_chunks = (out_rows + MAX_OUT_ROWS - 1) / MAX_OUT_ROWS;
      // const int start_r = 0, end_r = num_chunks;
      auto out = torch::empty({batch_size, num_heads, input.sizes()[2], out_cols}, torch::TensorOptions().dtype(torch::kFloat32).memory_format(torch::MemoryFormat::Contiguous));
      // std::unique_ptr<AlignedMemory> out = std::make_unique<AlignedMemory>(batch_size * num_heads * input.sizes()[2] * out_cols * 4, CACHE_LINE_SIZE);

      // Get pointer to output tensor's data
      float* out_ptr = static_cast<float*>(out.data_ptr());

      for (int batch_i = 0; batch_i < batch_size; batch_i++) {
        const int start_h = 0, end_h = 32;
        // at::parallel_for(0, 32, 0, [&](int start_h, int end_h) {
        // #pragma omp parallel for schedule(static, 16) 
        for (int head_i = start_h; head_i < end_h; head_i++) {
          // Calculate offset for current batch and head
          float* current_out_ptr = out_ptr + 
              (batch_i * num_heads * input.sizes()[2] * out_cols + 
               head_i * input.sizes()[2] * out_cols);
          
          sparse_linear_forward(
              input[batch_i][head_i], 
              weight_metadata[batch_i][head_i], 
              weight_values, 
              weight_values_is[batch_i][head_i], 
              out_cols,
              current_out_ptr,  // Pass pointer to output location
              weight_values_bs[batch_i][head_i].item<int>(), 
              bias);
        }
        // }); // Head parallelization.
      }

      return out.to(torch::kBFloat16);
      // return torch::from_blob(out->get(), {batch_size, num_heads, input.sizes()[2], out_cols}).to(torch::kBFloat16);
    }


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &sparse_linear_forward, "sparse linear forward",
  pybind11::arg("input"), pybind11::arg("weight_metadata"), pybind11::arg("weight_values"), pybind11::arg("weight_values_is"), pybind11::arg("out_cols"), pybind11::arg("output_ptr"), pybind11::arg("weight_values_start") = 0, pybind11::arg("bias") = nullptr);

  m.def("matmul", &sparse_matmul, "sparse matmul",
  pybind11::arg("input"), pybind11::arg("weight_metadata"), pybind11::arg("weight_values"), pybind11::arg("weight_values_bs") ,pybind11::arg("weight_values_is"), pybind11::arg("out_cols"), pybind11::arg("bias") = nullptr);
}