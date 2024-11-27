// #include <torch/extension.h>
// #include <torch/nn/functional/linear.h>
// #include <optional>
// #include <torch/torch.h>
// #include <cstdint>
// #include <iostream>

// #include <immintrin.h>
// #include <stdlib.h>
// #include <stdint.h>
// #include <stdio.h>
// #include <sys/syscall.h>
// #include <unistd.h>
// #include <stdbool.h>
// #include <thread>
// #define _OPENMP
// #include <ATen/ParallelOpenMP.h>

// #include "example_utils.hpp"
// #include "oneapi/dnnl/dnnl.hpp"

// using namespace dnnl;

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   // m.def("onednn_linear", &onednn_linear_get_primitive, "onednn linear get primitive",
//   // pybind11::arg("N"), pybind11::arg("IC"), pybind11::arg("OC"));

//   m.def("onednn_linear", &onednn_linear_forward, "onednn linear forward",
//   pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("bias") = nullptr);
// }



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

#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"
// #define TIMEDEBUG
#define CUSTOM_CLOCKS_PER_SEC 1
#include <time.h> 



using namespace dnnl;

class LayerContainer {
  public:
    engine eng;
    // inner_product_forward::primitive_desc inner_product_pd;
    inner_product_forward inner_product;
    memory::desc src_md, weights_md, dst_md;

};

void * onednn_linear_get_primitive(
  memory::dim N, memory::dim IC, memory::dim OC
) {
    // Create execution engine (CPU in this case)
    dnnl::engine engine(dnnl::engine::kind::cpu, 0);

    // Prepare memory descriptors
    memory::dims src_dims = {N, IC};
    memory::dims weights_dims = {OC, IC};
    memory::dims dst_dims = {N, OC};

    auto src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nc);
    auto weights_md = memory::desc(weights_dims, memory::data_type::f32, memory::format_tag::oi);
    auto dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::nc);

    // Create inner product primitive descriptor
    auto inner_product_pd = inner_product_forward::primitive_desc(
        engine,
        prop_kind::forward_inference,
        src_md,
        weights_md,
        dst_md
    );
    #ifdef TIMEDEBUG
      clock_t start, end;
    #endif

    #ifdef TIMEDEBUG
      start = clock();
    #endif
    // Create the inner product primitive
    auto inner_product = inner_product_forward(inner_product_pd);
    #ifdef TIMEDEBUG
      end = clock();
      std::cout << "Time taken to create the primitive: " << ((double)(end - start)) / CUSTOM_CLOCKS_PER_SEC << "\n";
    #endif

    auto container = new LayerContainer();
    container->eng = engine;
    container->inner_product = inner_product;
    container->src_md = src_md;
    container->weights_md = weights_md;
    container->dst_md = dst_md;

    return container;
}

// Helper function to create MatMul primitive descriptor
torch::Tensor onednn_linear_forward(
    void* container_v, 
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias_opt) {

    #ifdef TIMEDEBUG
      clock_t start, end;
    #endif

    #ifdef TIMEDEBUG
      start = clock();
    #endif
    auto container = * ((LayerContainer *) container_v);
    #ifdef TIMEDEBUG
      end = clock();
      std::cout << "Time taken to parse container: " << ((double)(end - start)) / CUSTOM_CLOCKS_PER_SEC << "\n";
    #endif
    auto inner_product = container.inner_product;

    #ifdef TIMEDEBUG
      start = clock();
    #endif
    dnnl::engine engine = container.eng;
    #ifdef TIMEDEBUG
      end = clock();
      std::cout << "Time taken to get engine: " << ((double)(end - start)) / CUSTOM_CLOCKS_PER_SEC << "\n";
    #endif

    #ifdef TIMEDEBUG
      start = clock();
    #endif
    dnnl::stream stream(engine);
    #ifdef TIMEDEBUG
      end = clock();
      std::cout << "Time taken to create stream: " << ((double)(end - start)) / CUSTOM_CLOCKS_PER_SEC << "\n";
    #endif

    #ifdef TIMEDEBUG
      start = clock();
    #endif
    // Get tensor dimensions
    auto input_sizes = input.sizes();
    auto weight_sizes = weight.sizes();

    memory::dim N = input_sizes[0];  // Input Rows
    memory::dim IC = input_sizes[1]; // Hidden Dimension
    memory::dim OC = weight_sizes[0]; // Output Cols

    // for (int i = 0; i < OC; i++) {
    //   for (int j = 0; j < IC; j++) {
    //     std::cout << "(" << i << "," << j << "): " << weight[i][j] << "\n";
    //   }
    // }

    // Prepare memory descriptors
    memory::dims src_dims = {N, IC};
    memory::dims weights_dims = {OC, IC};
    memory::dims dst_dims = {N, OC};

    auto src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nc);
    auto weights_md = memory::desc(weights_dims, memory::data_type::f32, memory::format_tag::oi);
    auto dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::nc);
    #ifdef TIMEDEBUG
      end = clock();
      std::cout << "Time taken to create memory descriptors except output: " << ((double)(end - start)) / CUSTOM_CLOCKS_PER_SEC << "\n";
    #endif

    #ifdef TIMEDEBUG
      start = clock();
    #endif
    // Prepare memory objects
    auto src_mem = memory(src_md, engine, input.data_ptr());
    auto weights_mem = memory(weights_md, engine, weight.data_ptr());
    #ifdef TIMEDEBUG
      end = clock();
      std::cout << "Time taken to prepare memory objects of source and weight: " << ((double)(end - start)) / CUSTOM_CLOCKS_PER_SEC << "\n";
    #endif

    #ifdef TIMEDEBUG
      start = clock();
    #endif
    // Create output tensor
    auto output = torch::empty({N, OC}, input.options());
    #ifdef TIMEDEBUG
      end = clock();
      std::cout << "Time taken to create output tensor: " << ((double)(end - start)) / CUSTOM_CLOCKS_PER_SEC << "\n";
    #endif
    #ifdef TIMEDEBUG
      start = clock();
    #endif
    auto dst_mem = memory(dst_md, engine, output.data_ptr());
    #ifdef TIMEDEBUG
      end = clock();
      std::cout << "Time taken to create dst_mem: " << ((double)(end - start)) / CUSTOM_CLOCKS_PER_SEC << "\n";
    #endif
    // Create the inner product primitive
    // auto inner_product = inner_product_forward(inner_product_pd);


    // Add arguments
    #ifdef TIMEDEBUG
      start = clock();
    #endif
    std::unordered_map<int, memory> args;
    args.insert({DNNL_ARG_SRC, src_mem});
    args.insert({DNNL_ARG_WEIGHTS, weights_mem});
    args.insert({DNNL_ARG_DST, dst_mem});
    #ifdef TIMEDEBUG
      end = clock();
      std::cout << "Time taken to insert arguments: " << ((double)(end - start)) / CUSTOM_CLOCKS_PER_SEC << "\n";
    #endif

    #ifdef TIMEDEBUG
      start = clock();
    #endif
    // Execute the primitive
    inner_product.execute(stream, args);
    #ifdef TIMEDEBUG
      end = clock();
      std::cout << "Time taken to execute the primitive: " << ((double)(end - start)) / CUSTOM_CLOCKS_PER_SEC << "\n";
    #endif
    // Wait for execution to complete
    // stream.wait();

    // Return the computed output
    return output;
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &onednn_linear_forward, "onednn linear forward",
  pybind11::arg("inner_product"), pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("bias") = nullptr);

  m.def("get_onednn_descriptor", &onednn_linear_get_primitive, "onednn linear get primitive",
  pybind11::arg("N"), pybind11::arg("IC"), pybind11::arg("OC"));
}