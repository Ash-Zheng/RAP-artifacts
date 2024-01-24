#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
// #include <cudf/column/column.hpp>

void init_cuda(
    int32_t device_id
); 

void sigrid_hash_main(
    torch::Tensor tensor,
    int64_t salt,
    int64_t maxValue
);

void sigrid_hash_main_64(
    torch::Tensor tensor,
    int64_t salt,
    int64_t maxValue
);

void sigrid_hash_main_fused(
    torch::Tensor tensor,
    int64_t salt,
    int64_t maxValue
);

torch::Tensor bucketize_main(
    torch::Tensor tensor,
    torch::Tensor borders
    // const std::vector<int>&  borders
);


torch::Tensor fill_null_float(
    int64_t data_ptr,
    int32_t length
);

std::vector<torch::Tensor> fill_null_float_fused(
    torch::Tensor data_ptrs,
    int32_t batch_size
);

void fill_null_float_inplace(
    torch::Tensor output_tensor,
    int64_t data_ptr,
    int32_t length
);

torch::Tensor fill_null_int64(
    int64_t data_ptr,
    int32_t length
);

void fill_null_int64_inplace(
    torch::Tensor output_tensor,
    int64_t data_ptr,
    int32_t length
);

std::vector<torch::Tensor> fill_null_int64_fused(
    torch::Tensor data_ptrs,
    int32_t batch_size
);



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sigrid_hash", &sigrid_hash_main, "sigrid_hash_main()");
  m.def("sigrid_hash_fused", &sigrid_hash_main_fused, "sigrid_hash_main_fused()");
  m.def("bucketize", &bucketize_main, "bucketize_main()");
  m.def("init_cuda", &init_cuda, "init_cuda()");

  m.def("fill_null_float", &fill_null_float, "fill_null_float()");
  m.def("fill_null_float_inplace", &fill_null_float_inplace, "fill_null_float_inplace()");
  m.def("fill_null_float_fused", &fill_null_float_fused, "fill_null_float_fused()");

  m.def("fill_null_int64", &fill_null_int64, "fill_null_int64()");
  m.def("fill_null_int64_inplace", &fill_null_int64_inplace, "fill_null_int64_inplace()");
  m.def("fill_null_int64_fused", &fill_null_int64_fused, "fill_null_int64_fused()");
}