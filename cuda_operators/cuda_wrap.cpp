#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
// #include <cudf/column/column.hpp>

void init_cuda(
    int32_t device_id
); 

int64_t copy_vector_to_GPU_uint64(const std::vector<uint64_t>& input);

int64_t copy_vector_to_GPU_int64(const std::vector<int64_t>& input);

int64_t copy_vector_to_GPU_int32(const std::vector<int32_t>& input);

int64_t get_shape_list(const std::vector<torch::Tensor> tensor_list, int dim);

int64_t copy_tensor_list_to_GPU(
    std::vector<torch::Tensor> tensor_list
);

torch::Tensor copy_tensor_list_to_GPU_tensor(
    std::vector<torch::Tensor> tensor_list
);

// ==================== Fill_Null ====================
torch::Tensor fill_null_float(
    int64_t data_ptr,
    int32_t length
);
std::vector<torch::Tensor> fill_null_float_fused(
    torch::Tensor data_ptrs,
    int32_t batch_size,
    int if_kernel=1
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
    int32_t batch_size,
    int if_kernel=1
);

std::vector<torch::Tensor> fill_null_float_fused_inplace(
    torch::Tensor data_ptrs,
    torch::Tensor output_tensor,
    int32_t batch_size
);
std::vector<torch::Tensor> fill_null_int64_fused_inplace(
    torch::Tensor data_ptrs,
    torch::Tensor output_tensor,
    int32_t batch_size
);
void fill_null_int64_list_gpu_part(
    torch::Tensor data_ptrs,
    int64_t out_tensor_list_ptr,
    int32_t batch_size
);
void fill_null_float_list_gpu_part(
    torch::Tensor data_ptrs,
    int64_t out_tensor_list_ptr,
    int32_t batch_size
);
void fill_null_float_list_gpu_part_tensor(
    torch::Tensor data_ptrs,
    torch::Tensor out_tensor_list_ptr_tensor,
    int32_t batch_size
);

void fill_null_int64_list_gpu_part_tensor(
    torch::Tensor data_ptrs,
    torch::Tensor out_tensor_list_ptr_tensor,
    int32_t batch_size
);

// ==================== Sigrid Hash ====================
void sigrid_hash_main(
    torch::Tensor tensor,
    int64_t salt,
    int64_t maxValue
);
void sigrid_hash_kernel(
    torch::Tensor tensor,
    int64_t salt,
    int64_t maxValue,
    uint64_t multiplier_,
    int shift_
);
std::pair<uint64_t, int> sigrid_hash_data_preprare(
    int64_t maxValue
);
void sigrid_hash_main_fused(
    torch::Tensor tensor,
    int64_t salt,
    int64_t maxValue
);
std::pair<std::vector<uint64_t>, std::vector<int32_t>> sigrid_hash_list_compute_shift(
    std::vector<uint64_t> maxValue_list
);

std::tuple<int64_t, int64_t, int64_t> sigridhash_ptr_prepare(
    std::vector<int64_t> maxValue_list
);

std::tuple<int64_t, int64_t, int64_t> sigrid_hash_list_compute_shift_fused(
    std::vector<int64_t> maxValue_list
);

void sigrid_hash_list_fused(
    std::vector<torch::Tensor> tensor_list,
    int64_t salt,
    int64_t maxValue_list,
    int64_t multiplier_list,
    int64_t shift_list,
    int if_kernel=1
);


std::tuple<int64_t, int64_t, int> sigridhash_cpu_part(
    std::vector<torch::Tensor> tensor_list
);
std::tuple<int64_t, int64_t, int> sigridhash_cpu_part_base(
    std::vector<torch::Tensor> tensor_list
);
void sigridhash_gpu_part(
    int64_t d_tensor_ptr_int64,
    int64_t d_offset_ptr_int64,
    int64_t salt,
    int64_t maxValue_list,
    int64_t multiplier_list,
    int64_t shift_list,
    int total_length,
    int batch_size
);
// void sigrid_hash_list_fused_CUDA_Graph(
//     std::vector<torch::Tensor> tensor_list,
//     int64_t salt,
//     int64_t maxValue_list,
//     int64_t multiplier_list,
//     int64_t shift_list,
//     int batch_size,
//     int32_t nBlock
// );


// ==================== Bucketize ====================
torch::Tensor bucketize_main(
    torch::Tensor tensor,
    torch::Tensor borders
);
std::vector<torch::Tensor> bucketize_list_fused(
    std::vector<torch::Tensor> tensor_list,
    std::vector<torch::Tensor> borders_list
);
std::tuple<int64_t, int64_t> bucketize_ptr_prepare(
    std::vector<torch::Tensor> tensor_list,
    std::vector<torch::Tensor> borders_list
);
std::vector<torch::Tensor> bucketize_list_fused_prepared(
    std::vector<torch::Tensor> tensor_list,
    std::vector<torch::Tensor> borders_list,
    int64_t borders_list_ptr,
    int64_t length_list_ptr,
    int if_kernel=1
);
std::tuple<int64_t, int64_t, int32_t, std::vector<torch::Tensor>> bucketize_cpu_part(
    std::vector<torch::Tensor> tensor_list,
    std::vector<torch::Tensor> borders_list
);
void bucketize_gpu_part(
    int64_t tensor_list_ptr,
    int64_t tensor_out_list_ptr,
    int64_t borders_list_ptr,
    int64_t length_list_ptr,
    int32_t total_length,
    int32_t batch_size
);
std::tuple<int64_t, int64_t, int32_t, std::vector<torch::Tensor>> bucketize_cpu_part_base(
    std::vector<torch::Tensor> tensor_list,
    std::vector<torch::Tensor> output_tensors,
    std::vector<torch::Tensor> borders_list
);


// ==================== Logit ====================
void logit_main(
    torch::Tensor input_tensor,
    float eps
);
void logit_list_fused(
    std::vector<torch::Tensor> tensor_list,
    std::vector<float> eps
);
int64_t logit_ptr_prepare(
    std::vector<torch::Tensor> tensor_list,
    std::vector<float> eps
);
void logit_list_fused_prepared(
    std::vector<torch::Tensor> tensor_list,
    int64_t eps_list_ptr,
    int if_kernel=1
);
std::pair<int64_t, int32_t> logit_cpu_part(
    std::vector<torch::Tensor> tensor_list
);
std::pair<int64_t, int32_t> logit_cpu_part_base(
    std::vector<torch::Tensor> tensor_list
);
void logit_gpu_part(
    int64_t d_tensor_ptr_int64,
    int64_t eps_list_ptr,
    int32_t total_length,
    int32_t batch_size
);


// ==================== FirstX ====================
torch::Tensor firstx_main(
    torch::Tensor input_tensor, 
    int x
);
std::vector<torch::Tensor> firstx_list_fused(
    std::vector<torch::Tensor> input_tensor_list, 
    std::vector<int> x_list
);
std::tuple<torch::Tensor, torch::Tensor>  firstx_ptr_prepare(
    std::vector<torch::Tensor> input_tensor_list, 
    std::vector<int> x_list
);
std::vector<torch::Tensor> firstx_list_fused_prepared(
    std::vector<torch::Tensor> input_tensor_list, 
    std::vector<int> x_list, 
    // int64_t x_list_ptr, 
    // int64_t width_list_ptr, 
    torch::Tensor x_list_ptr,
    torch::Tensor width_list_ptr,
    int if_kernel=1
);
std::tuple<int64_t, int64_t, int32_t, std::vector<torch::Tensor>> firstx_cpu_part(
    std::vector<torch::Tensor> input_tensor_list, 
    std::vector<int> x_list
);
void firstx_gpu_part(
    int64_t d_tensor_in_ptr_int64,
    int64_t d_tensor_out_ptr_int64,
    torch::Tensor x_list_ptr,
    torch::Tensor width_list_ptr,
    int32_t length,
    int32_t batch_size,
    int if_float
);
std::tuple<int64_t, int64_t, int32_t, std::vector<torch::Tensor>> firstx_cpu_part_base(
    std::vector<torch::Tensor> input_tensor_list, 
    std::vector<torch::Tensor> output_tensors, 
    std::vector<int> x_list
);



// ==================== BoxCox ====================
void boxcox_main(
    torch::Tensor input_tensor,
    float lambda
);
void boxcox_list_fused(
    std::vector<torch::Tensor> tensor_list,
    std::vector<float> lambda_list
);
int64_t boxcox_ptr_prepare(
    std::vector<torch::Tensor> tensor_list,
    std::vector<float> lambda_list
);
void boxcox_list_fused_prepared(
    std::vector<torch::Tensor> tensor_list,
    int64_t lambda_list_ptr,
    int if_kernel=1
);
std::pair<int64_t, int32_t>boxcox_cpu_part(
    std::vector<torch::Tensor> tensor_list
);
void boxcox_gpu_part(
    int64_t tensor_list_ptr,
    int64_t lambda_list_ptr,
    int32_t total_length,
    int32_t batch_size
);
std::pair<int64_t, int32_t>boxcox_cpu_part_base(
    std::vector<torch::Tensor> tensor_list
);


// ==================== Clamp ====================
void clamp_main(
    torch::Tensor input_tensor,
    int64_t low,
    int64_t high
);
void clamp_list_fused(
    std::vector<torch::Tensor> tensor_list,
    std::vector<int64_t> low_list,
    std::vector<int64_t> high_list
);
std::tuple<int64_t, int64_t> clamp_ptr_prepare(
    std::vector<torch::Tensor> tensor_list,
    std::vector<int64_t> low_list,
    std::vector<int64_t> high_list
);
void clamp_list_fused_prepared(
    std::vector<torch::Tensor> tensor_list,
    int64_t low_list_ptr,
    int64_t high_list_ptr,
    int if_kernel=1
);
std::pair<int64_t, int32_t> clamp_cpu_part(
    std::vector<torch::Tensor> tensor_list
);
void clamp_gpu_part(
    int64_t tensor_list_ptr,
    int64_t low_list_ptr,
    int64_t high_list_ptr,
    int32_t total_length,
    int32_t batch_size
);
std::pair<int64_t, int32_t> clamp_cpu_part_base(
    std::vector<torch::Tensor> tensor_list
);


// ==================== OneHot ====================
torch::Tensor onehot_main(
    torch::Tensor tensor,
    float low,
    float high,
    int num_classes
);
std::vector<torch::Tensor> onehot_list_fused(
    std::vector<torch::Tensor> tensor_list,
    std::vector<float> low_list,
    std::vector<float> high_list,
    std::vector<int> num_classes_list
);
std::tuple<int64_t, int64_t, int64_t> onehot_ptr_prepare(
    std::vector<torch::Tensor> tensor_list,
    std::vector<float> low_list,
    std::vector<float> high_list,
    std::vector<int> num_classes_list
);
std::vector<torch::Tensor> onehot_list_fused_prepared(
    std::vector<torch::Tensor> tensor_list,
    std::vector<int> num_classes_list,
    int64_t low_list_ptr,
    int64_t high_list_ptr,
    int64_t num_classes_list_ptr,
    int if_kernel=1
);
std::tuple<int64_t, int64_t, int32_t, std::vector<torch::Tensor>> onehot_cpu_part(
    std::vector<torch::Tensor> tensor_list,
    std::vector<int> num_classes_list
);
void onehot_gpu_part(
    int64_t tensor_list_ptr,
    int64_t tensor_out_list_ptr,
    int64_t low_list_ptr,
    int64_t high_list_ptr,
    int64_t num_classes_list_ptr,
    int length,
    int batch_size
);
std::tuple<int64_t, int64_t, int32_t, std::vector<torch::Tensor>> onehot_cpu_part_base(
    std::vector<torch::Tensor> tensor_list,
    std::vector<torch::Tensor> output_tensors,
    std::vector<int> num_classes_list
);


// ==================== N-Gram ====================
torch::Tensor ngram_main(
    torch::Tensor tensor,
    int n_of_gram
);
void ngram_main_inplace(
    torch::Tensor tensor,
    torch::Tensor output_tensor,
    int n_of_gram
);
std::vector<torch::Tensor> ngram_list_fused(
    std::vector<torch::Tensor> tensor_list,
    std::vector<int> n_of_gram_list
);
std::tuple<int64_t, int64_t> ngram_ptr_prepare(
    std::vector<torch::Tensor> tensor_list,
    std::vector<int> n_of_gram_list
);
std::vector<torch::Tensor> ngram_list_fused_prepared(
    std::vector<torch::Tensor> tensor_list,
    std::vector<int> n_of_gram_list,
    int64_t width_list_ptr,
    int64_t n_of_gram_list_ptr,
    int if_kernel=1
);
std::tuple<int64_t, int64_t, int32_t, std::vector<torch::Tensor>> ngram_cpu_part(
    std::vector<torch::Tensor> tensor_list,
    std::vector<int> n_of_gram_list
);
std::tuple<int64_t, int32_t> mapid_cpu_part(
    std::vector<torch::Tensor> tensor_list
);
void ngram_gpu_part(
    int64_t d_tensor_in_ptr_int64,
    int64_t d_tensor_out_ptr_int64,
    int64_t width_list_ptr,
    int64_t n_of_gram_list_ptr,
    int32_t length,
    int32_t batch_size
);
std::tuple<int64_t, int64_t, int32_t, std::vector<torch::Tensor>> ngram_cpu_part_base(
    std::vector<torch::Tensor> tensor_list,
    std::vector<torch::Tensor> output_tensors,
    std::vector<int> n_of_gram_list
);

// ==================== MapID ====================
void mapid_main(
    torch::Tensor tensor,
    torch::Tensor mapping
);
void mapid_list_fused(
    std::vector<torch::Tensor> tensor_list,
    std::vector<torch::Tensor> mapping_list
);
void mapid_list_fused_prepared(
    std::vector<torch::Tensor> tensor_list,
    int64_t mapping_list_ptr,
    int64_t width_list,
    int64_t mapping_width_list,
    int if_kernel=1
);
std::tuple<int64_t, int64_t, int64_t> mapid_ptr_prepare(
    std::vector<torch::Tensor> tensor_list,
    std::vector<torch::Tensor> mapping_list
);
std::tuple<int64_t, int32_t> mapid_cpu_part(
    std::vector<torch::Tensor> tensor_list
);
void mapid_gpu_part(
    int64_t d_tensor_in_ptr_int64,
    int64_t mapping_list_ptr,
    int64_t width_list,
    int64_t mapping_width_list,
    int length,
    int batch_size
);
std::tuple<int64_t, int32_t> mapid_cpu_part_base(
    std::vector<torch::Tensor> tensor_list
);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init_cuda", &init_cuda, "init_cuda()");

  
  m.def("copy_tensor_list_to_GPU", &copy_tensor_list_to_GPU, "copy_tensor_list_to_GPU()");
  m.def("copy_vector_to_GPU_uint64", &copy_vector_to_GPU_uint64, "copy_vector_to_GPU_uint64()");
  m.def("copy_vector_to_GPU_int64", &copy_vector_to_GPU_int64, "copy_vector_to_GPU_int64()");
  m.def("copy_vector_to_GPU_int32", &copy_vector_to_GPU_int32, "copy_vector_to_GPU_int32()");
  m.def("get_shape_list", &get_shape_list, "get_shape_list()");
  m.def("copy_tensor_list_to_GPU_tensor", &copy_tensor_list_to_GPU_tensor, "copy_tensor_list_to_GPU_tensor()");

  m.def("fill_null_float", &fill_null_float, "fill_null_float()");
  m.def("fill_null_float_inplace", &fill_null_float_inplace, "fill_null_float_inplace()");
  m.def("fill_null_float_fused", &fill_null_float_fused, "fill_null_float_fused()", py::arg("data_ptrs"), py::arg("batch_size"), py::arg("if_kernel")=1);
  m.def("fill_null_float_fused_inplace", &fill_null_float_fused_inplace, "fill_null_float_fused_inplace()");
  m.def("fill_null_float_list_gpu_part", &fill_null_float_list_gpu_part, "fill_null_float_list_gpu_part()");
  m.def("fill_null_float_list_gpu_part_tensor", &fill_null_float_list_gpu_part_tensor, "fill_null_float_list_gpu_part_tensor()");
  
  m.def("fill_null_int64", &fill_null_int64, "fill_null_int64()");
  m.def("fill_null_int64_inplace", &fill_null_int64_inplace, "fill_null_int64_inplace()");
  m.def("fill_null_int64_fused", &fill_null_int64_fused, "fill_null_int64_fused()", py::arg("data_ptrs"), py::arg("batch_size"), py::arg("if_kernel")=1);
  m.def("fill_null_int64_fused_inplace", &fill_null_int64_fused_inplace, "fill_null_int64_fused_inplace()");
  m.def("fill_null_int64_list_gpu_part", &fill_null_int64_list_gpu_part, "fill_null_int64_list_gpu_part()");
  m.def("fill_null_int64_list_gpu_part_tensor", &fill_null_int64_list_gpu_part_tensor, "fill_null_int64_list_gpu_part_tensor()");

  m.def("sigrid_hash", &sigrid_hash_main, "sigrid_hash_main()");
  m.def("sigrid_hash_kernel", &sigrid_hash_kernel, "sigrid_hash_kernel()");
  m.def("sigrid_hash_data_preprare", &sigrid_hash_data_preprare, "sigrid_hash_data_preprare()");
  m.def("sigridhash_ptr_prepare", &sigridhash_ptr_prepare, "sigridhash_ptr_prepare()");
  m.def("sigrid_hash_list_fused", &sigrid_hash_list_fused, "sigrid_hash_list_fused()", py::arg("tensor_list"), py::arg("salt"), py::arg("maxValue_list"), py::arg("multiplier_list"), py::arg("shift_list"), py::arg("if_kernel")=1);
  m.def("sigrid_hash_list_compute_shift", &sigrid_hash_list_compute_shift, "sigrid_hash_list_compute_shift()");
  m.def("sigrid_hash_list_compute_shift_fused", &sigrid_hash_list_compute_shift_fused, "sigrid_hash_list_compute_shift_fused()");
  m.def("sigridhash_cpu_part", &sigridhash_cpu_part, "sigridhash_cpu_part()");
  m.def("sigridhash_cpu_part_base", &sigridhash_cpu_part_base, "sigridhash_cpu_part_base()");
  m.def("sigridhash_gpu_part", &sigridhash_gpu_part, "sigridhash_gpu_part()");


  m.def("bucketize", &bucketize_main, "bucketize_main()");
  m.def("bucketize_list_fused", &bucketize_list_fused, "bucketize_list_fused()");
  m.def("bucketize_ptr_prepare", &bucketize_ptr_prepare, "bucketize_ptr_prepare()");
  m.def("bucketize_list_fused_prepared", &bucketize_list_fused_prepared, "bucketize_list_fused_prepared()", py::arg("tensor_list"), py::arg("borders_list"), py::arg("borders_list_ptr"), py::arg("length_list_ptr"), py::arg("if_kernel")=1);
  m.def("bucketize_cpu_part", &bucketize_cpu_part, "bucketize_cpu_part()");
  m.def("bucketize_cpu_part_base", &bucketize_cpu_part_base, "bucketize_cpu_part_base()");
  m.def("bucketize_gpu_part", &bucketize_gpu_part, "bucketize_gpu_part()");

  m.def("logit", &logit_main, "logit()");
  m.def("logit_list_fused", &logit_list_fused, "logit_list_fused()");
  m.def("logit_ptr_prepare", &logit_ptr_prepare, "logit_ptr_prepare()");
  m.def("logit_list_fused_prepared", &logit_list_fused_prepared, "logit_list_fused_prepared()", py::arg("tensor_list"), py::arg("eps_list_ptr"), py::arg("if_kernel")=1);
  m.def("logit_cpu_part", &logit_cpu_part, "logit_cpu_part()");
  m.def("logit_cpu_part_base", &logit_cpu_part_base, "logit_cpu_part_base()");
  m.def("logit_gpu_part", &logit_gpu_part, "logit_gpu_part()");


  m.def("firstx", &firstx_main, "firstx_main()");
  m.def("firstx_list_fused", &firstx_list_fused, "firstx_list_fused()");
  m.def("firstx_ptr_prepare", &firstx_ptr_prepare, "firstx_ptr_prepare()");
  m.def("firstx_list_fused_prepared", &firstx_list_fused_prepared, "firstx_list_fused_prepared()", py::arg("input_tensor_list"), py::arg("x_list"), py::arg("x_list_ptr"), py::arg("width_list_ptr"), py::arg("if_kernel")=1);
  m.def("firstx_cpu_part", &firstx_cpu_part, "firstx_cpu_part()");
  m.def("firstx_cpu_part_base", &firstx_cpu_part_base, "firstx_cpu_part_base()");
  m.def("firstx_gpu_part", &firstx_gpu_part, "firstx_gpu_part()");

  m.def("boxcox", &boxcox_main, "boxcox_main()");
  m.def("boxcox_list_fused", &boxcox_list_fused, "boxcox_list_fused()");
  m.def("boxcox_ptr_prepare", &boxcox_ptr_prepare, "boxcox_ptr_prepare()");
  m.def("boxcox_list_fused_prepared", &boxcox_list_fused_prepared, "boxcox_list_fused_prepared()", py::arg("tensor_list"), py::arg("lambda_list_ptr"), py::arg("if_kernel")=1);
  m.def("boxcox_cpu_part_base", &boxcox_cpu_part_base, "boxcox_cpu_part_base()");
  m.def("boxcox_cpu_part", &boxcox_cpu_part, "boxcox_cpu_part()");
  m.def("boxcox_gpu_part", &boxcox_gpu_part, "boxcox_gpu_part()");


  m.def("clamp", &clamp_main, "clamp_main()");
  m.def("clamp_list_fused", &clamp_list_fused, "clamp_list_fused()");
  m.def("clamp_ptr_prepare", &clamp_ptr_prepare, "clamp_ptr_prepare()");
  m.def("clamp_list_fused_prepared", &clamp_list_fused_prepared, "clamp_list_fused_prepared()", py::arg("tensor_list"), py::arg("low_list_ptr"), py::arg("high_list_ptr"), py::arg("if_kernel")=1);
  m.def("clamp_cpu_part_base", &clamp_cpu_part_base, "clamp_cpu_part_base()");
  m.def("clamp_cpu_part", &clamp_cpu_part, "clamp_cpu_part()");
  m.def("clamp_gpu_part", &clamp_gpu_part, "clamp_gpu_part()");


  m.def("onehot", &onehot_main, "onehot_main()");
  m.def("onehot_list_fused", &onehot_list_fused, "onehot_list_fused()");
  m.def("onehot_ptr_prepare", &onehot_ptr_prepare, "onehot_ptr_prepare()");
  m.def("onehot_list_fused_prepared", &onehot_list_fused_prepared, "onehot_list_fused_prepared()", py::arg("tensor_list"), py::arg("num_classes_list"), py::arg("low_list_ptr"), py::arg("high_list_ptr"), py::arg("num_classes_list_ptr"), py::arg("if_kernel")=1);
  m.def("onehot_cpu_part_base", &onehot_cpu_part_base, "onehot_cpu_part_base()");
  m.def("onehot_cpu_part", &onehot_cpu_part, "onehot_cpu_part()");
  m.def("onehot_gpu_part", &onehot_gpu_part, "onehot_gpu_part()");


  m.def("ngram", &ngram_main, "ngram_main()");
  m.def("ngram_inplace", &ngram_main_inplace, "ngram_main_inplace()");
  m.def("ngram_list_fused", &ngram_list_fused, "ngram_list_fused()");
  m.def("ngram_ptr_prepare", &ngram_ptr_prepare, "ngram_ptr_prepare()");
  m.def("ngram_list_fused_prepared", &ngram_list_fused_prepared, "ngram_list_fused_prepared()", py::arg("tensor_list"), py::arg("n_of_gram_list"), py::arg("width_list_ptr"), py::arg("n_of_gram_list_ptr"), py::arg("if_kernel")=1);
  m.def("ngram_cpu_part_base", &ngram_cpu_part_base, "ngram_cpu_part_base()");
  m.def("ngram_cpu_part", &ngram_cpu_part, "ngram_cpu_part()");
  m.def("ngram_gpu_part", &ngram_gpu_part, "ngram_gpu_part()");


  m.def("mapid", &mapid_main, "mapid_main()");
  m.def("mapid_list_fused", &mapid_list_fused, "mapid_list_fused()");
  m.def("mapid_list_fused_prepared", &mapid_list_fused_prepared, "mapid_list_fused_prepared()", py::arg("tensor_list"), py::arg("mapping_list_ptr"), py::arg("width_list"), py::arg("mapping_width_list"), py::arg("if_kernel") = 1);
//   m.def("mapid_list_fused_prepared", &mapid_list_fused_prepared, "mapid_list_fused_prepared()", py::arg("if_kernel") = 1);
  m.def("mapid_ptr_prepare", &mapid_ptr_prepare, "mapid_ptr_prepare()");
  m.def("mapid_cpu_part_base", &mapid_cpu_part_base, "mapid_cpu_part_base()");
  m.def("mapid_cpu_part", &mapid_cpu_part, "mapid_cpu_part()");
  m.def("mapid_gpu_part", &mapid_gpu_part, "mapid_gpu_part()");
}

