#include<cuda.h>
#include<stdio.h>
#include <functional>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <tuple>

using namespace std;
#include <chrono>
using namespace std::chrono;

using uint128_t = unsigned __int128;
bool graphCreated=false;
cudaGraph_t graph;
cudaGraphExec_t instance;

//constexpr uint64_t _FNV_offset_basis = 14695981039346656037ULL;
//constexpr uint64_t _FNV_prime        = 1099511628211ULL;

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }
}

void init_cuda(
    int32_t device_id
)
{
    cudaSetDevice(device_id);
}


template <typename T>
int64_t copy_vector_to_GPU(const std::vector<T>& input) {
    uint64_t* device_array = nullptr;
    size_t size = input.size() * sizeof(T);

    // Allocate GPU memory
    cudaError_t err = cudaMalloc((void**)&device_array, size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory: " << cudaGetErrorString(err) << std::endl;
        return 0;
    }

    // Copy the input vector to GPU memory
    err = cudaMemcpy(device_array, input.data(), size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy vector to GPU: " << cudaGetErrorString(err) << std::endl;
        cudaFree(device_array); // Release the allocated memory
        return 0;
    }

    
    int64_t pointer_as_int64 = reinterpret_cast<int64_t>(device_array);
    return pointer_as_int64;
}

int64_t copy_vector_to_GPU_uint64(const std::vector<uint64_t>& input) {
    return copy_vector_to_GPU(input);
}

int64_t copy_vector_to_GPU_int64(const std::vector<int64_t>& input) {
    return copy_vector_to_GPU(input);
}

int64_t copy_vector_to_GPU_int32(const std::vector<int32_t>& input) {
    return copy_vector_to_GPU(input);
}

int64_t copy_vector_to_GPU_float(const std::vector<float>& input) {
    return copy_vector_to_GPU(input);
}

int64_t copy_vector_to_GPU_float_ptr(const std::vector<float*>& input) {
    return copy_vector_to_GPU(input);
}

int64_t copy_vector_to_GPU_int32_ptr(const std::vector<int32_t*>& input) {
    return copy_vector_to_GPU(input);
}

int64_t copy_vector_to_GPU_int64_ptr(const std::vector<int64_t*>& input) {
    return copy_vector_to_GPU(input);
}

int64_t get_shape_list(const std::vector<torch::Tensor> tensor_list, int dim)
{
    std::vector<int32_t> shape_list;
    for (int i = 0; i < tensor_list.size(); i++)
    {
        shape_list.push_back(tensor_list[i].size(dim));
    }
    return copy_vector_to_GPU_int32(shape_list);
}

int64_t copy_tensor_list_to_GPU(
    std::vector<torch::Tensor> tensor_list
){
    if(tensor_list[0].scalar_type() == at::kLong){
        std::vector<int64_t*> tensor_list_ptr;
        for(int i=0; i<tensor_list.size(); i++){
            tensor_list_ptr.push_back(tensor_list[i].data_ptr<int64_t>());
        }    
        int64_t pointer_as_int64 = copy_vector_to_GPU(tensor_list_ptr);
        return pointer_as_int64;
    }
    else{ // input_tensor.scalar_type() == at::kFloat
        std::vector<float*> tensor_list_ptr;
        for(int i=0; i<tensor_list.size(); i++){
            tensor_list_ptr.push_back(tensor_list[i].data_ptr<float>());
        }
        int64_t pointer_as_int64 = copy_vector_to_GPU(tensor_list_ptr);
        return pointer_as_int64;
    }
}

torch::Tensor copy_tensor_list_to_GPU_tensor(
    std::vector<torch::Tensor> tensor_list
){
    int32_t list_len = tensor_list.size();
    // int64 tensor to save the pointer
    // get the device of input tensor list  
    torch::Device device(tensor_list[0].device());
    // create a tensor to save the pointer with the same device of input tensor list
    torch::Tensor tensor_ptr = torch::zeros({list_len}, at::kLong).to(device);
    
    if(tensor_list[0].scalar_type() == at::kLong){
        for(int i=0; i<tensor_list.size(); i++){
            tensor_ptr[i] = reinterpret_cast<int64_t>(tensor_list[i].data_ptr<int64_t>());
        }    
    }
    else{ // input_tensor.scalar_type() == at::kFloat
        for(int i=0; i<tensor_list.size(); i++){
            tensor_ptr[i] = reinterpret_cast<int64_t>(tensor_list[i].data_ptr<float>());
        }
    }
    return tensor_ptr;
}



// =======================================================================================================================================
// ======================================================= Sigrid Hash ===================================================================
// =======================================================================================================================================

__device__
uint64_t twang_mix64(uint64_t key)
{
    key = (~key) + (key << 21); // key *= (1 << 21) - 1; key -= 1;
    key = key ^ (key >> 24);
    key = key + (key << 3) + (key << 8); // key *= 1 + (1 << 3) + (1 << 8)
    key = key ^ (key >> 14);
    key = key + (key << 2) + (key << 4); // key *= 1 + (1 << 2) + (1 << 4)
    key = key ^ (key >> 28);
    key = key + (key << 31); // key *= 1 + (1 << 31)
    return key;
}

__device__
uint64_t hashfunction(uint64_t value)
{
    constexpr uint64_t _FNV_offset_basis = 14695981039346656037ULL;
    constexpr uint64_t _FNV_prime = 1099511628211ULL;
    unsigned char*_First=(unsigned char*)(&value);
    uint64_t _Val = _FNV_offset_basis;
    for (int _Idx = 0; _Idx < 8; ++_Idx) {
        _Val ^= uint64_t(_First[_Idx]);
        _Val *= _FNV_prime;
    }
    return _Val;
}

__device__
uint64_t hash_128_to_64(const uint64_t&upper, const uint64_t&lower)
{
  const uint64_t kMul = 0x9ddfea08eb382d69ULL;
  uint64_t a = (lower ^ upper) * kMul;
  a ^= (a >> 47);
  uint64_t b = (upper ^ a) * kMul;
  b ^= (b >> 47);
  b *= kMul;
  return b;
}


__global__
void sigrid_hash(uint64_t*tensor_in, int tensor_size,const int64_t salt,const int64_t maxValue,const uint64_t multiplier,const int shift)
{
    int idx=blockDim.x*blockIdx.x+threadIdx.x;

    // printf("blockDim.x:%d, blockDim.y:%d, gridDim.x:%d, gridDim.y:%d, blockIdx.x:%d,  blockIdx.y:%d, threadIdx.x:%d, threadIdx.y:%d\n", blockDim.x, blockDim.y, gridDim.x, gridDim.y, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
    if(idx<tensor_size)
    {
        uint64_t result;
        uint64_t upper = hashfunction(salt);
        uint64_t lower = hashfunction(twang_mix64(tensor_in[idx]));
        int64_t hashed = hash_128_to_64(upper,lower);
        
        if (maxValue > 1) {
            int64_t sign = hashed >> (64 - 1);
            // __int128 left=((__int128)(multiplier));
            uint64_t right=sign ^ hashed;
            // int64_t q = sign^((left * right)>>(64 + shift)); 
            int64_t q = sign^((__umul64hi(multiplier, right))>>(shift));   
            result = hashed - q * maxValue;
        }
        else
        {
            result=hashed;
        }
        tensor_in[idx]=result;
    }
}

__global__
void sigrid_hash_fused(uint64_t**tensor_in_list, const int64_t salt, int64_t* maxValue_list, uint64_t* multiplier_list, int32_t* shift_list, int32_t batch_size, int32_t length)
{
    int idx=blockDim.x*blockIdx.y+threadIdx.x;
    int row_idx=blockIdx.x;

    if(idx < batch_size && row_idx < length)
    {
        int64_t maxValue = maxValue_list[row_idx];
        uint64_t multiplier = multiplier_list[row_idx];
        int shift = shift_list[row_idx];
       
        uint64_t* tensor_in = tensor_in_list[row_idx];

        uint64_t result;
        uint64_t upper = hashfunction(salt);
        uint64_t lower = hashfunction(twang_mix64(tensor_in[idx]));
        int64_t hashed = hash_128_to_64(upper,lower);

        if (maxValue > 1) {
            int64_t sign = hashed >> (64 - 1);
            uint64_t right=sign ^ hashed;
            int64_t q = sign^((__umul64hi(multiplier, right))>>(shift));   
            result = hashed - q * maxValue;
        }
        else
        {
            result=hashed;
        }
        tensor_in[idx]=result;
    }
}

__global__
void sigrid_hash_fused_multi_dim(uint64_t**tensor_in_list, int32_t* d_offset_list, const int64_t salt, int64_t* maxValue_list, uint64_t* multiplier_list, int32_t* shift_list, int32_t batch_size, int32_t length)
{
    int idx=blockDim.x*blockIdx.y+threadIdx.x;
    int row_idx=blockIdx.x;

    if(idx < batch_size && row_idx < length)
    {
        uint64_t* tensor_in = tensor_in_list[row_idx];
        int32_t offset = d_offset_list[row_idx];
        
        int64_t maxValue = maxValue_list[offset];
        uint64_t multiplier = multiplier_list[offset];
        int shift = shift_list[offset];

        uint64_t result;
        uint64_t upper = hashfunction(salt);
        uint64_t lower = hashfunction(twang_mix64(tensor_in[idx]));
        int64_t hashed = hash_128_to_64(upper,lower);

        if (maxValue > 1) {
            int64_t sign = hashed >> (64 - 1);
            uint64_t right=sign ^ hashed;
            int64_t q = sign^((__umul64hi(multiplier, right))>>(shift));   
            result = hashed - q * maxValue;
        }
        else
        {
            result=hashed;
        }
        if(result < 0 || result >= maxValue)
        {
            result = 0;
        }
        tensor_in[idx]=result;
    }
}

void computeMultiperAndShift(int64_t divisor,int precision,uint64_t&multiplier,int&shift) {
  constexpr int N = 64;
  int l = ceil(std::log2(divisor));
  uint128_t low = (static_cast<uint128_t>(1) << (N + l)) / divisor;
  uint128_t high = ((static_cast<uint128_t>(1) << (N + l)) +
                    ((static_cast<uint128_t>(1) << (N + l - precision)))) / divisor;
  while (low / 2 < high / 2 && l > 0) {
    low = low / 2;
    high = high / 2;
    --l;
  }
  multiplier=(uint64_t)high;
  shift=l;
}



void sigrid_hash_main(
    torch::Tensor tensor,
    int64_t salt,
    int64_t maxValue
)
{
    uint64_t multiplier_;
    int shift_;
    computeMultiperAndShift(maxValue, 63, multiplier_, shift_);    
    
    int tensor_size = tensor.numel();
    
    int thread_num = 128; // 32 512
    sigrid_hash<<<(tensor_size+thread_num-1)/thread_num, thread_num>>>(
        (uint64_t*)tensor.data_ptr(), 
        tensor_size, 
        salt, 
        maxValue, 
        multiplier_, 
        shift_
    );
    cudaDeviceSynchronize();
}

void sigrid_hash_kernel(
    torch::Tensor tensor,
    int64_t salt,
    int64_t maxValue,
    uint64_t multiplier_,
    int shift_
)
{
    int tensor_size = tensor.numel();
    int thread_num = 128; // 32 512
    sigrid_hash<<<(tensor_size+thread_num-1)/thread_num, thread_num>>>(
        (uint64_t*)tensor.data_ptr(), 
        tensor_size, 
        salt, 
        maxValue, 
        multiplier_, 
        shift_
    );
    cudaDeviceSynchronize();
}

std::pair<uint64_t, int> sigrid_hash_data_preprare(
    int64_t maxValue
)
{
    uint64_t multiplier_;
    int shift_;
    computeMultiperAndShift(maxValue, 63, multiplier_, shift_);

    return std::make_pair(multiplier_, shift_);
}


void sigrid_hash_list_fused(
    std::vector<torch::Tensor> tensor_list,
    int64_t salt,
    int64_t maxValue_list,
    int64_t multiplier_list,
    int64_t shift_list,
    int if_kernel=1
)
{
    int length = tensor_list.size();
    int batch_size = tensor_list[0].sizes()[0];

    // ================== thrust implementation (support multi-dim) ==================
    int total_length = 0;
    for(int i=0;i<length;i++){
        total_length += tensor_list[i].sizes()[1];
    }

    thrust::host_vector<uint64_t*> h_tensor_list(total_length);
    thrust::host_vector<int32_t> h_offset_list(total_length);
  
    int offset = 0;
    for(int i=0;i<length;i++){
        for(int j=0;j<tensor_list[i].sizes()[1];j++){
            h_tensor_list[offset] = (uint64_t*)tensor_list[i].data_ptr() + j*batch_size;
            h_offset_list[offset] = i;
            offset += 1;
        }
    }

    thrust::device_vector<uint64_t*> d_tensor_list = h_tensor_list;
    thrust::device_vector<int32_t> d_offset_list = h_offset_list;

    int thread_num = 128; 
    dim3 grid_size(total_length, (batch_size+thread_num-1)/thread_num);

    if(if_kernel){
        sigrid_hash_fused_multi_dim<<<grid_size, thread_num>>>(
            thrust::raw_pointer_cast(d_tensor_list.data()),
            thrust::raw_pointer_cast(d_offset_list.data()),
            salt, 
            reinterpret_cast<int64_t*>(maxValue_list), 
            reinterpret_cast<uint64_t*>(multiplier_list),  
            reinterpret_cast<int32_t*>(shift_list),
            batch_size,
            total_length
        );
    }

    cudaDeviceSynchronize();
    // ===============================================================================
}

std::pair<std::vector<uint64_t>, std::vector<int32_t>> sigrid_hash_list_compute_shift(
    std::vector<uint64_t> maxValue_list
)
{
    
    int length = maxValue_list.size();

    std::vector<uint64_t> multiplier_ptr(length);
    std::vector<int32_t> shift_ptr(length);

    for(int i=0;i<length;i++){
        constexpr int N = 64;
        int64_t divisor = maxValue_list[i];
        int l = ceil(std::log2(divisor));
        uint128_t low = (static_cast<uint128_t>(1) << (N + l)) / divisor;
        uint128_t high = ((static_cast<uint128_t>(1) << (N + l)) +
                            ((static_cast<uint128_t>(1) << (N + l - 63)))) / divisor;
        while (low / 2 < high / 2 && l > 0) {
            low = low / 2;
            high = high / 2;
            --l;
        }
        multiplier_ptr[i]=(uint64_t)high;
        shift_ptr[i]=l;
    }

    return std::make_pair(multiplier_ptr, shift_ptr);
}


std::tuple<int64_t, int64_t, int64_t> sigrid_hash_list_compute_shift_fused(
    std::vector<int64_t> maxValue_list
)
{
    
    int length = maxValue_list.size();

    std::vector<uint64_t> multiplier_ptr(length);
    std::vector<int32_t> shift_ptr(length);

    for(int i=0;i<length;i++){
        constexpr int N = 64;
        int64_t divisor = maxValue_list[i];
        int l = ceil(std::log2(divisor));
        uint128_t low = (static_cast<uint128_t>(1) << (N + l)) / divisor;
        uint128_t high = ((static_cast<uint128_t>(1) << (N + l)) +
                            ((static_cast<uint128_t>(1) << (N + l - 63)))) / divisor;
        while (low / 2 < high / 2 && l > 0) {
            low = low / 2;
            high = high / 2;
            --l;
        }
        multiplier_ptr[i]=(uint64_t)high;
        shift_ptr[i]=l;
    }

    int64_t table_length_gpu = copy_vector_to_GPU_int64(maxValue_list);
    int64_t multiplier_ptr_gpu = copy_vector_to_GPU_uint64(multiplier_ptr);
    int64_t shift_ptr_gpu = copy_vector_to_GPU_int32(shift_ptr);

    return std::make_tuple(table_length_gpu, multiplier_ptr_gpu, shift_ptr_gpu);
}

std::tuple<int64_t, int64_t, int64_t> sigridhash_ptr_prepare(
    std::vector<int64_t> maxValue_list
)
{
    
    int length = maxValue_list.size();

    std::vector<uint64_t> multiplier_ptr(length);
    std::vector<int32_t> shift_ptr(length);

    for(int i=0;i<length;i++){
        constexpr int N = 64;
        int64_t divisor = maxValue_list[i];
        int l = ceil(std::log2(divisor));
        uint128_t low = (static_cast<uint128_t>(1) << (N + l)) / divisor;
        uint128_t high = ((static_cast<uint128_t>(1) << (N + l)) +
                            ((static_cast<uint128_t>(1) << (N + l - 63)))) / divisor;
        while (low / 2 < high / 2 && l > 0) {
            low = low / 2;
            high = high / 2;
            --l;
        }
        multiplier_ptr[i]=(uint64_t)high;
        shift_ptr[i]=l;
    }

    int64_t table_length_gpu = copy_vector_to_GPU_int64(maxValue_list);
    int64_t multiplier_ptr_gpu = copy_vector_to_GPU_uint64(multiplier_ptr);
    int64_t shift_ptr_gpu = copy_vector_to_GPU_int32(shift_ptr);

    return std::make_tuple(table_length_gpu, multiplier_ptr_gpu, shift_ptr_gpu);
}

std::tuple<int64_t, int64_t, int> sigridhash_cpu_part(
    std::vector<torch::Tensor> tensor_list
){
    int length = tensor_list.size();
    int batch_size = tensor_list[0].sizes()[0];

    // ================== thrust implementation (support multi-dim) ==================
    int total_length = 0;
    for(int i=0;i<length;i++){
        total_length += tensor_list[i].sizes()[1];
    }

    thrust::host_vector<uint64_t*> h_tensor_list(total_length);
    thrust::host_vector<int32_t> h_offset_list(total_length);
  
    int offset = 0;
    for(int i=0;i<length;i++){
        for(int j=0;j<tensor_list[i].sizes()[1];j++){
            h_tensor_list[offset] = (uint64_t*)tensor_list[i].data_ptr() + j*batch_size;
            h_offset_list[offset] = i;
            offset += 1;
        }
    }

    thrust::device_vector<uint64_t*> d_tensor_list = h_tensor_list;
    thrust::device_vector<int32_t> d_offset_list = h_offset_list;

    auto d_tensor_ptr = thrust::raw_pointer_cast(d_tensor_list.data());
    auto d_offset_ptr = thrust::raw_pointer_cast(d_offset_list.data());

    // cast to int64_t
    int64_t d_tensor_ptr_int64 = reinterpret_cast<int64_t>(d_tensor_ptr);
    int64_t d_offset_ptr_int64 = reinterpret_cast<int64_t>(d_offset_ptr);

    return std::make_tuple(d_tensor_ptr_int64, d_offset_ptr_int64, total_length);
}

std::tuple<int64_t, int64_t, int> sigridhash_cpu_part_base(
    std::vector<torch::Tensor> tensor_list
){
    int length = tensor_list.size();
    int batch_size = tensor_list[0].sizes()[0];

    // ================== thrust implementation (support multi-dim) ==================
    int total_length = 0;
    for(int i=0;i<length;i++){
        total_length += tensor_list[i].sizes()[1];
    }

    std::vector<uint64_t*> h_tensor_list(total_length);
    std::vector<int32_t> h_offset_list(total_length);
  
    int offset = 0;
    for(int i=0;i<length;i++){
        for(int j=0;j<tensor_list[i].sizes()[1];j++){
            h_tensor_list[offset] = (uint64_t*)tensor_list[i].data_ptr() + j*batch_size;
            h_offset_list[offset] = i;
            offset += 1;
        }
    }

    // thrust::device_vector<uint64_t*> d_tensor_list = h_tensor_list;
    // thrust::device_vector<int32_t> d_offset_list = h_offset_list;

    // auto d_tensor_ptr = thrust::raw_pointer_cast(d_tensor_list.data());
    // auto d_offset_ptr = thrust::raw_pointer_cast(d_offset_list.data());

    // // cast to int64_t
    // int64_t d_tensor_ptr_int64 = reinterpret_cast<int64_t>(d_tensor_ptr);
    // int64_t d_offset_ptr_int64 = reinterpret_cast<int64_t>(d_offset_ptr);

    int64_t d_tensor_ptr_int64 = copy_vector_to_GPU(h_tensor_list);
    int64_t d_offset_ptr_int64 = copy_vector_to_GPU(h_offset_list);

    return std::make_tuple(d_tensor_ptr_int64, d_offset_ptr_int64, total_length);
}

void sigridhash_gpu_part(
    int64_t d_tensor_ptr_int64,
    int64_t d_offset_ptr_int64,
    int64_t salt,
    int64_t maxValue_list,
    int64_t multiplier_list,
    int64_t shift_list,
    int total_length,
    int batch_size
){
    int thread_num = 128; 
    dim3 grid_size(total_length, (batch_size+thread_num-1)/thread_num);
    
    sigrid_hash_fused_multi_dim<<<grid_size, thread_num>>>(
        reinterpret_cast<uint64_t**>(d_tensor_ptr_int64),
        reinterpret_cast<int32_t*>(d_offset_ptr_int64),
        salt, 
        reinterpret_cast<int64_t*>(maxValue_list), 
        reinterpret_cast<uint64_t*>(multiplier_list),  
        reinterpret_cast<int32_t*>(shift_list),
        batch_size,
        total_length
    );

    cudaDeviceSynchronize();
}


void sigrid_hash_list_fused_CUDA_Graph(
    std::vector<torch::Tensor> tensor_list,
    int64_t salt,
    int64_t maxValue_list,
    int64_t multiplier_list,
    int64_t shift_list,
    int batch_size,
    int32_t nBlock
)
{
    int length = tensor_list.size();
    if(nBlock>0 && nBlock<length){
        length = nBlock;
    }
    // printf("length:%d\n", length);
    size_t size = length * sizeof(uint64_t*);


    uint64_t** h_tensor_list = static_cast<uint64_t**>(malloc(length * sizeof(uint64_t*)));
    uint64_t** d_tensor_list = nullptr;
    cudaError_t err = cudaMalloc((void**)&d_tensor_list, size);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    for(int i=0;i<length;i++){
        h_tensor_list[i] = (uint64_t*)tensor_list[i].data_ptr();
    }

    err = cudaMemcpy(d_tensor_list, h_tensor_list, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy vector to GPU: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_tensor_list); // Release the allocated memory
        return;
    }

    // dim3 block_size(32);
    // dim3 grid_size(length, (batch_size+31)/32); // gridDim.x = nTable, gridDim.y = batch_size/32, blockDim.x = 32.
    int thread_num = 128; 
    dim3 grid_size(length, (batch_size+thread_num-1)/thread_num);

    // create new stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    if(!graphCreated){
        printf("create CUDA graph\n");

        err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        if (err != cudaSuccess) {
            std::cerr << "cudaStreamBeginCapture: " << cudaGetErrorString(err) << std::endl;
            return;
        }

        for(int i=0;i<length;i++){
            sigrid_hash<<<(batch_size+thread_num-1)/thread_num, thread_num, 0 ,stream>>>(
                h_tensor_list[i], 
                batch_size, 
                0, 
                65536, 
                9223372036854775809, 
                15
            );
        }
        err = cudaStreamEndCapture(stream, &graph);
        if (err != cudaSuccess) {
            std::cerr << "cudaStreamEndCapture: " << cudaGetErrorString(err) << std::endl;
            return;
        }
        err = cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
        if (err != cudaSuccess) {
            std::cerr << "cudaGraphInstantiate: " << cudaGetErrorString(err) << std::endl;
            return;
        }

        graphCreated=true;
    }
    err = cudaGraphLaunch(instance, stream);
    if (err != cudaSuccess) {
        std::cerr << "cudaGraphLaunch: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    cudaDeviceSynchronize();

    cudaFree(d_tensor_list);
    free(h_tensor_list);
}


// =======================================================================================================================================
// ======================================================== Bucketize ====================================================================
// =======================================================================================================================================
__global__
void bucketize(float*tensor_in,int64_t*tensor_out,int tensor_size, float*borders,int borders_size)
{
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    if(idx<tensor_size)
    {
        float data=tensor_in[idx];
        //lower_bound
        // int index=0;
        // int i;
        // for(i=0;i<borders_size;++i)
        // {
        //     if(data<=borders[i])
        //     {
        //         break;
        //     }
        // }
        // index=i;
        // int result;
        // //index
        // if (index >= borders_size - 1) 
        // {
        //     result=index;
        // }
        // result = data < borders[index + 1] ? index : index + 1;
        int64_t result = 0;
        for(int i=0; i<borders_size; i++){
            if(data>borders[i]){
                result = i+1;
            }
        }
        tensor_out[idx]=result;
    }
}

torch::Tensor bucketize_main(
    torch::Tensor tensor,
    torch::Tensor borders
)
{
    // torch::Tensor output_tensor = torch::zeros(tensor.shape);
    torch::Tensor output_tensor = torch::zeros({tensor.sizes()}, at::kLong).to(at::kCUDA);
    int tensor_size = tensor.numel();

    bucketize<<<(tensor_size+127)/128,128>>>(
        (float*)tensor.data_ptr(),
        (int64_t*)output_tensor.data_ptr(),
        tensor_size, 
        (float*)borders.data_ptr(), 
        borders.numel()
        // borders.size()
    );

    cudaDeviceSynchronize();

    return output_tensor;
}

__global__
void bucketize_fused_kernel(float**d_tensor_list,float**d_border_list, int64_t**tensor_out_list, int batch_size, int* d_length_list, int length)
{
    int idx=blockDim.x*blockIdx.y+threadIdx.x;
    int row_idx=blockIdx.x;

    if(idx < batch_size && row_idx < length)
    {
        float* tensor_in=d_tensor_list[row_idx];
        float* borders=d_border_list[row_idx];
        int borders_size=d_length_list[row_idx];
        // int64_t* tensor_out=out_tensor+row_idx*batch_size;
        int64_t* tensor_out=tensor_out_list[row_idx];
        
        float data=tensor_in[idx];
        //lower_bound
        int64_t result = 0;
        for(int i=0; i<borders_size; i++){
            if(data>borders[i]){
                result = i+1;
            }
        }
        tensor_out[idx]=result;
    }
}

__global__
void bucketize_fused_kernel_base(float**d_tensor_list,float**d_border_list, int64_t* out_tensor, int batch_size, int* d_length_list, int length)
{
    int idx=blockDim.x*blockIdx.y+threadIdx.x;
    int row_idx=blockIdx.x;

    if(idx < batch_size && row_idx < length)
    {
        float* tensor_in=d_tensor_list[row_idx];
        float* borders=d_border_list[row_idx];
        int borders_size=d_length_list[row_idx];
        int64_t* tensor_out=out_tensor+row_idx*batch_size;
        
        float data=tensor_in[idx];
        //lower_bound
        int64_t result = 0;
        for(int i=0; i<borders_size; i++){
            if(data>borders[i]){
                result = i+1;
            }
        }
        tensor_out[idx]=result;
    }
}

std::tuple<int64_t, int64_t> bucketize_ptr_prepare(
    std::vector<torch::Tensor> tensor_list,
    std::vector<torch::Tensor> borders_list
){
    int length = tensor_list.size();
    int total_length = 0;
    for(int i=0;i<length;i++){
        total_length += tensor_list[i].sizes()[1];
    }

    std::vector<float*> h_border_list(total_length);
    std::vector<int32_t> h_length_list(total_length);

    int offset = 0;
    for(int i=0;i<length;i++){
        for(int j=0;j<tensor_list[i].sizes()[1];j++){
            h_border_list[offset] = (float*)borders_list[i].data_ptr();
            h_length_list[offset] = borders_list[i].size(0);
            offset += 1;
        }
    }
    
    int64_t borders_list_ptr = copy_vector_to_GPU_float_ptr(h_border_list);
    int64_t length_list_ptr = copy_vector_to_GPU_int32(h_length_list);

    return std::make_tuple(borders_list_ptr, length_list_ptr);
}

// std::vector<torch::Tensor> bucketize_list_fused_prepared(
//     std::vector<torch::Tensor> tensor_list,
//     std::vector<torch::Tensor> borders_list,
//     int64_t borders_list_ptr,
//     int64_t length_list_ptr,
//     int if_kernel=1
// )
// {
//     int length = tensor_list.size();
//     int batch_size = tensor_list[0].sizes()[0];
//     if( length != borders_list.size() ){
//         throw std::invalid_argument("tensor_list and borders_list must have the same length");
//         exit(0);
//     }

//     // ================== thrust implementation (support multi-dim) ==================
//     int total_length = 0;
//     for(int i=0;i<length;i++){
//         total_length += tensor_list[i].sizes()[1];
//     }

//     torch::Device device(torch::kCUDA);
//     auto options = torch::TensorOptions().device(device).dtype(at::kLong);
//     torch::Tensor aggregated_tensor = torch::empty({total_length, batch_size}, options);
//     std::vector<torch::Tensor> output_tensors;
    
//     thrust::host_vector<float*> h_tensor_list(total_length);
//     thrust::host_vector<int64_t*> h_tensor_out_list(length);

//     int offset = 0;
//     int width = 0;
//     for(int i=0;i<length;i++){
//         width = tensor_list[i].sizes()[1];
//         output_tensors.push_back(aggregated_tensor.slice(0, offset, offset + width).view({batch_size, width}));
//         h_tensor_out_list[i] = (int64_t*)output_tensors[i].data_ptr();
//         for(int j=0;j<tensor_list[i].sizes()[1];j++){
//             h_tensor_list[offset] = (float*)tensor_list[i].data_ptr() + j * batch_size;
//             offset += 1;
//         }
//     }


//     thrust::device_vector<float*> d_tensor_list = h_tensor_list;
//     thrust::device_vector<int64_t*> d_tensor_out_list = h_tensor_out_list;

//     int thread_num = 128; 
//     dim3 grid_size(total_length, (batch_size+thread_num-1)/thread_num);

//     if(if_kernel){
//         bucketize_fused_kernel<<<grid_size, thread_num>>>(
//             thrust::raw_pointer_cast(d_tensor_list.data()),
//             reinterpret_cast<float**>(borders_list_ptr),
//             // (int64_t*)aggregated_tensor.data_ptr(),
//             thrust::raw_pointer_cast(d_tensor_out_list.data()),
//             batch_size,
//             reinterpret_cast<int32_t*>(length_list_ptr),
//             total_length
//         );
//         cudaDeviceSynchronize();
//     }

//     // ===============================================================================

//     return output_tensors;
// }

std::vector<torch::Tensor> bucketize_list_fused_prepared(
    std::vector<torch::Tensor> tensor_list,
    std::vector<torch::Tensor> borders_list,
    int64_t borders_list_ptr,
    int64_t length_list_ptr,
    int if_kernel=1
)
{
    int length = tensor_list.size();
    int batch_size = tensor_list[0].sizes()[0];
    if( length != borders_list.size() ){
        throw std::invalid_argument("tensor_list and borders_list must have the same length");
        exit(0);
    }

    // ================== thrust implementation (support multi-dim) ==================
    int total_length = 0;
    for(int i=0;i<length;i++){
        total_length += tensor_list[i].sizes()[1];
    }

    thrust::host_vector<float*> h_tensor_list(total_length);

    int offset = 0;
    for(int i=0;i<length;i++){
        for(int j=0;j<tensor_list[i].sizes()[1];j++){
            h_tensor_list[offset] = (float*)tensor_list[i].data_ptr() + j * batch_size;
            offset += 1;
        }
    }

    thrust::device_vector<float*> d_tensor_list = h_tensor_list;

    std::vector<torch::Tensor> output_tensors;
    torch::Tensor aggregated_tensor = torch::empty({total_length, batch_size}, at::kLong).to(at::kCUDA);

    int thread_num = 128; 
    dim3 grid_size(total_length, (batch_size+thread_num-1)/thread_num);

    if(if_kernel){
        bucketize_fused_kernel_base<<<grid_size, thread_num>>>(
            thrust::raw_pointer_cast(d_tensor_list.data()),
            reinterpret_cast<float**>(borders_list_ptr),
            (int64_t*)aggregated_tensor.data_ptr(),
            batch_size,
            reinterpret_cast<int32_t*>(length_list_ptr),
            total_length
        );
        cudaDeviceSynchronize();
    }

    offset = 0;
    int width = 0;
    for(int i=0; i<length; i++){
        width = tensor_list[i].sizes()[1];
        // output_tensors.push_back(aggregated_tensor[offset, offset + width].view({batch_size, width}));
        output_tensors.push_back(aggregated_tensor.slice(0, offset, offset + width).view({batch_size, width}));
        offset += width;
    }

    // ===============================================================================

    return output_tensors;
}


std::vector<torch::Tensor> bucketize_list_fused(
    std::vector<torch::Tensor> tensor_list,
    std::vector<torch::Tensor> borders_list
)
{
    int length = tensor_list.size();
    int batch_size = tensor_list[0].sizes()[0];
    if( length != borders_list.size() ){
        throw std::invalid_argument("tensor_list and borders_list must have the same length");
        exit(0);
    }

    // ================== thrust implementation (support multi-dim) ==================
    int total_length = 0;
    for(int i=0;i<length;i++){
        total_length += tensor_list[i].sizes()[1];
    }

    torch::Device device(torch::kCUDA);
    auto options = torch::TensorOptions().device(device).dtype(at::kLong);
    torch::Tensor aggregated_tensor = torch::empty({total_length, batch_size}, options);
    std::vector<torch::Tensor> output_tensors;
    

    thrust::host_vector<float*> h_tensor_list(total_length);
    thrust::host_vector<float*> h_border_list(total_length);
    thrust::host_vector<int32_t> h_length_list(total_length);
    thrust::host_vector<int64_t*> h_tensor_out_list(total_length);
    
    int offset = 0; 
    int width = 0;
    for(int i=0;i<length;i++){
        width = tensor_list[i].sizes()[1];
        output_tensors.push_back(aggregated_tensor.slice(0, offset, offset + width).view({batch_size, width}));
        offset += width;
    }

    offset = 0;
    for(int i=0;i<length;i++){
        for(int j=0;j<tensor_list[i].sizes()[1];j++){
            h_tensor_list[offset] = (float*)tensor_list[i].data_ptr() + j * batch_size;
            h_tensor_out_list[offset] = (int64_t*)output_tensors[i].data_ptr() + j * batch_size;
            h_border_list[offset] = (float*)borders_list[i].data_ptr();
            h_length_list[offset] = borders_list[i].size(0);
            offset += 1;
        }
    }

    thrust::device_vector<float*> d_tensor_list = h_tensor_list;
    thrust::device_vector<float*> d_border_list = h_border_list;
    thrust::device_vector<int32_t> d_length_list = h_length_list;
    thrust::device_vector<int64_t*> d_tensor_out_list = h_tensor_out_list;

    int thread_num = 128; 
    dim3 grid_size(total_length, (batch_size+thread_num-1)/thread_num);

    bucketize_fused_kernel<<<grid_size, thread_num>>>(
        thrust::raw_pointer_cast(d_tensor_list.data()),
        thrust::raw_pointer_cast(d_border_list.data()),
        // (int64_t*)aggregated_tensor.data_ptr(),
        thrust::raw_pointer_cast(d_tensor_out_list.data()),
        batch_size,
        thrust::raw_pointer_cast(d_length_list.data()),
        total_length
    );

    cudaDeviceSynchronize();

    // ===============================================================================

    return output_tensors;
}


std::tuple<int64_t, int64_t, int32_t, std::vector<torch::Tensor>> bucketize_cpu_part(
    std::vector<torch::Tensor> tensor_list,
    std::vector<torch::Tensor> borders_list
){
    int length = tensor_list.size();
    int batch_size = tensor_list[0].sizes()[0];
    if( length != borders_list.size() ){
        throw std::invalid_argument("tensor_list and borders_list must have the same length");
        exit(0);
    }

    // ================== thrust implementation (support multi-dim) ==================
    int total_length = 0;
    for(int i=0;i<length;i++){
        total_length += tensor_list[i].sizes()[1];
    }

    torch::Device device(torch::kCUDA);
    auto options = torch::TensorOptions().device(device).dtype(at::kLong);
    torch::Tensor aggregated_tensor = torch::empty({total_length, batch_size}, options);
    std::vector<torch::Tensor> output_tensors;
    
    // thrust::host_vector<float*> h_tensor_list(total_length);
    // thrust::host_vector<int64_t*> h_tensor_out_list(total_length);
    std::vector<float*> h_tensor_list(total_length);
    std::vector<int64_t*> h_tensor_out_list(total_length);
    
    int offset = 0; 
    int width = 0;
    for(int i=0;i<length;i++){
        width = tensor_list[i].sizes()[1];
        output_tensors.push_back(aggregated_tensor.slice(0, offset, offset + width).view({batch_size, width}));
        offset += width;
    }

    offset = 0;
    for(int i=0;i<length;i++){
        for(int j=0;j<tensor_list[i].sizes()[1];j++){
            h_tensor_list[offset] = (float*)tensor_list[i].data_ptr() + j * batch_size;
            h_tensor_out_list[offset] = (int64_t*)output_tensors[i].data_ptr() + j * batch_size;
            offset += 1;
        }
    }

    // thrust::device_vector<float*> d_tensor_list = h_tensor_list;
    // thrust::device_vector<int64_t*> d_tensor_out_list = h_tensor_out_list;
    // //cast to int64
    // int64_t d_tensor_ptr_int64 = reinterpret_cast<int64_t>(thrust::raw_pointer_cast(d_tensor_list.data()));
    // int64_t d_tensor_out_ptr_int64 = reinterpret_cast<int64_t>(thrust::raw_pointer_cast(d_tensor_out_list.data()));

    int64_t d_tensor_ptr_int64 = copy_vector_to_GPU(h_tensor_list);
    int64_t d_tensor_out_ptr_int64 = copy_vector_to_GPU(h_tensor_out_list);

    // ===============================================================================
    return std::make_tuple(d_tensor_ptr_int64, d_tensor_out_ptr_int64, total_length, output_tensors);
}

std::tuple<int64_t, int64_t, int32_t, std::vector<torch::Tensor>> bucketize_cpu_part_base(
    std::vector<torch::Tensor> tensor_list,
    std::vector<torch::Tensor> output_tensors,
    std::vector<torch::Tensor> borders_list
){
    int length = tensor_list.size();
    int batch_size = tensor_list[0].sizes()[0];
    if( length != borders_list.size() ){
        throw std::invalid_argument("tensor_list and borders_list must have the same length");
        exit(0);
    }

    // ================== thrust implementation (support multi-dim) ==================
    int total_length = 0;
    for(int i=0;i<length;i++){
        total_length += tensor_list[i].sizes()[1];
    }

    std::vector<float*> h_tensor_list(total_length);
    std::vector<int64_t*> h_tensor_out_list(total_length);
    
    int offset = 0; 
    for(int i=0;i<length;i++){
        for(int j=0;j<tensor_list[i].sizes()[1];j++){
            h_tensor_list[offset] = (float*)tensor_list[i].data_ptr() + j * batch_size;
            h_tensor_out_list[offset] = (int64_t*)output_tensors[i].data_ptr() + j * batch_size;
            offset += 1;
        }
    }

    //cast to int64
    int64_t d_tensor_ptr_int64 = copy_vector_to_GPU(h_tensor_list);
    int64_t d_tensor_out_ptr_int64 = copy_vector_to_GPU(h_tensor_out_list);

    // ===============================================================================
    return std::make_tuple(d_tensor_ptr_int64, d_tensor_out_ptr_int64, total_length, output_tensors);
}

void bucketize_gpu_part(
    int64_t tensor_list_ptr,
    int64_t tensor_out_list_ptr,
    int64_t borders_list_ptr,
    int64_t length_list_ptr,
    int32_t total_length,
    int32_t batch_size
){
    int thread_num = 128; 
    dim3 grid_size(total_length, (batch_size+thread_num-1)/thread_num);

    bucketize_fused_kernel<<<grid_size, thread_num>>>(
        // thrust::raw_pointer_cast(d_tensor_list.data()),
        reinterpret_cast<float**>(tensor_list_ptr),
        reinterpret_cast<float**>(borders_list_ptr),
        // (int64_t*)aggregated_tensor.data_ptr(),
        // thrust::raw_pointer_cast(d_tensor_out_list.data()),
        reinterpret_cast<int64_t**>(tensor_out_list_ptr),
        batch_size,
        reinterpret_cast<int32_t*>(length_list_ptr),
        total_length
    );

    cudaDeviceSynchronize();
}


// =======================================================================================================================================
// ======================================================== data loading functions =======================================================
// =======================================================================================================================================
__global__ void fill_null_float_kernel(
    int* d_data_ptr,
    float* tensor_ptr,
    int32_t length
)
{
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    if(idx<length)
    {
        int32_t data = d_data_ptr[idx];
        tensor_ptr[idx] = __int2float_rd(data);
    }
}

__global__ void fill_null_float_kernel_fused(
    int64_t* d_data_ptrs,
    float* tensor_ptr,
    int32_t length,
    int32_t batch_size
)
{
    int idx=blockDim.x*blockIdx.y+threadIdx.x;
    int row_idx=blockIdx.x;
    
    if(idx < batch_size && row_idx < length)
    {
        // printf("row_idx: %d, idx: %d\n", row_idx, idx);
        int* d_data_ptr = (int*)d_data_ptrs[row_idx];
        int32_t data = d_data_ptr[idx];
        tensor_ptr[row_idx*batch_size+idx] =  __int2float_rd(data);
    }
}

__global__ void fill_null_int64_kernel_fused_list(
    int64_t* d_data_ptrs,
    int64_t** tensor_ptr_list,
    int32_t length,
    int32_t batch_size
)
{
    int idx=blockDim.x*blockIdx.y+threadIdx.x;
    int row_idx=blockIdx.x;
    
    if(idx < batch_size && row_idx < length)
    {
        int64_t* tensor_ptr = tensor_ptr_list[row_idx];

        int* d_data_ptr = (int*)d_data_ptrs[row_idx];
        int32_t data = d_data_ptr[idx];
        tensor_ptr[idx] = (int64_t)data;
    }
}

__global__ void fill_null_float_kernel_fused_list(
    int64_t* d_data_ptrs,
    float** tensor_ptr_list,
    int32_t length,
    int32_t batch_size
)
{
    int idx=blockDim.x*blockIdx.y+threadIdx.x;
    int row_idx=blockIdx.x;
    
    if(idx < batch_size && row_idx < length)
    {
        float* tensor_ptr = tensor_ptr_list[row_idx];

        int* d_data_ptr = (int*)d_data_ptrs[row_idx];
        int32_t data = d_data_ptr[idx];
        tensor_ptr[idx] =  __int2float_rd(data);
    }
}

void fill_null_int64_list_gpu_part(
    torch::Tensor data_ptrs,
    int64_t out_tensor_list_ptr,
    int32_t batch_size
){
    int length = data_ptrs.size(0);

    dim3 block_size(128);
    dim3 grid_size(length, (batch_size+127)/128); // gridDim.x = nTable, gridDim.y = batch_size/32, blockDim.x = 32.
 
    fill_null_int64_kernel_fused_list<<<grid_size, block_size>>>(
        (int64_t*)data_ptrs.data_ptr(),
        reinterpret_cast<int64_t**>(out_tensor_list_ptr),
        length,
        batch_size
    );

    cudaDeviceSynchronize();
}

void fill_null_int64_list_gpu_part_tensor(
    torch::Tensor data_ptrs,
    torch::Tensor out_tensor_list_ptr_tensor,
    int32_t batch_size
){
    int length = data_ptrs.size(0);

    dim3 block_size(128);
    dim3 grid_size(length, (batch_size+127)/128); // gridDim.x = nTable, gridDim.y = batch_size/32, blockDim.x = 32.

    int64_t pointer_as_int64 = reinterpret_cast<int64_t>(out_tensor_list_ptr_tensor.data_ptr());
 
    fill_null_int64_kernel_fused_list<<<grid_size, block_size>>>(
        (int64_t*)data_ptrs.data_ptr(),
        reinterpret_cast<int64_t**>(pointer_as_int64),
        length,
        batch_size
    );

    cudaDeviceSynchronize();
}

void fill_null_float_list_gpu_part(
    torch::Tensor data_ptrs,
    int64_t out_tensor_list_ptr,
    int32_t batch_size
){
    int length = data_ptrs.size(0);

    dim3 block_size(128);
    dim3 grid_size(length, (batch_size+127)/128); // gridDim.x = nTable, gridDim.y = batch_size/32, blockDim.x = 32.
 
    fill_null_float_kernel_fused_list<<<grid_size, block_size>>>(
        (int64_t*)data_ptrs.data_ptr(),
        reinterpret_cast<float**>(out_tensor_list_ptr),
        length,
        batch_size
    );

    cudaDeviceSynchronize();
}

void fill_null_float_list_gpu_part_tensor(
    torch::Tensor data_ptrs,
    torch::Tensor out_tensor_list_ptr_tensor,
    int32_t batch_size
){
    int length = data_ptrs.size(0);

    dim3 block_size(128);
    dim3 grid_size(length, (batch_size+127)/128); // gridDim.x = nTable, gridDim.y = batch_size/32, blockDim.x = 32.
    int64_t pointer_as_int64 = reinterpret_cast<int64_t>(out_tensor_list_ptr_tensor.data_ptr());
    
    fill_null_float_kernel_fused_list<<<grid_size, block_size>>>(
        (int64_t*)data_ptrs.data_ptr(),
        reinterpret_cast<float**>(pointer_as_int64),
        length,
        batch_size
    );

    cudaDeviceSynchronize();
}


torch::Tensor fill_null_float(
    int64_t data_ptr,
    int32_t length
)
{
    // torch::Tensor output_tensor = torch::zeros({length, 1}, at::kFloat).to(at::kCUDA);
    torch::Device device(torch::kCUDA);
    auto options = torch::TensorOptions().device(device).dtype(at::kFloat);
    torch::Tensor output_tensor = torch::empty({length, 1}, options);
    
    int* d_data_ptr = (int*)data_ptr;
 
    fill_null_float_kernel<<<(length+127)/128,128>>>(
    // fill_null_float_kernel<<<(length+1024)/1024,1024>>>(
        d_data_ptr,
        (float*)output_tensor.data_ptr(),
        length
    );
    cudaDeviceSynchronize();
    
    return output_tensor;
}

std::vector<torch::Tensor> fill_null_float_fused(
// torch::Tensor fill_null_float_fused(
    torch::Tensor data_ptrs,
    int32_t batch_size,
    int if_kernel=1
)
{
    int length = data_ptrs.size(0);
    std::vector<torch::Tensor> output_tensors;
    // printf("length: %d, batch_size: %d\n", length, batch_size);
    // auto aggregated_tensor = torch::zeros({length, batch_size}, at::kFloat).to(at::kCUDA);
    // torch::Tensor aggregated_tensor = torch::empty({length, batch_size}, at::kFloat).to(at::kCUDA);
    torch::Device device(torch::kCUDA);
    auto options = torch::TensorOptions().device(device).dtype(at::kFloat);
    torch::Tensor aggregated_tensor = torch::empty({length, batch_size}, options);
    
    dim3 block_size(128);
    dim3 grid_size(length, (batch_size+127)/128); // gridDim.x = nTable, gridDim.y = batch_size/32, blockDim.x = 32.
 
    if(if_kernel){
        fill_null_float_kernel_fused<<<grid_size, block_size>>>(
            (int64_t*)data_ptrs.data_ptr(),
            (float*)aggregated_tensor.data_ptr(),
            length,
            batch_size
        );
    }

    cudaDeviceSynchronize();

    for(int i=0; i<length; i++)
        output_tensors.push_back(aggregated_tensor[i].view({batch_size, 1}));

    return output_tensors;
}

std::vector<torch::Tensor> fill_null_float_fused_inplace(
    torch::Tensor data_ptrs,
    torch::Tensor output_tensor,
    int32_t batch_size
)
{
    int length = data_ptrs.size(0);
    std::vector<torch::Tensor> output_tensors;

    dim3 block_size(128);
    dim3 grid_size(length, (batch_size+127)/128); // gridDim.x = nTable, gridDim.y = batch_size/32, blockDim.x = 32.
 
    fill_null_float_kernel_fused<<<grid_size, block_size>>>(
        (int64_t*)data_ptrs.data_ptr(),
        (float*)output_tensor.data_ptr(),
        length,
        batch_size
    );

    cudaDeviceSynchronize();

    for(int i=0; i<length; i++)
        output_tensors.push_back(output_tensor[i].view({batch_size, 1}));

    return output_tensors;
}

void fill_null_float_inplace(
    torch::Tensor output_tensor,
    int64_t data_ptr,
    int32_t length
)
{
    int* d_data_ptr = (int*)data_ptr;
 
    fill_null_float_kernel<<<(length+127)/128,128>>>(
    // fill_null_float_kernel<<<(length+1024)/1024,1024>>>(
        d_data_ptr,
        (float*)output_tensor.data_ptr(),
        length
    );
    cudaDeviceSynchronize();
}

__global__ void fill_null_int64_kernel(
    int* d_data_ptr,
    int64_t* tensor_ptr,
    int32_t length
)
{
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    if(idx<length)
    {
        int32_t data = d_data_ptr[idx];
        tensor_ptr[idx] = (int64_t)data;
    }
}


__global__ void fill_null_int64_kernel_fused(
    int64_t* d_data_ptrs,
    int64_t* tensor_ptr,
    int32_t length,
    int32_t batch_size
)
{
    int idx=blockDim.x*blockIdx.y+threadIdx.x;
    int row_idx=blockIdx.x;
    
    if(idx < batch_size && row_idx < length)
    {
        int* d_data_ptr = (int*)d_data_ptrs[row_idx];
        int32_t data = d_data_ptr[idx];
        tensor_ptr[row_idx*batch_size+idx] = (int64_t)data;
    }
}

std::vector<torch::Tensor> fill_null_int64_fused_inplace(
    torch::Tensor data_ptrs,
    torch::Tensor output_tensor,
    int32_t batch_size
)
{
    int length = data_ptrs.size(0);
    std::vector<torch::Tensor> output_tensors;
    
    dim3 block_size(128);
    dim3 grid_size(length, (batch_size+127)/128); // gridDim.x = nTable, gridDim.y = batch_size/32, blockDim.x = 32.

    fill_null_int64_kernel_fused<<<grid_size, block_size>>>(
        (int64_t*)data_ptrs.data_ptr(),
        (int64_t*)output_tensor.data_ptr(),
        length,
        batch_size
    );

    cudaDeviceSynchronize();

    for(int i=0; i<length; i++)
        output_tensors.push_back(output_tensor[i].view({batch_size, 1}));

    return output_tensors;
}

torch::Tensor fill_null_int64(
    int64_t data_ptr,
    int32_t length
)
{
    // torch::Tensor output_tensor = torch::zeros({length, 1}, at::kLong).to(at::kCUDA);
    torch::Device device(torch::kCUDA);
    auto options = torch::TensorOptions().device(device).dtype(at::kLong);
    torch::Tensor output_tensor = torch::empty({length, 1}, options);

    int* d_data_ptr = (int*)data_ptr;
 
    fill_null_int64_kernel<<<(length+127)/128,128>>>(
        d_data_ptr,
        (int64_t*)output_tensor.data_ptr(),
        length
    );
    cudaDeviceSynchronize();
    
    return output_tensor;
}


std::vector<torch::Tensor> fill_null_int64_fused(
    torch::Tensor data_ptrs,
    int32_t batch_size,
    int if_kernel=1
)
{
    int length = data_ptrs.size(0);
    std::vector<torch::Tensor> output_tensors;
   
    // auto aggregated_tensor = torch::empty({length, batch_size}, at::kLong).to(at::kCUDA);
    torch::Device device(torch::kCUDA);
    auto options = torch::TensorOptions().device(device).dtype(at::kLong);
    torch::Tensor aggregated_tensor = torch::empty({length, batch_size}, options);

    dim3 block_size(128);
    dim3 grid_size(length, (batch_size+127)/128); // gridDim.x = nTable, gridDim.y = batch_size/32, blockDim.x = 32.
 
    if(if_kernel){
        fill_null_int64_kernel_fused<<<grid_size, block_size>>>(
            (int64_t*)data_ptrs.data_ptr(),
            (int64_t*)aggregated_tensor.data_ptr(),
            length,
            batch_size
        );
    }

    cudaDeviceSynchronize();
    
    for(int i=0; i<length; i++)
        output_tensors.push_back(aggregated_tensor[i].view({batch_size, 1}));

    return output_tensors;
}


void fill_null_int64_inplace(
    torch::Tensor output_tensor,
    int64_t data_ptr,
    int32_t length
)
{
    int* d_data_ptr = (int*)data_ptr;
 
    fill_null_int64_kernel<<<(length+127)/128,128>>>(
        d_data_ptr,
        (int64_t*)output_tensor.data_ptr(),
        length
    );
    cudaDeviceSynchronize();
}


// =======================================================================================================================================
// ================================================================= logit ===============================================================
// =======================================================================================================================================

__global__ void logit_fused_kernel(
    float** d_data_ptrs,
    int32_t length,
    int32_t batch_size,
    float* eps_list
)
{
    int idx=blockDim.x*blockIdx.y+threadIdx.x;
    int row_idx=blockIdx.x;
    
    if(idx < batch_size && row_idx < length)
    {
        float* tensor_in = d_data_ptrs[row_idx];
        float eps = eps_list[row_idx];

        float tmp = tensor_in[idx];
        if(tmp < eps)
            tmp = eps;
        else if(tmp > 1.0f-eps)
            tmp = 1.0f-eps;
        float result = logf(tmp/(1.0f-tmp));
        tensor_in[idx] = result;
    }
}


void logit_list_fused(
    std::vector<torch::Tensor> tensor_list,
    std::vector<float> eps_list
)
{
    int length = tensor_list.size();
    int32_t batch_size = tensor_list[0].sizes()[0];
    // ================== thrust implementation (support multi-dim) ==================
    int total_length = 0;
    for(int i=0;i<length;i++){
        total_length += tensor_list[i].sizes()[1];
    }

    thrust::host_vector<float*> h_tensor_list(total_length);
    thrust::host_vector<float> h_eps_list(total_length);

    int offset = 0;
    for(int i=0;i<length;i++){
        for(int j=0;j<tensor_list[i].sizes()[1];j++){
            h_tensor_list[offset] = (float*)tensor_list[i].data_ptr() + j * batch_size;
            h_eps_list[offset] = eps_list[i];
            offset += 1;
        }
    }

    thrust::device_vector<float*> d_tensor_list = h_tensor_list;
    thrust::device_vector<float> d_eps_list = h_eps_list;

    int thread_num = 128; 
    dim3 grid_size(total_length, (batch_size+thread_num-1)/thread_num);
    logit_fused_kernel<<<grid_size, thread_num>>>(
        thrust::raw_pointer_cast(d_tensor_list.data()),
        total_length, 
        batch_size, 
        thrust::raw_pointer_cast(d_eps_list.data())
    );

    cudaDeviceSynchronize();
    // ===============================================================================
}


int64_t logit_ptr_prepare(
    std::vector<torch::Tensor> tensor_list,
    std::vector<float> eps_list
)
{
    int length = tensor_list.size();

    int total_length = 0;
    for(int i=0;i<length;i++){
        total_length += tensor_list[i].sizes()[1];
    }

    std::vector<float> h_eps_list(total_length);

    int offset = 0;
    for(int i=0;i<length;i++){
        for(int j=0;j<tensor_list[i].sizes()[1];j++){
            h_eps_list[offset] = eps_list[i];
            offset += 1;
        }
    }

    int64_t eps_list_ptr = copy_vector_to_GPU_float(h_eps_list);

    return eps_list_ptr;
}


void logit_list_fused_prepared(
    std::vector<torch::Tensor> tensor_list,
    int64_t eps_list_ptr,
    int if_kernel=1
)
{
    int length = tensor_list.size();
    int32_t batch_size = tensor_list[0].sizes()[0];
    // ================== thrust implementation (support multi-dim) ==================
    int total_length = 0;
    for(int i=0;i<length;i++){
        total_length += tensor_list[i].sizes()[1];
    }

    thrust::host_vector<float*> h_tensor_list(total_length);

    int offset = 0;
    for(int i=0;i<length;i++){
        for(int j=0;j<tensor_list[i].sizes()[1];j++){
            h_tensor_list[offset] = (float*)tensor_list[i].data_ptr() + j * batch_size;
            offset += 1;
        }
    }

    thrust::device_vector<float*> d_tensor_list = h_tensor_list;

    int thread_num = 128; 
    dim3 grid_size(total_length, (batch_size+thread_num-1)/thread_num);
    if(if_kernel){
        logit_fused_kernel<<<grid_size, thread_num>>>(
            thrust::raw_pointer_cast(d_tensor_list.data()),
            total_length, 
            batch_size, 
            reinterpret_cast<float*>(eps_list_ptr)
        );
    }

    cudaDeviceSynchronize();
    // ===============================================================================
}

std::pair<int64_t, int32_t> logit_cpu_part(
    std::vector<torch::Tensor> tensor_list
)
{
    int length = tensor_list.size();
    int32_t batch_size = tensor_list[0].sizes()[0];
    // ================== thrust implementation (support multi-dim) ==================
    int total_length = 0;
    for(int i=0;i<length;i++){
        total_length += tensor_list[i].sizes()[1];
    }

    thrust::host_vector<float*> h_tensor_list(total_length);

    int offset = 0;
    for(int i=0;i<length;i++){
        for(int j=0;j<tensor_list[i].sizes()[1];j++){
            h_tensor_list[offset] = (float*)tensor_list[i].data_ptr() + j * batch_size;
            offset += 1;
        }
    }

    thrust::device_vector<float*> d_tensor_list = h_tensor_list;

    int64_t d_tensor_ptr_int64 = reinterpret_cast<int64_t>(thrust::raw_pointer_cast(d_tensor_list.data()));
    
    return std::make_pair(d_tensor_ptr_int64, total_length);
}

std::pair<int64_t, int32_t> logit_cpu_part_base(
    std::vector<torch::Tensor> tensor_list
)
{
    int length = tensor_list.size();
    int32_t batch_size = tensor_list[0].sizes()[0];
    // ================== thrust implementation (support multi-dim) ==================
    int total_length = 0;
    for(int i=0;i<length;i++){
        total_length += tensor_list[i].sizes()[1];
    }

    std::vector<float*> h_tensor_list(total_length);

    int offset = 0;
    for(int i=0;i<length;i++){
        for(int j=0;j<tensor_list[i].sizes()[1];j++){
            h_tensor_list[offset] = (float*)tensor_list[i].data_ptr() + j * batch_size;
            offset += 1;
        }
    }

    int64_t d_tensor_ptr_int64 = copy_vector_to_GPU(h_tensor_list);
    
    return std::make_pair(d_tensor_ptr_int64, total_length);
}

void logit_gpu_part(
    int64_t d_tensor_ptr_int64,
    int64_t eps_list_ptr,
    int32_t total_length,
    int32_t batch_size
)
{

    int thread_num = 128; 
    dim3 grid_size(total_length, (batch_size+thread_num-1)/thread_num);
    logit_fused_kernel<<<grid_size, thread_num>>>(
        // thrust::raw_pointer_cast(d_tensor_list.data()),
        reinterpret_cast<float**>(d_tensor_ptr_int64),
        total_length, 
        batch_size, 
        reinterpret_cast<float*>(eps_list_ptr)
    );

    cudaDeviceSynchronize();
    // ===============================================================================
}


__global__ void logit_kernel(
    float* tensor_in,
    int32_t batch_size,
    float eps
)
{
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    
    if(idx < batch_size)
    {
        float tmp = tensor_in[idx];
        if(tmp < eps)
            tmp = eps;
        else if(tmp > 1.0f-eps)
            tmp = 1.0f-eps;
        float result = logf(tmp/(1.0f-tmp));
        tensor_in[idx] = result;
    }
}


void logit_main(
    torch::Tensor input_tensor,
    float eps
)
{
    int thread_num = 128; 
    int nItem = input_tensor.numel();
    int nBlock = (nItem+thread_num-1)/thread_num;
    logit_kernel<<< nBlock, thread_num>>>((float*)input_tensor.data_ptr(), nItem, eps);
    cudaDeviceSynchronize();
}



// =======================================================================================================================================
// ================================================================= FirstX ==============================================================
// =======================================================================================================================================

template <typename T>
__global__ void firstx_kernel_fused(
    T** tensor_in_list,
    T** tensor_out_list,
    int32_t batch_size,
    int32_t* n_list, // the x of first x
    int32_t length,
    int32_t* width_list
)
{
    // int idx=blockDim.x*blockIdx.x+threadIdx.x;
    int idx=blockDim.x*blockIdx.y+threadIdx.x;
    int row_idx=blockIdx.x;

    if(idx < batch_size && row_idx < length)
    {
        T* tensor_in = tensor_in_list[row_idx];
        T* tensor_out = tensor_out_list[row_idx];
        int32_t n = n_list[row_idx];
        int32_t width = width_list[row_idx];

        for(int i=0;i<n;i++){
            tensor_out[idx*n+i] = tensor_in[idx*width+i];
        }
    }
}

template <typename T>
__global__ void firstx_kernel(
    T* tensor_in,
    T* tensor_out,
    int32_t batch_size,
    int32_t n, // the x of first x
    int32_t width // the width of original tensor
)
{
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    
    if(idx < batch_size)
    {
        for(int i=0;i<n;i++){
            tensor_out[idx*n+i] = tensor_in[idx*width+i];
        }
    }
}

torch::Tensor firstx_main(torch::Tensor input_tensor, int x){
    int batch_size = input_tensor.sizes()[0];
    int width = input_tensor.sizes()[1];

    int thread_num = 128;
    int nBlock = (batch_size + thread_num - 1)/thread_num;
    torch::Tensor output_tensor;
    torch::Device device(torch::kCUDA);

    if(input_tensor.scalar_type() == at::kLong){
        // output_tensor = torch::zeros({batch_size, x}, at::kLong).to(at::kCUDA);
        auto options = torch::TensorOptions().device(device).dtype(at::kLong);
        output_tensor = torch::empty({batch_size, x}, options);
        firstx_kernel<<<nBlock, thread_num>>>(
            (int64_t*)input_tensor.data_ptr(),
            (int64_t*)output_tensor.data_ptr(),
            batch_size,
            x,
            width
        );
    }
    else{ // input_tensor.scalar_type() == at::kFloat
        // output_tensor = torch::zeros({batch_size, x}, at::kFloat).to(at::kCUDA);
        auto options = torch::TensorOptions().device(device).dtype(at::kFloat);
        output_tensor = torch::empty({batch_size, x}, options);
        firstx_kernel<<<nBlock, thread_num>>>(
            (float*)input_tensor.data_ptr(),
            (float*)output_tensor.data_ptr(),
            batch_size,
            x,
            width
        );
    }
    cudaDeviceSynchronize();

    return output_tensor;
}


std::vector<torch::Tensor> firstx_list_fused(std::vector<torch::Tensor> input_tensor_list, std::vector<int> x_list){
    int length = input_tensor_list.size();
    int batch_size = input_tensor_list[0].sizes()[0];

    int thread_num = 128;
    dim3 grid_size(length, (batch_size+thread_num-1)/thread_num);

    std::vector<torch::Tensor> output_tensors;

    thrust::host_vector<int> h_width_list(length);
    thrust::host_vector<int> h_x_list(length);

    int total_length = 0;
    for(int i=0;i<length;i++){
        if(x_list[i]>input_tensor_list[i].sizes()[1]){
            h_x_list[i] = input_tensor_list[i].sizes()[1];
            total_length += input_tensor_list[i].sizes()[1];
        }
        else{
            h_x_list[i] = x_list[i];
            total_length += x_list[i];
        }
        h_width_list[i] = input_tensor_list[i].sizes()[1];
    }

    thrust::device_vector<int> d_width_list(h_width_list);
    thrust::device_vector<int> d_x_list(h_x_list);


    if(input_tensor_list[0].scalar_type() == at::kLong){
        torch::Tensor aggregated_tensor = torch::empty({total_length, batch_size}, at::kLong).to(at::kCUDA);
        thrust::host_vector<int64_t*> h_tensor_in_list(length);
        thrust::host_vector<int64_t*> h_tensor_out_list(length);
        
        int offset = 0;
        for(int i=0; i<length; i++){
            output_tensors.push_back(aggregated_tensor.slice(0, offset, offset + x_list[i]).view({batch_size, x_list[i]}));
            h_tensor_in_list[i] = (int64_t*)input_tensor_list[i].data_ptr();
            h_tensor_out_list[i] = (int64_t*)output_tensors[i].data_ptr();
            offset += x_list[i];
        }

        thrust::device_vector<int64_t*> d_tensor_in_list(h_tensor_in_list);
        thrust::device_vector<int64_t*> d_tensor_out_list(h_tensor_out_list);

        firstx_kernel_fused<<<grid_size, thread_num>>>(
            thrust::raw_pointer_cast(d_tensor_in_list.data()),
            thrust::raw_pointer_cast(d_tensor_out_list.data()),
            batch_size,
            thrust::raw_pointer_cast(d_x_list.data()),
            length,
            thrust::raw_pointer_cast(d_width_list.data())
        );
    }
    else{ // input_tensor.scalar_type() == at::kFloat
        torch::Tensor aggregated_tensor = torch::empty({total_length, batch_size}, at::kFloat).to(at::kCUDA);
        thrust::host_vector<float*> h_tensor_in_list(length);
        thrust::host_vector<float*> h_tensor_out_list(length);
        
        int offset = 0;
        for(int i=0; i<length; i++){
            output_tensors.push_back(aggregated_tensor.slice(0, offset, offset + x_list[i]).view({batch_size, x_list[i]}));
            h_tensor_in_list[i] = (float*)input_tensor_list[i].data_ptr();
            h_tensor_out_list[i] = (float*)output_tensors[i].data_ptr();
            offset += x_list[i];
        }

        thrust::device_vector<float*> d_tensor_in_list(h_tensor_in_list);
        thrust::device_vector<float*> d_tensor_out_list(h_tensor_out_list);

        firstx_kernel_fused<<<grid_size, thread_num>>>(
            thrust::raw_pointer_cast(d_tensor_in_list.data()),
            thrust::raw_pointer_cast(d_tensor_out_list.data()),
            batch_size,
            thrust::raw_pointer_cast(d_x_list.data()),
            length,
            thrust::raw_pointer_cast(d_width_list.data())
        );
    }

    cudaDeviceSynchronize();
    return output_tensors;
}


std::tuple<torch::Tensor, torch::Tensor>  firstx_ptr_prepare(std::vector<torch::Tensor> input_tensor_list, std::vector<int> x_list){
    int length = input_tensor_list.size();
    int batch_size = input_tensor_list[0].sizes()[0];

    // create tensor for x_list and width_list
    torch::Device device(torch::kCUDA);
    torch::Tensor x_list_tensor = torch::empty({length}, at::kInt).to(device);
    torch::Tensor width_list_tensor = torch::empty({length}, at::kInt).to(device);

    for(int i=0;i<length;i++){
        width_list_tensor[i] = input_tensor_list[i].sizes()[1];
        if(x_list[i]>input_tensor_list[i].sizes()[1])
            x_list_tensor[i] = input_tensor_list[i].sizes()[1];
        else
            x_list_tensor[i] = x_list[i];
    }

    return std::make_tuple(x_list_tensor, width_list_tensor);
}


std::vector<torch::Tensor> firstx_list_fused_prepared(
    std::vector<torch::Tensor> input_tensor_list, 
    std::vector<int> x_list, 
    // int64_t x_list_ptr, 
    // int64_t width_list_ptr, 
    torch::Tensor x_list_ptr,
    torch::Tensor width_list_ptr,
    int if_kernel=1
){
    int length = input_tensor_list.size();
    int batch_size = input_tensor_list[0].sizes()[0];

    int total_length = 0;
    for(int i=0;i<length;i++){
        if(x_list[i]>input_tensor_list[i].sizes()[1])
            total_length += input_tensor_list[i].sizes()[1];
        else
            total_length += x_list[i];
    }

    int thread_num = 128;
    dim3 grid_size(length, (batch_size+thread_num-1)/thread_num);

    std::vector<torch::Tensor> output_tensors;
    torch::Device device(torch::kCUDA);

    if(input_tensor_list[0].scalar_type() == at::kLong){
        // torch::Tensor aggregated_tensor = torch::empty({total_length, batch_size}, at::kLong).to(at::kCUDA);
        auto options = torch::TensorOptions().device(device).dtype(at::kLong);
        torch::Tensor aggregated_tensor = torch::empty({total_length, batch_size}, options);
        thrust::host_vector<int64_t*> h_tensor_in_list(length);
        thrust::host_vector<int64_t*> h_tensor_out_list(length);
        
        int offset = 0;
        for(int i=0; i<length; i++){
            output_tensors.push_back(aggregated_tensor.slice(0, offset, offset + x_list[i]).view({batch_size, x_list[i]}));
            h_tensor_in_list[i] = (int64_t*)input_tensor_list[i].data_ptr();
            h_tensor_out_list[i] = (int64_t*)output_tensors[i].data_ptr();
            offset += x_list[i];
        }

        thrust::device_vector<int64_t*> d_tensor_in_list(h_tensor_in_list);
        thrust::device_vector<int64_t*> d_tensor_out_list(h_tensor_out_list);

        if(if_kernel){
            firstx_kernel_fused<<<grid_size, thread_num>>>(
                thrust::raw_pointer_cast(d_tensor_in_list.data()),
                thrust::raw_pointer_cast(d_tensor_out_list.data()),
                batch_size,
                reinterpret_cast<int32_t*>(x_list_ptr.data_ptr()),
                length,
                reinterpret_cast<int32_t*>(width_list_ptr.data_ptr())
            );
        }
    }
    else{ // input_tensor.scalar_type() == at::kFloat
        // torch::Tensor aggregated_tensor = torch::empty({total_length, batch_size}, at::kFloat).to(at::kCUDA);
        auto options = torch::TensorOptions().device(device).dtype(at::kFloat);
        torch::Tensor aggregated_tensor = torch::empty({total_length, batch_size}, options);
        thrust::host_vector<float*> h_tensor_in_list(length);
        thrust::host_vector<float*> h_tensor_out_list(length);
        
        int offset = 0;
        for(int i=0; i<length; i++){
            output_tensors.push_back(aggregated_tensor.slice(0, offset, offset + x_list[i]).view({batch_size, x_list[i]}));
            h_tensor_in_list[i] = (float*)input_tensor_list[i].data_ptr();
            h_tensor_out_list[i] = (float*)output_tensors[i].data_ptr();
            offset += x_list[i];
        }

        thrust::device_vector<float*> d_tensor_in_list(h_tensor_in_list);
        thrust::device_vector<float*> d_tensor_out_list(h_tensor_out_list);

        if(if_kernel){
            firstx_kernel_fused<<<grid_size, thread_num>>>(
                thrust::raw_pointer_cast(d_tensor_in_list.data()),
                thrust::raw_pointer_cast(d_tensor_out_list.data()),
                batch_size,
                reinterpret_cast<int32_t*>(x_list_ptr.data_ptr()),
                length,
                reinterpret_cast<int32_t*>(width_list_ptr.data_ptr())
            );
        }
    }

    cudaDeviceSynchronize();
    return output_tensors;
}

std::tuple<int64_t, int64_t, int32_t, std::vector<torch::Tensor>> firstx_cpu_part(
    std::vector<torch::Tensor> input_tensor_list, 
    std::vector<int> x_list
){
    int length = input_tensor_list.size();
    int batch_size = input_tensor_list[0].sizes()[0];

    int total_length = 0;
    for(int i=0;i<length;i++){
        if(x_list[i]>input_tensor_list[i].sizes()[1])
            total_length += input_tensor_list[i].sizes()[1];
        else
            total_length += x_list[i];
    }

    int thread_num = 128;
    dim3 grid_size(length, (batch_size+thread_num-1)/thread_num);

    std::vector<torch::Tensor> output_tensors;
    torch::Device device(torch::kCUDA);

    if(input_tensor_list[0].scalar_type() == at::kLong){
        // torch::Tensor aggregated_tensor = torch::empty({total_length, batch_size}, at::kLong).to(at::kCUDA);
        auto options = torch::TensorOptions().device(device).dtype(at::kLong);
        torch::Tensor aggregated_tensor = torch::empty({total_length, batch_size}, options);
        // thrust::host_vector<int64_t*> h_tensor_in_list(length);
        // thrust::host_vector<int64_t*> h_tensor_out_list(length);
        std::vector<int64_t*> h_tensor_in_list(length);
        std::vector<int64_t*> h_tensor_out_list(length);
        
        int offset = 0;
        for(int i=0; i<length; i++){
            output_tensors.push_back(aggregated_tensor.slice(0, offset, offset + x_list[i]).view({batch_size, x_list[i]}));
            h_tensor_in_list[i] = (int64_t*)input_tensor_list[i].data_ptr();
            h_tensor_out_list[i] = (int64_t*)output_tensors[i].data_ptr();
            offset += x_list[i];
        }

        thrust::device_vector<int64_t*> d_tensor_in_list(h_tensor_in_list);
        thrust::device_vector<int64_t*> d_tensor_out_list(h_tensor_out_list);

        int64_t d_tensor_in_ptr_int64 = copy_vector_to_GPU(h_tensor_in_list);
        int64_t d_tensor_out_ptr_int64 = copy_vector_to_GPU(h_tensor_out_list);

        // int64_t d_tensor_in_ptr_int64 = reinterpret_cast<int64_t>(thrust::raw_pointer_cast(d_tensor_in_list.data()));
        // int64_t d_tensor_out_ptr_int64 = reinterpret_cast<int64_t>(thrust::raw_pointer_cast(d_tensor_out_list.data()));

        return std::make_tuple(d_tensor_in_ptr_int64, d_tensor_out_ptr_int64, length, output_tensors);
        
    }
    else{ // input_tensor.scalar_type() == at::kFloat
        // torch::Tensor aggregated_tensor = torch::empty({total_length, batch_size}, at::kFloat).to(at::kCUDA);
        auto options = torch::TensorOptions().device(device).dtype(at::kFloat);
        torch::Tensor aggregated_tensor = torch::empty({total_length, batch_size}, options);
        // thrust::host_vector<float*> h_tensor_in_list(length);
        // thrust::host_vector<float*> h_tensor_out_list(length);
        std::vector<float*> h_tensor_in_list(length);
        std::vector<float*> h_tensor_out_list(length);
        
        int offset = 0;
        for(int i=0; i<length; i++){
            output_tensors.push_back(aggregated_tensor.slice(0, offset, offset + x_list[i]).view({batch_size, x_list[i]}));
            h_tensor_in_list[i] = (float*)input_tensor_list[i].data_ptr();
            h_tensor_out_list[i] = (float*)output_tensors[i].data_ptr();
            offset += x_list[i];
        }

        // thrust::device_vector<float*> d_tensor_in_list(h_tensor_in_list);
        // thrust::device_vector<float*> d_tensor_out_list(h_tensor_out_list);
        // int64_t d_tensor_in_ptr_int64 = reinterpret_cast<int64_t>(thrust::raw_pointer_cast(d_tensor_in_list.data()));
        // int64_t d_tensor_out_ptr_int64 = reinterpret_cast<int64_t>(thrust::raw_pointer_cast(d_tensor_out_list.data()));

        int64_t d_tensor_in_ptr_int64 = copy_vector_to_GPU(h_tensor_in_list);
        int64_t d_tensor_out_ptr_int64 = copy_vector_to_GPU(h_tensor_out_list);

        return std::make_tuple(d_tensor_in_ptr_int64, d_tensor_out_ptr_int64, length, output_tensors);
    }
}

std::tuple<int64_t, int64_t, int32_t, std::vector<torch::Tensor>> firstx_cpu_part_base(
    std::vector<torch::Tensor> input_tensor_list, 
    std::vector<torch::Tensor> output_tensors, 
    std::vector<int> x_list
){
    int length = input_tensor_list.size();
    int batch_size = input_tensor_list[0].sizes()[0];

    int total_length = 0;
    for(int i=0;i<length;i++){
        if(x_list[i]>input_tensor_list[i].sizes()[1])
            total_length += input_tensor_list[i].sizes()[1];
        else
            total_length += x_list[i];
    }

    int thread_num = 128;
    dim3 grid_size(length, (batch_size+thread_num-1)/thread_num);

    if(input_tensor_list[0].scalar_type() == at::kLong){
        std::vector<int64_t*> h_tensor_in_list(length);
        std::vector<int64_t*> h_tensor_out_list(length);
        
        int offset = 0;
        for(int i=0; i<length; i++){
            h_tensor_in_list[i] = (int64_t*)input_tensor_list[i].data_ptr();
            h_tensor_out_list[i] = (int64_t*)output_tensors[i].data_ptr();
            offset += x_list[i];
        }

        int64_t d_tensor_in_ptr_int64 = copy_vector_to_GPU(h_tensor_in_list);
        int64_t d_tensor_out_ptr_int64 = copy_vector_to_GPU(h_tensor_out_list);

        return std::make_tuple(d_tensor_in_ptr_int64, d_tensor_out_ptr_int64, length, output_tensors);
    }
    else{ // input_tensor.scalar_type() == at::kFloat
        std::vector<float*> h_tensor_in_list(length);
        std::vector<float*> h_tensor_out_list(length);
        
        int offset = 0;
        for(int i=0; i<length; i++){
            h_tensor_in_list[i] = (float*)input_tensor_list[i].data_ptr();
            h_tensor_out_list[i] = (float*)output_tensors[i].data_ptr();
            offset += x_list[i];
        }

        int64_t d_tensor_in_ptr_int64 = copy_vector_to_GPU(h_tensor_in_list);
        int64_t d_tensor_out_ptr_int64 = copy_vector_to_GPU(h_tensor_out_list);

        return std::make_tuple(d_tensor_in_ptr_int64, d_tensor_out_ptr_int64, length, output_tensors);
    }
}

void firstx_gpu_part(
    int64_t d_tensor_in_ptr_int64,
    int64_t d_tensor_out_ptr_int64,
    torch::Tensor x_list_ptr,
    torch::Tensor width_list_ptr,
    int32_t length,
    int32_t batch_size,
    int if_float
)
{
    int thread_num = 128;
    dim3 grid_size(length, (batch_size+thread_num-1)/thread_num);

    if(if_float){
        firstx_kernel_fused<<<grid_size, thread_num>>>(
            reinterpret_cast<float**>(d_tensor_in_ptr_int64),
            reinterpret_cast<float**>(d_tensor_out_ptr_int64),
            batch_size,
            reinterpret_cast<int32_t*>(x_list_ptr.data_ptr()),
            length,
            reinterpret_cast<int32_t*>(width_list_ptr.data_ptr())
        );
    }
    else{
        firstx_kernel_fused<<<grid_size, thread_num>>>(
            reinterpret_cast<int64_t**>(d_tensor_in_ptr_int64),
            reinterpret_cast<int64_t**>(d_tensor_out_ptr_int64),
            batch_size,
            reinterpret_cast<int32_t*>(x_list_ptr.data_ptr()),
            length,
            reinterpret_cast<int32_t*>(width_list_ptr.data_ptr())
        );
    }

    cudaDeviceSynchronize();
}

// =======================================================================================================================================
// ================================================================= BoxCox ==============================================================
// =======================================================================================================================================

__global__ void boxcox_fused_kernel(
    float** d_data_ptrs,
    int32_t length,
    int32_t batch_size,
    float* lambda_list
)
{
    int idx=blockDim.x*blockIdx.y+threadIdx.x;
    int row_idx=blockIdx.x;
    float* tensor_in = d_data_ptrs[row_idx];
    
    if(idx < batch_size && tensor_in[idx] > 0.0f)
    {
        float result = 0.0;
        float lambda = lambda_list[row_idx];

        if(lambda == 0.0f)
            result = logf(tensor_in[idx]);
        else
            result = (powf(tensor_in[idx], lambda) - 1.0f) / lambda;
        tensor_in[idx] = result;
    }
}


void boxcox_list_fused(
    std::vector<torch::Tensor> tensor_list,
    std::vector<float> lambda_list
)
{
    int length = tensor_list.size();
    int32_t batch_size = tensor_list[0].sizes()[0];
    // ================== thrust implementation (support multi-dim) ==================
    int total_length = 0;
    for(int i=0;i<length;i++){
        total_length += tensor_list[i].sizes()[1];
    }

    thrust::host_vector<float*> h_tensor_list(total_length);
    thrust::host_vector<float> h_lambda_list(total_length);

    int offset = 0;
    for(int i=0;i<length;i++){
        for(int j=0;j<tensor_list[i].sizes()[1];j++){
            h_tensor_list[offset] = (float*)tensor_list[i].data_ptr() + j * batch_size;
            h_lambda_list[offset] = lambda_list[i];
            offset += 1;
        }
    }

    thrust::device_vector<float*> d_tensor_list = h_tensor_list;
    thrust::device_vector<float> d_lambda_list = h_lambda_list;

    int thread_num = 128; 
    dim3 grid_size(total_length, (batch_size+thread_num-1)/thread_num);
    boxcox_fused_kernel<<<grid_size, thread_num>>>(
        thrust::raw_pointer_cast(d_tensor_list.data()),
        total_length, 
        batch_size, 
        thrust::raw_pointer_cast(d_lambda_list.data())
    );

    cudaDeviceSynchronize();
    // ===============================================================================
}

int64_t boxcox_ptr_prepare(
    std::vector<torch::Tensor> tensor_list,
    std::vector<float> lambda_list
)
{
    int length = tensor_list.size();

    int total_length = 0;
    for(int i=0;i<length;i++){
        total_length += tensor_list[i].sizes()[1];
    }

    std::vector<float> h_lambda_list(total_length);

    int offset = 0;
    for(int i=0;i<length;i++){
        for(int j=0;j<tensor_list[i].sizes()[1];j++){
            h_lambda_list[offset] = lambda_list[i];
            offset += 1;
        }
    }

    int64_t lambda_list_ptr = copy_vector_to_GPU_float(h_lambda_list);

    return lambda_list_ptr;
}


void boxcox_list_fused_prepared(
    std::vector<torch::Tensor> tensor_list,
    int64_t lambda_list_ptr,
    int if_kernel=1
)
{
    int length = tensor_list.size();
    int32_t batch_size = tensor_list[0].sizes()[0];
    // ================== thrust implementation (support multi-dim) ==================
    int total_length = 0;
    for(int i=0;i<length;i++){
        total_length += tensor_list[i].sizes()[1];
    }

    thrust::host_vector<float*> h_tensor_list(total_length);

    int offset = 0;
    for(int i=0;i<length;i++){
        for(int j=0;j<tensor_list[i].sizes()[1];j++){
            h_tensor_list[offset] = (float*)tensor_list[i].data_ptr() + j * batch_size;
            offset += 1;
        }
    }

    thrust::device_vector<float*> d_tensor_list = h_tensor_list;

    int thread_num = 128; 
    dim3 grid_size(total_length, (batch_size+thread_num-1)/thread_num);
    if(if_kernel){
        boxcox_fused_kernel<<<grid_size, thread_num>>>(
            thrust::raw_pointer_cast(d_tensor_list.data()),
            total_length, 
            batch_size, 
            reinterpret_cast<float*>(lambda_list_ptr)
        );
    }

    cudaDeviceSynchronize();
    // ===============================================================================
}

std::pair<int64_t, int32_t>boxcox_cpu_part(
    std::vector<torch::Tensor> tensor_list
){
    int length = tensor_list.size();
    int32_t batch_size = tensor_list[0].sizes()[0];
    // ================== thrust implementation (support multi-dim) ==================
    int total_length = 0;
    for(int i=0;i<length;i++){
        total_length += tensor_list[i].sizes()[1];
    }

    thrust::host_vector<float*> h_tensor_list(total_length);

    int offset = 0;
    for(int i=0;i<length;i++){
        for(int j=0;j<tensor_list[i].sizes()[1];j++){
            h_tensor_list[offset] = (float*)tensor_list[i].data_ptr() + j * batch_size;
            offset += 1;
        }
    }

    thrust::device_vector<float*> d_tensor_list = h_tensor_list;

    // cast to int64_t
    int64_t d_tensor_ptr_int64 = reinterpret_cast<int64_t>(thrust::raw_pointer_cast(d_tensor_list.data()));

    return std::make_pair(d_tensor_ptr_int64, total_length);
}

std::pair<int64_t, int32_t>boxcox_cpu_part_base(
    std::vector<torch::Tensor> tensor_list
){
    int length = tensor_list.size();
    int32_t batch_size = tensor_list[0].sizes()[0];
    // ================== thrust implementation (support multi-dim) ==================
    int total_length = 0;
    for(int i=0;i<length;i++){
        total_length += tensor_list[i].sizes()[1];
    }

    std::vector<float*> h_tensor_list(total_length);

    int offset = 0;
    for(int i=0;i<length;i++){
        for(int j=0;j<tensor_list[i].sizes()[1];j++){
            h_tensor_list[offset] = (float*)tensor_list[i].data_ptr() + j * batch_size;
            offset += 1;
        }
    }

    // cast to int64_t
    int64_t d_tensor_ptr_int64 = copy_vector_to_GPU(h_tensor_list);

    return std::make_pair(d_tensor_ptr_int64, total_length);
}

void boxcox_gpu_part(
    int64_t tensor_list_ptr,
    int64_t lambda_list_ptr,
    int32_t total_length,
    int32_t batch_size
){
    int thread_num = 128; 
    dim3 grid_size(total_length, (batch_size+thread_num-1)/thread_num);
    boxcox_fused_kernel<<<grid_size, thread_num>>>(
        // thrust::raw_pointer_cast(d_tensor_list.data()),
        reinterpret_cast<float**>(tensor_list_ptr),
        total_length, 
        batch_size, 
        reinterpret_cast<float*>(lambda_list_ptr)
    );

    cudaDeviceSynchronize();
}


__global__ void boxcox_kernel(
    float* tensor_in,
    int32_t batch_size,
    float lambda
)
{
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    
    if(idx < batch_size && tensor_in[idx] > 0.0f)
    {
        float result = 0.0;
        
        if(lambda == 0.0f)
            result = logf(tensor_in[idx]);
        else
            result = (powf(tensor_in[idx], lambda) - 1.0f) / lambda;
        tensor_in[idx] = result;
    }
}


void boxcox_main(
    torch::Tensor input_tensor,
    float lambda
)
{
    int thread_num = 128; 
    int nItem = input_tensor.numel();
    int nBlock = (nItem+thread_num-1)/thread_num;
    boxcox_kernel<<< nBlock, thread_num>>>((float*)input_tensor.data_ptr(), nItem, lambda);
    cudaDeviceSynchronize();
}



// =======================================================================================================================================
// ================================================================= Clamp ===============================================================
// =======================================================================================================================================
__global__ void clamp_fused_kernel(
    int64_t** d_data_ptrs,
    int32_t length,
    int32_t batch_size,
    int64_t* low_list,
    int64_t* high_list
)
{
    int idx=blockDim.x*blockIdx.y+threadIdx.x;
    int row_idx=blockIdx.x;
    int64_t* tensor_in = d_data_ptrs[row_idx];
    
    if(idx < batch_size)
    {
        int64_t low = low_list[row_idx];
        int64_t high = high_list[row_idx];
        if(tensor_in[idx] > high)
            tensor_in[idx] = high;
        if (tensor_in[idx] < low)
            tensor_in[idx] = low;
    }
}

void clamp_list_fused(
    std::vector<torch::Tensor> tensor_list,
    std::vector<int64_t> low_list,
    std::vector<int64_t> high_list
)
{
    int length = tensor_list.size();
    int32_t batch_size = tensor_list[0].sizes()[0];
    // ================== thrust implementation (support multi-dim) ==================
    int total_length = 0;
    for(int i=0;i<length;i++){
        total_length += tensor_list[i].sizes()[1];
    }

    thrust::host_vector<int64_t*> h_tensor_list(total_length);
    thrust::host_vector<int64_t> h_low_list(total_length);
    thrust::host_vector<int64_t> h_high_list(total_length);

    int offset = 0;
    for(int i=0;i<length;i++){
        for(int j=0;j<tensor_list[i].sizes()[1];j++){
            h_tensor_list[offset] = (int64_t*)tensor_list[i].data_ptr() + j * batch_size;
            h_low_list[offset] = low_list[i];
            h_high_list[offset] = high_list[i];
            offset += 1;
        }
    }

    thrust::device_vector<int64_t*> d_tensor_list = h_tensor_list;
    thrust::device_vector<int64_t> d_low_list = h_low_list;
    thrust::device_vector<int64_t> d_high_list = h_high_list;

    int thread_num = 128; 
    dim3 grid_size(total_length, (batch_size+thread_num-1)/thread_num);
    clamp_fused_kernel<<<grid_size, thread_num>>>(
        thrust::raw_pointer_cast(d_tensor_list.data()),
        total_length, 
        batch_size, 
        thrust::raw_pointer_cast(d_low_list.data()),
        thrust::raw_pointer_cast(d_high_list.data())
    );

    cudaDeviceSynchronize();
    // ===============================================================================
}

std::tuple<int64_t, int64_t> clamp_ptr_prepare(
    std::vector<torch::Tensor> tensor_list,
    std::vector<int64_t> low_list,
    std::vector<int64_t> high_list
)
{
    int length = tensor_list.size();

    int64_t low_list_ptr = copy_vector_to_GPU_int64(low_list);
    int64_t high_list_ptr = copy_vector_to_GPU_int64(high_list);


    int total_length = 0;
    for(int i=0;i<length;i++){
        total_length += tensor_list[i].sizes()[1];
    }

    std::vector<int64_t> h_low_list(total_length);
    std::vector<int64_t> h_high_list(total_length);

    int offset = 0;
    for(int i=0;i<length;i++){
        for(int j=0;j<tensor_list[i].sizes()[1];j++){
            h_low_list[offset] = low_list[i];
            h_high_list[offset] = high_list[i];
            offset += 1;
        }
    }

    low_list_ptr = copy_vector_to_GPU_int64(h_low_list);
    high_list_ptr = copy_vector_to_GPU_int64(h_high_list);

    return std::make_tuple(low_list_ptr, high_list_ptr);
}

void clamp_list_fused_prepared(
    std::vector<torch::Tensor> tensor_list,
    int64_t low_list_ptr,
    int64_t high_list_ptr,
    int if_kernel=1
)
{
    int length = tensor_list.size();
    int32_t batch_size = tensor_list[0].sizes()[0];
    // ================== thrust implementation (support multi-dim) ==================
    int total_length = 0;
    for(int i=0;i<length;i++){
        total_length += tensor_list[i].sizes()[1];
    }

    thrust::host_vector<int64_t*> h_tensor_list(total_length);

    int offset = 0;
    for(int i=0;i<length;i++){
        for(int j=0;j<tensor_list[i].sizes()[1];j++){
            h_tensor_list[offset] = (int64_t*)tensor_list[i].data_ptr() + j * batch_size;
            offset += 1;
        }
    }

    thrust::device_vector<int64_t*> d_tensor_list = h_tensor_list;

    int thread_num = 128; 
    dim3 grid_size(total_length, (batch_size+thread_num-1)/thread_num);

    if(if_kernel){
        clamp_fused_kernel<<<grid_size, thread_num>>>(
            thrust::raw_pointer_cast(d_tensor_list.data()),
            total_length, 
            batch_size, 
            reinterpret_cast<int64_t*>(low_list_ptr),
            reinterpret_cast<int64_t*>(high_list_ptr)
        );
    }  

    cudaDeviceSynchronize();
    // ===============================================================================
}


std::pair<int64_t, int32_t> clamp_cpu_part(
    std::vector<torch::Tensor> tensor_list
){
    int length = tensor_list.size();
    int32_t batch_size = tensor_list[0].sizes()[0];
    // ================== thrust implementation (support multi-dim) ==================
    int total_length = 0;
    for(int i=0;i<length;i++){
        total_length += tensor_list[i].sizes()[1];
    }

    thrust::host_vector<int64_t*> h_tensor_list(total_length);

    int offset = 0;
    for(int i=0;i<length;i++){
        for(int j=0;j<tensor_list[i].sizes()[1];j++){
            h_tensor_list[offset] = (int64_t*)tensor_list[i].data_ptr() + j * batch_size;
            offset += 1;
        }
    }

    thrust::device_vector<int64_t*> d_tensor_list = h_tensor_list;
    
    // cast to int64_t
    int64_t d_tensor_ptr_int64 = reinterpret_cast<int64_t>(thrust::raw_pointer_cast(d_tensor_list.data()));

    return std::make_pair(d_tensor_ptr_int64, total_length);
}

std::pair<int64_t, int32_t> clamp_cpu_part_base(
    std::vector<torch::Tensor> tensor_list
){
    int length = tensor_list.size();
    int32_t batch_size = tensor_list[0].sizes()[0];
    // ================== thrust implementation (support multi-dim) ==================
    int total_length = 0;
    for(int i=0;i<length;i++){
        total_length += tensor_list[i].sizes()[1];
    }

    std::vector<int64_t*> h_tensor_list(total_length);

    int offset = 0;
    for(int i=0;i<length;i++){
        for(int j=0;j<tensor_list[i].sizes()[1];j++){
            h_tensor_list[offset] = (int64_t*)tensor_list[i].data_ptr() + j * batch_size;
            offset += 1;
        }
    }

    // cast to int64_t
    int64_t d_tensor_ptr_int64 = copy_vector_to_GPU(h_tensor_list);

    return std::make_pair(d_tensor_ptr_int64, total_length);
}

void clamp_gpu_part(
    int64_t tensor_list_ptr,
    int64_t low_list_ptr,
    int64_t high_list_ptr,
    int32_t total_length,
    int32_t batch_size
){
    int thread_num = 128; 
    dim3 grid_size(total_length, (batch_size+thread_num-1)/thread_num);

    clamp_fused_kernel<<<grid_size, thread_num>>>(
        // thrust::raw_pointer_cast(d_tensor_list.data()),
        reinterpret_cast<int64_t**>(tensor_list_ptr),
        total_length, 
        batch_size, 
        reinterpret_cast<int64_t*>(low_list_ptr),
        reinterpret_cast<int64_t*>(high_list_ptr)
    );

    cudaDeviceSynchronize();
}



__global__ void clamp_kernel(
    int64_t* tensor_in,
    int32_t batch_size,
    int64_t low,
    int64_t high
)
{
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    
    if(idx < batch_size)
    {
        if(tensor_in[idx] > high)
            tensor_in[idx] = high;
        if (tensor_in[idx] < low)
            tensor_in[idx] = low;
    }
}

void clamp_main(
    torch::Tensor input_tensor,
    int64_t low,
    int64_t high
)
{
    int thread_num = 128; 
    int nItem = input_tensor.numel();
    int nBlock = (nItem+thread_num-1)/thread_num;
    clamp_kernel<<< nBlock, thread_num>>>((int64_t*)input_tensor.data_ptr(), nItem, low, high);
    cudaDeviceSynchronize();
}


// =======================================================================================================================================
// ================================================================= OneHot ==============================================================
// =======================================================================================================================================
__global__
void onehot_kernel(float*tensor_in, float*tensor_out, int32_t batch_size, float low, float high, int32_t num_classes)
{
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    if(idx<batch_size)
    {
        float data=tensor_in[idx];
        float step = (high - low) / float(num_classes);
        int32_t index = int((data - low) / step);

        if (index < 0 || index >= num_classes){
            index = 0;
        }

        // for(int i=0; i<num_classes; i++){
        //     tensor_out[idx * num_classes + i] = 0.0;
        // }

        tensor_out[idx * num_classes + index] = 1.0;
    }
}

torch::Tensor onehot_main(
    torch::Tensor tensor,
    float low,
    float high,
    int num_classes
)
{
    // torch::Tensor output_tensor = torch::zeros(tensor.shape);
    int batch_size = tensor.sizes()[0];
    int width = tensor.sizes()[1];
    if(width > 1){
        printf("onehot only support 1-d tensor, the shape should be [batch_size, 1]\n");
        exit(0);
    }

    // torch::Tensor output_tensor = torch::empty({batch_size, num_classes}, at::kFloat).to(at::kCUDA);
    torch::Device device(torch::kCUDA);
    auto options = torch::TensorOptions().device(device).dtype(at::kFloat);
    // torch::Tensor output_tensor = torch::empty({batch_size, num_classes}, options);
    torch::Tensor output_tensor = torch::zeros({batch_size, num_classes}, options);

    onehot_kernel<<<(batch_size+127)/128,128>>>(
        (float*)tensor.data_ptr(),
        (float*)output_tensor.data_ptr(),
        batch_size, 
        low,
        high,
        num_classes
    );
    cudaDeviceSynchronize();
    return output_tensor;
}

__global__
void onehot_fused_kernel(float**tensor_in_list, float**tensor_out_list, int32_t batch_size, float* low_list, float* high_list, int32_t* num_classes_list, int32_t length)
{
    int idx=blockDim.x*blockIdx.y+threadIdx.x;
    int row_idx=blockIdx.x;

    if(idx < batch_size && row_idx < length)
    {
        float* tensor_in=tensor_in_list[row_idx];
        float* tensor_out=tensor_out_list[row_idx];
        float data=tensor_in[idx];
        float low=low_list[row_idx];
        float high=high_list[row_idx];
        int32_t num_classes=num_classes_list[row_idx];

        float step = (high - low) / float(num_classes);
        int32_t index = int((data - low) / step);

        if (index < 0 || index >= num_classes){
            index = 0;
        }

        for(int i=0; i<num_classes; i++){
            tensor_out[idx * num_classes + i] = 0.0;
        }

        tensor_out[idx * num_classes + index] = 1.0;
    }
}

__global__
void onehot_fused_kernel_optimized(float**tensor_list, int32_t batch_size, float* low_list, float* high_list, int32_t* num_classes_list, int32_t length)
{
    int idx=blockDim.x*blockIdx.y+threadIdx.x;
    int row_idx=blockIdx.x;

    if(idx < batch_size && row_idx < length)
    {
        float* tensor_in=tensor_list[row_idx*2];
        float* tensor_out=tensor_list[row_idx*2+1];
        float data=tensor_in[idx];
        float low=low_list[row_idx];
        float high=high_list[row_idx];
        int32_t num_classes=num_classes_list[row_idx];

        float step = (high - low) / float(num_classes);
        int32_t index = int((data - low) / step);

        if (index < 0 || index >= num_classes){
            index = 0;
        }

        // for(int i=0; i<num_classes; i++){
        //     tensor_out[idx * num_classes + i] = 0.0;
        // }

        tensor_out[idx * num_classes + index] = 1.0;
    }
}


std::vector<torch::Tensor> onehot_list_fused(
    std::vector<torch::Tensor> tensor_list,
    std::vector<float> low_list,
    std::vector<float> high_list,
    std::vector<int> num_classes_list
)
{
    int length = tensor_list.size();
    int batch_size = tensor_list[0].sizes()[0];

    // ================== thrust implementation (support multi-dim) ==================
    int total_length = 0;
    for(int i=0;i<length;i++){
        if(tensor_list[i].sizes()[1] > 1){
            printf("onehot only support 1-d tensor, the shape should be [batch_size, 1]\n");
            exit(0);
        }
        total_length += num_classes_list[i];
    }

    std::vector<torch::Tensor> output_tensors;
    // torch::Tensor aggregated_tensor = torch::empty({total_length, batch_size}, at::kFloat).to(at::kCUDA);
    torch::Device device(torch::kCUDA);
    auto options = torch::TensorOptions().device(device).dtype(at::kFloat);
    // torch::Tensor aggregated_tensor = torch::empty({total_length, batch_size}, options);
    torch::Tensor aggregated_tensor = torch::zeros({total_length, batch_size}, options);

    int offset = 0;
    for(int i=0; i<length; i++){
        output_tensors.push_back(aggregated_tensor.slice(0, offset, offset + num_classes_list[i]).view({batch_size, num_classes_list[i]}));
        offset += num_classes_list[i];
    }

    thrust::host_vector<float*> h_tensor_in_list(length);
    thrust::host_vector<float*> h_tensor_out_list(length);
    thrust::host_vector<float> h_low_list(length);
    thrust::host_vector<float> h_high_list(length);
    thrust::host_vector<int32_t> h_num_classes_list(length);
    // thrust::host_vector<float> h_arg_list(length * 3);

    for(int i=0;i<length;i++){
        h_tensor_in_list[i] = (float*)tensor_list[i].data_ptr();
        h_tensor_out_list[i] = (float*)output_tensors[i].data_ptr();
        h_low_list[i] = low_list[i];
        h_high_list[i] = high_list[i];
        h_num_classes_list[i] = num_classes_list[i];
        // h_arg_list[i * 3] = low_list[i];
        // h_arg_list[i * 3 + 1] = high_list[i];
        // h_arg_list[i * 3 + 2] = float(num_classes_list[i]);
    }

    thrust::device_vector<float*> d_tensor_in_list = h_tensor_in_list;
    thrust::device_vector<float*> d_tensor_out_list = h_tensor_out_list;
    thrust::device_vector<float> d_low_list = h_low_list;
    thrust::device_vector<float> d_high_list = h_high_list;
    thrust::device_vector<int32_t> d_num_classes_list = h_num_classes_list;
    // thrust::device_vector<float> d_arg_list = h_arg_list;


    int thread_num = 128; 
    dim3 grid_size(length, (batch_size+thread_num-1)/thread_num);

    onehot_fused_kernel<<<grid_size, thread_num>>>(
        thrust::raw_pointer_cast(d_tensor_in_list.data()),
        thrust::raw_pointer_cast(d_tensor_out_list.data()),
        batch_size,
        thrust::raw_pointer_cast(d_low_list.data()),
        thrust::raw_pointer_cast(d_high_list.data()),
        thrust::raw_pointer_cast(d_num_classes_list.data()),
        length
    );

    cudaDeviceSynchronize();

    // ===============================================================================

    return output_tensors;
}


std::tuple<int64_t, int64_t, int64_t> onehot_ptr_prepare(
    std::vector<torch::Tensor> tensor_list,
    std::vector<float> low_list,
    std::vector<float> high_list,
    std::vector<int> num_classes_list
)
{
    int length = tensor_list.size();

    int64_t low_list_ptr = copy_vector_to_GPU_float(low_list);
    int64_t high_list_ptr = copy_vector_to_GPU_float(high_list);
    int64_t num_classes_list_ptr = copy_vector_to_GPU_int32(num_classes_list);

    return std::make_tuple(low_list_ptr, high_list_ptr, num_classes_list_ptr);
}

std::vector<torch::Tensor> onehot_list_fused_prepared(
    std::vector<torch::Tensor> tensor_list,
    std::vector<int> num_classes_list,
    int64_t low_list_ptr,
    int64_t high_list_ptr,
    int64_t num_classes_list_ptr,
    int if_kernel=1
)
{
    int length = tensor_list.size();
    int batch_size = tensor_list[0].sizes()[0];

    // ================== thrust implementation (support multi-dim) ==================
    int total_length = 0;
    for(int i=0;i<length;i++){
        if(tensor_list[i].sizes()[1] > 1){
            printf("onehot only support 1-d tensor, the shape should be [batch_size, 1]\n");
            exit(0);
        }
        total_length += num_classes_list[i];
    }

    std::vector<torch::Tensor> output_tensors;
    // torch::Tensor aggregated_tensor = torch::empty({total_length, batch_size}, at::kFloat).to(at::kCUDA);
    torch::Device device(torch::kCUDA);
    auto options = torch::TensorOptions().device(device).dtype(at::kFloat);
    // torch::Tensor aggregated_tensor = torch::empty({total_length, batch_size}, options);
    torch::Tensor aggregated_tensor = torch::zeros({total_length, batch_size}, options);

    int offset = 0;
    for(int i=0; i<length; i++){
        output_tensors.push_back(aggregated_tensor.slice(0, offset, offset + num_classes_list[i]).view({batch_size, num_classes_list[i]}));
        offset += num_classes_list[i];
        // output_tensors.push_back(torch::empty({batch_size, num_classes_list[i]}, options));
    }

    // thrust::host_vector<float*> h_tensor_in_list(length);
    // thrust::host_vector<float*> h_tensor_out_list(length);
    thrust::host_vector<float*> h_tensor_list(length * 2);

    for(int i=0;i<length;i++){
        // h_tensor_in_list[i] = (float*)tensor_list[i].data_ptr();
        // h_tensor_out_list[i] = (float*)output_tensors[i].data_ptr();
        h_tensor_list[i * 2] = (float*)tensor_list[i].data_ptr();
        h_tensor_list[i * 2 + 1] = (float*)output_tensors[i].data_ptr();
    }

    // thrust::device_vector<float*> d_tensor_in_list = h_tensor_in_list;
    // thrust::device_vector<float*> d_tensor_out_list = h_tensor_out_list;
    thrust::device_vector<float*> d_tensor_list = h_tensor_list;

    int thread_num = 128; 
    dim3 grid_size(length, (batch_size+thread_num-1)/thread_num);

    if(if_kernel){
        onehot_fused_kernel_optimized<<<grid_size, thread_num>>>(
            // thrust::raw_pointer_cast(d_tensor_in_list.data()),
            // thrust::raw_pointer_cast(d_tensor_out_list.data()),
            thrust::raw_pointer_cast(d_tensor_list.data()),
            batch_size,
            reinterpret_cast<float*>(low_list_ptr),
            reinterpret_cast<float*>(high_list_ptr),
            reinterpret_cast<int32_t*>(num_classes_list_ptr),
            length
        );
    }

    cudaDeviceSynchronize();

    // ===============================================================================

    return output_tensors;
}



std::tuple<int64_t, int64_t, int32_t, std::vector<torch::Tensor>> onehot_cpu_part(
    std::vector<torch::Tensor> tensor_list,
    std::vector<int> num_classes_list
)
{
    int length = tensor_list.size();
    int batch_size = tensor_list[0].sizes()[0];

    // ================== thrust implementation (support multi-dim) ==================
    int total_length = 0;
    for(int i=0;i<length;i++){
        if(tensor_list[i].sizes()[1] > 1){
            printf("onehot only support 1-d tensor, the shape should be [batch_size, 1]\n");
            exit(0);
        }
        total_length += num_classes_list[i];
    }

    std::vector<torch::Tensor> output_tensors;
    torch::Device device(torch::kCUDA);
    auto options = torch::TensorOptions().device(device).dtype(at::kFloat);
    torch::Tensor aggregated_tensor = torch::empty({total_length, batch_size}, options);

    int offset = 0;
    for(int i=0; i<length; i++){
        output_tensors.push_back(aggregated_tensor.slice(0, offset, offset + num_classes_list[i]).view({batch_size, num_classes_list[i]}));
        offset += num_classes_list[i];
    }

    // thrust::host_vector<float*> h_tensor_list(length * 2);
    // thrust::host_vector<float*> h_tensor_in_list(length);
    // thrust::host_vector<float*> h_tensor_out_list(length);
    std::vector<float*> h_tensor_in_list(total_length);
    std::vector<float*> h_tensor_out_list(total_length);

    for(int i=0;i<length;i++){
        h_tensor_in_list[i] = (float*)tensor_list[i].data_ptr();
        h_tensor_out_list[i] = (float*)output_tensors[i].data_ptr();
        // h_tensor_list[i * 2] = (float*)tensor_list[i].data_ptr();
        // h_tensor_list[i * 2 + 1] = (float*)output_tensors[i].data_ptr();
    }

    // thrust::device_vector<float*> d_tensor_list = h_tensor_list;
    // thrust::device_vector<float*> d_tensor_in_list = h_tensor_in_list;
    // thrust::device_vector<float*> d_tensor_out_list = h_tensor_out_list;
    // int64_t d_tensor_in_ptr_int64 = reinterpret_cast<int64_t>(thrust::raw_pointer_cast(d_tensor_in_list.data()));
    // int64_t d_tensor_out_ptr_int64 = reinterpret_cast<int64_t>(thrust::raw_pointer_cast(d_tensor_out_list.data()));

    int64_t d_tensor_in_ptr_int64 = copy_vector_to_GPU(h_tensor_in_list);
    int64_t d_tensor_out_ptr_int64 = copy_vector_to_GPU(h_tensor_out_list);

    return std::make_tuple(d_tensor_in_ptr_int64, d_tensor_out_ptr_int64, length, output_tensors);
}

std::tuple<int64_t, int64_t, int32_t, std::vector<torch::Tensor>> onehot_cpu_part_base(
    std::vector<torch::Tensor> tensor_list,
    std::vector<torch::Tensor> output_tensors,
    std::vector<int> num_classes_list
)
{
    int length = tensor_list.size();
    int batch_size = tensor_list[0].sizes()[0];

    // ================== thrust implementation (support multi-dim) ==================
    int total_length = 0;
    for(int i=0;i<length;i++){
        if(tensor_list[i].sizes()[1] > 1){
            printf("onehot only support 1-d tensor, the shape should be [batch_size, 1]\n");
            exit(0);
        }
        total_length += num_classes_list[i];
    }


    // thrust::host_vector<float*> h_tensor_list(length * 2);
    std::vector<float*> h_tensor_in_list(length);
    std::vector<float*> h_tensor_out_list(length);

    for(int i=0;i<length;i++){
        h_tensor_in_list[i] = (float*)tensor_list[i].data_ptr();
        h_tensor_out_list[i] = (float*)output_tensors[i].data_ptr();
    }


    int64_t d_tensor_in_ptr_int64 = copy_vector_to_GPU(h_tensor_in_list);
    int64_t d_tensor_out_ptr_int64 = copy_vector_to_GPU(h_tensor_out_list);

    return std::make_tuple(d_tensor_in_ptr_int64, d_tensor_out_ptr_int64, length, output_tensors);
}

void onehot_gpu_part(
    int64_t tensor_list_ptr,
    int64_t tensor_out_list_ptr,
    int64_t low_list_ptr,
    int64_t high_list_ptr,
    int64_t num_classes_list_ptr,
    int length,
    int batch_size
)
{
    int thread_num = 128; 
    dim3 grid_size(length, (batch_size+thread_num-1)/thread_num);

    // onehot_fused_kernel_optimized<<<grid_size, thread_num>>>(
    onehot_fused_kernel<<<grid_size, thread_num>>>(
        reinterpret_cast<float**>(tensor_list_ptr),
        reinterpret_cast<float**>(tensor_out_list_ptr),
        batch_size,
        reinterpret_cast<float*>(low_list_ptr),
        reinterpret_cast<float*>(high_list_ptr),
        reinterpret_cast<int32_t*>(num_classes_list_ptr),
        length
    );

    cudaDeviceSynchronize();
}



// =======================================================================================================================================
// ================================================================= N-Gram ==============================================================
// =======================================================================================================================================
__global__
void ngram_kernel(int64_t*tensor_in, int64_t*tensor_out, int32_t batch_size, int32_t width, int32_t n_of_gram)
{
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    if(idx<batch_size)
    {
        int32_t final_length = (width - n_of_gram + 1) * n_of_gram;

        int offset = 0;
        for (int i = 0; i < width - n_of_gram + 1; i++){
            for (int j = 0; j < n_of_gram; j++){
                tensor_out[idx * final_length + offset] = tensor_in[idx * width + i + j];
                offset++;
            }
        }
    }
}

torch::Tensor ngram_main(
    torch::Tensor tensor,
    int n_of_gram
)
{
    // torch::Tensor output_tensor = torch::zeros(tensor.shape);
    int batch_size = tensor.sizes()[0];
    int width = tensor.sizes()[1];

    n_of_gram = std::min(n_of_gram, width);
    int32_t final_length = (width - n_of_gram + 1) * n_of_gram;

    // torch::Tensor output_tensor = torch::empty({batch_size, final_length}, at::kLong).to(at::kCUDA);
    torch::Device device(torch::kCUDA);
    auto options = torch::TensorOptions().device(device).dtype(at::kLong);
    torch::Tensor output_tensor = torch::empty({batch_size, final_length}, options);

    ngram_kernel<<<(batch_size+127)/128,128>>>(
        (int64_t*)tensor.data_ptr(),
        (int64_t*)output_tensor.data_ptr(),
        batch_size, 
        width,
        n_of_gram
    );
    cudaDeviceSynchronize();
    return output_tensor;
}

void ngram_main_inplace(
    torch::Tensor tensor,
    torch::Tensor output_tensor,
    int n_of_gram
)
{
    // torch::Tensor output_tensor = torch::zeros(tensor.shape);
    int batch_size = tensor.sizes()[0];
    int width = tensor.sizes()[1];

    n_of_gram = std::min(n_of_gram, width);
    // int32_t final_length = (width - n_of_gram + 1) * n_of_gram;

    // torch::Tensor output_tensor = torch::empty({batch_size, final_length}, at::kLong).to(at::kCUDA);
    // torch::Device device(torch::kCUDA);
    // auto options = torch::TensorOptions().device(device).dtype(at::kLong);
    // torch::Tensor output_tensor = torch::empty({batch_size, final_length}, options);

    ngram_kernel<<<(batch_size+127)/128,128>>>(
        (int64_t*)tensor.data_ptr(),
        (int64_t*)output_tensor.data_ptr(),
        batch_size, 
        width,
        n_of_gram
    );
    cudaDeviceSynchronize();
    // return output_tensor;
}

__global__
void ngram_fused_kernel(int64_t**tensor_in_list, int64_t**tensor_out_list, int32_t batch_size, int32_t* width_list, int32_t* n_of_gram_list, int32_t length)
{
    int idx=blockDim.x*blockIdx.y+threadIdx.x;
    int row_idx=blockIdx.x;

    if(idx < batch_size && row_idx < length)
    {
        int64_t* tensor_in=tensor_in_list[row_idx];
        int64_t* tensor_out=tensor_out_list[row_idx];

        int width = width_list[row_idx];
        int n_of_gram = n_of_gram_list[row_idx];

        int32_t final_length = (width - n_of_gram + 1) * n_of_gram;

        int offset = 0;
        for (int i = 0; i < width - n_of_gram + 1; i++){
            for (int j = 0; j < n_of_gram; j++){
                tensor_out[idx * final_length + offset] = tensor_in[idx * width + i + j];
                offset++;
            }
        }
    }
}


std::vector<torch::Tensor> ngram_list_fused(
    std::vector<torch::Tensor> tensor_list,
    std::vector<int> n_of_gram_list
)
{
    int length = tensor_list.size();
    int batch_size = tensor_list[0].sizes()[0];

    // ================== thrust implementation (support multi-dim) ==================

    thrust::host_vector<int64_t*> h_tensor_in_list(length);
    thrust::host_vector<int64_t*> h_tensor_out_list(length);
    thrust::host_vector<int32_t> h_width_list(length);
    thrust::host_vector<int32_t> h_n_of_gram_list(length);

    int total_length = 0;
    for(int i=0;i<length;i++){
        h_n_of_gram_list[i] = n_of_gram_list[i];
        if(n_of_gram_list[i] > tensor_list[i].sizes()[1])
            h_n_of_gram_list[i] = tensor_list[i].sizes()[1];

        h_width_list[i] = tensor_list[i].sizes()[1];
        total_length += (h_width_list[i] - h_n_of_gram_list[i] + 1) * h_n_of_gram_list[i];
    }

    std::vector<torch::Tensor> output_tensors;
    // torch::Tensor aggregated_tensor = torch::empty({total_length, batch_size}, at::kLong).to(at::kCUDA);
    torch::Device device(torch::kCUDA);
    auto options = torch::TensorOptions().device(device).dtype(at::kLong);
    torch::Tensor aggregated_tensor = torch::empty({total_length, batch_size}, options);

    int offset = 0;
    int tmp = 0;
    for(int i=0; i<length; i++){
        tmp = (h_width_list[i] - h_n_of_gram_list[i] + 1) * h_n_of_gram_list[i];  
        output_tensors.push_back(aggregated_tensor.slice(0, offset, offset + tmp).view({batch_size, tmp}));
        offset += tmp;

        h_tensor_in_list[i] = (int64_t*)tensor_list[i].data_ptr();
        h_tensor_out_list[i] = (int64_t*)output_tensors[i].data_ptr();
    }


    thrust::device_vector<int64_t*> d_tensor_in_list = h_tensor_in_list;
    thrust::device_vector<int64_t*> d_tensor_out_list = h_tensor_out_list;
    thrust::device_vector<int32_t> d_width_list = h_width_list;
    thrust::device_vector<int32_t> d_n_of_gram_list = h_n_of_gram_list;

    int thread_num = 128; 
    dim3 grid_size(length, (batch_size+thread_num-1)/thread_num);

    ngram_fused_kernel<<<grid_size, thread_num>>>(
        thrust::raw_pointer_cast(d_tensor_in_list.data()),
        thrust::raw_pointer_cast(d_tensor_out_list.data()),
        batch_size,
        thrust::raw_pointer_cast(d_width_list.data()),
        thrust::raw_pointer_cast(d_n_of_gram_list.data()),
        length
    );

    cudaDeviceSynchronize();

    // ===============================================================================

    return output_tensors;
}

std::tuple<int64_t, int64_t> ngram_ptr_prepare(
    std::vector<torch::Tensor> tensor_list,
    std::vector<int> n_of_gram_list
)
{
    int length = tensor_list.size();
    int batch_size = tensor_list[0].sizes()[0];

    std::vector<int32_t> h_width_list(length);
    std::vector<int32_t> h_n_of_gram_list(length);

    for(int i=0;i<length;i++){
        h_n_of_gram_list[i] = n_of_gram_list[i];
        if(n_of_gram_list[i] > tensor_list[i].sizes()[1])
            h_n_of_gram_list[i] = tensor_list[i].sizes()[1];

        h_width_list[i] = tensor_list[i].sizes()[1];
    }

    int64_t width_list_ptr = copy_vector_to_GPU_int32(h_width_list);
    int64_t n_of_gram_list_ptr = copy_vector_to_GPU_int32(h_n_of_gram_list);

    return std::make_tuple(width_list_ptr, n_of_gram_list_ptr);
}


std::vector<torch::Tensor> ngram_list_fused_prepared(
    std::vector<torch::Tensor> tensor_list,
    std::vector<int> n_of_gram_list,
    int64_t width_list_ptr,
    int64_t n_of_gram_list_ptr,
    int if_kernel=1
)
{
    int length = tensor_list.size();
    int batch_size = tensor_list[0].sizes()[0];

    // ================== thrust implementation (support multi-dim) ==================

    thrust::host_vector<int64_t*> h_tensor_in_list(length);
    thrust::host_vector<int64_t*> h_tensor_out_list(length);

    int total_length = 0;
    for(int i=0;i<length;i++){
        int tmp = n_of_gram_list[i];
        if(tmp > tensor_list[i].sizes()[1])
            tmp = tensor_list[i].sizes()[1];
        total_length += (tensor_list[i].sizes()[1] - tmp + 1) * tmp;
    }

    std::vector<torch::Tensor> output_tensors;
    // torch::Tensor aggregated_tensor = torch::empty({total_length, batch_size}, at::kLong).to(at::kCUDA);
    torch::Device device(torch::kCUDA);
    auto options = torch::TensorOptions().device(device).dtype(at::kLong);
    torch::Tensor aggregated_tensor = torch::empty({total_length, batch_size}, options);

    int offset = 0;
    for(int i=0; i<length; i++){
        int tmp = n_of_gram_list[i];
        if(tmp > tensor_list[i].sizes()[1])
            tmp = tensor_list[i].sizes()[1];
        tmp = (tensor_list[i].sizes()[1] - tmp + 1) * tmp;  
        output_tensors.push_back(aggregated_tensor.slice(0, offset, offset + tmp).view({batch_size, tmp}));
        offset += tmp;

        h_tensor_in_list[i] = (int64_t*)tensor_list[i].data_ptr();
        h_tensor_out_list[i] = (int64_t*)output_tensors[i].data_ptr();
    }


    thrust::device_vector<int64_t*> d_tensor_in_list = h_tensor_in_list;
    thrust::device_vector<int64_t*> d_tensor_out_list = h_tensor_out_list;

    int thread_num = 128; 
    dim3 grid_size(length, (batch_size+thread_num-1)/thread_num);

    if(if_kernel){
        ngram_fused_kernel<<<grid_size, thread_num>>>(
            thrust::raw_pointer_cast(d_tensor_in_list.data()),
            thrust::raw_pointer_cast(d_tensor_out_list.data()),
            batch_size,
            reinterpret_cast<int32_t*>(width_list_ptr),
            reinterpret_cast<int32_t*>(n_of_gram_list_ptr),
            length
        );
    }
    cudaDeviceSynchronize();

    // ===============================================================================

    return output_tensors;
}


std::tuple<int64_t, int64_t, int32_t, std::vector<torch::Tensor>> ngram_cpu_part(
    std::vector<torch::Tensor> tensor_list,
    std::vector<int> n_of_gram_list
)
{
    int length = tensor_list.size();
    int batch_size = tensor_list[0].sizes()[0];

    // ================== thrust implementation (support multi-dim) ==================

    // thrust::host_vector<int64_t*> h_tensor_in_list(length);
    // thrust::host_vector<int64_t*> h_tensor_out_list(length);
    std::vector<int64_t*> h_tensor_in_list(length);
    std::vector<int64_t*> h_tensor_out_list(length);

    int total_length = 0;
    for(int i=0;i<length;i++){
        int tmp = n_of_gram_list[i];
        if(tmp > tensor_list[i].sizes()[1])
            tmp = tensor_list[i].sizes()[1];
        total_length += (tensor_list[i].sizes()[1] - tmp + 1) * tmp;
    }

    std::vector<torch::Tensor> output_tensors;
    // torch::Tensor aggregated_tensor = torch::empty({total_length, batch_size}, at::kLong).to(at::kCUDA);
    torch::Device device(torch::kCUDA);
    auto options = torch::TensorOptions().device(device).dtype(at::kLong);
    torch::Tensor aggregated_tensor = torch::empty({total_length, batch_size}, options);

    int offset = 0;
    for(int i=0; i<length; i++){
        int tmp = n_of_gram_list[i];
        if(tmp > tensor_list[i].sizes()[1])
            tmp = tensor_list[i].sizes()[1];
        tmp = (tensor_list[i].sizes()[1] - tmp + 1) * tmp;  
        output_tensors.push_back(aggregated_tensor.slice(0, offset, offset + tmp).view({batch_size, tmp}));
        offset += tmp;

        h_tensor_in_list[i] = (int64_t*)tensor_list[i].data_ptr();
        h_tensor_out_list[i] = (int64_t*)output_tensors[i].data_ptr();
    }

    // thrust::device_vector<int64_t*> d_tensor_in_list = h_tensor_in_list;
    // thrust::device_vector<int64_t*> d_tensor_out_list = h_tensor_out_list;
    // int64_t d_tensor_in_ptr_int64 = reinterpret_cast<int64_t>(thrust::raw_pointer_cast(d_tensor_in_list.data()));
    // int64_t d_tensor_out_ptr_int64 = reinterpret_cast<int64_t>(thrust::raw_pointer_cast(d_tensor_out_list.data()));

    int64_t d_tensor_in_ptr_int64 = copy_vector_to_GPU(h_tensor_in_list);
    int64_t d_tensor_out_ptr_int64 = copy_vector_to_GPU(h_tensor_out_list);
    // ===============================================================================

    return std::make_tuple(d_tensor_in_ptr_int64, d_tensor_out_ptr_int64, length, output_tensors);
}

std::tuple<int64_t, int64_t, int32_t, std::vector<torch::Tensor>> ngram_cpu_part_base(
    std::vector<torch::Tensor> tensor_list,
    std::vector<torch::Tensor> output_tensors,
    std::vector<int> n_of_gram_list
)
{
    int length = tensor_list.size();
    int batch_size = tensor_list[0].sizes()[0];

    // ================== thrust implementation (support multi-dim) ==================

    std::vector<int64_t*> h_tensor_in_list(length);
    std::vector<int64_t*> h_tensor_out_list(length);

    int total_length = 0;
    for(int i=0;i<length;i++){
        int tmp = n_of_gram_list[i];
        if(tmp > tensor_list[i].sizes()[1])
            tmp = tensor_list[i].sizes()[1];
        total_length += (tensor_list[i].sizes()[1] - tmp + 1) * tmp;
    }

    for(int i=0; i<length; i++){
        h_tensor_in_list[i] = (int64_t*)tensor_list[i].data_ptr();
        h_tensor_out_list[i] = (int64_t*)output_tensors[i].data_ptr();
    }

    int64_t d_tensor_in_ptr_int64 = copy_vector_to_GPU(h_tensor_in_list);
    int64_t d_tensor_out_ptr_int64 = copy_vector_to_GPU(h_tensor_out_list);

    // ===============================================================================

    return std::make_tuple(d_tensor_in_ptr_int64, d_tensor_out_ptr_int64, length, output_tensors);
}

void ngram_gpu_part(
    int64_t d_tensor_in_ptr_int64,
    int64_t d_tensor_out_ptr_int64,
    int64_t width_list_ptr,
    int64_t n_of_gram_list_ptr,
    int32_t length,
    int32_t batch_size
)
{
    int thread_num = 128; 
    dim3 grid_size(length, (batch_size+thread_num-1)/thread_num);

    ngram_fused_kernel<<<grid_size, thread_num>>>(
        // thrust::raw_pointer_cast(d_tensor_in_list.data()),
        // thrust::raw_pointer_cast(d_tensor_out_list.data()),
        reinterpret_cast<int64_t**>(d_tensor_in_ptr_int64),
        reinterpret_cast<int64_t**>(d_tensor_out_ptr_int64),
        batch_size,
        reinterpret_cast<int32_t*>(width_list_ptr),
        reinterpret_cast<int32_t*>(n_of_gram_list_ptr),
        length
    );
    cudaDeviceSynchronize();

}


// =======================================================================================================================================
// ================================================================== MapID ==============================================================
// =======================================================================================================================================
__global__
void mapid_kernel(int64_t*tensor_in, int64_t*mapping, int32_t batch_size, int32_t width, int32_t mapping_width)
{
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    if(idx<batch_size)
    {
        for(int i=0;i<width;i++)
        {   
            if(tensor_in[idx*width+i] < 0)
                tensor_in[idx*width+i] = 0;
            else
                tensor_in[idx*width+i] = mapping[tensor_in[idx*width+i] % mapping_width];
        }
    }
}

void mapid_main(
    torch::Tensor tensor,
    torch::Tensor mapping
)
{
    int batch_size = tensor.sizes()[0];
    int width = tensor.sizes()[1];
    int mapping_width = mapping.sizes()[0];

    mapid_kernel<<<(batch_size+127)/128,128>>>(
        (int64_t*)tensor.data_ptr(),
        (int64_t*)mapping.data_ptr(),
        batch_size, 
        width,
        mapping_width
    );
    cudaDeviceSynchronize();
}

__global__
void mapid_fused_kernel(int64_t**tensor_in_list, int64_t**mapping_list, int32_t batch_size, int32_t* width_list, int32_t* mapping_width_list, int32_t length)
{
    int idx=blockDim.x*blockIdx.y+threadIdx.x;
    int row_idx=blockIdx.x;

    if(idx < batch_size && row_idx < length)
    {
        int64_t* tensor_in=tensor_in_list[row_idx];
        int64_t* mapping=mapping_list[row_idx];

        int width = width_list[row_idx];
        int mapping_width = mapping_width_list[row_idx];

        for(int i=0;i<width;i++)
        {
            if(tensor_in[idx*width+i] < 0)
                tensor_in[idx*width+i] = 0;
            else
                tensor_in[idx*width+i] = mapping[tensor_in[idx*width+i] % mapping_width];
        }
    }
}


void mapid_list_fused(
    std::vector<torch::Tensor> tensor_list,
    std::vector<torch::Tensor> mapping_list
)
{
    int length = tensor_list.size();
    int batch_size = tensor_list[0].sizes()[0];

    // ================== thrust implementation (support multi-dim) ==================

    thrust::host_vector<int64_t*> h_tensor_in_list(length);
    thrust::host_vector<int64_t*> h_mapping_list(length);
    thrust::host_vector<int32_t> h_width_list(length);
    thrust::host_vector<int32_t> h_mapping_width_list(length);

    for(int i=0;i<length;i++){
        h_tensor_in_list[i] = (int64_t*)tensor_list[i].data_ptr();
        h_mapping_list[i] = (int64_t*)mapping_list[i].data_ptr();
        h_width_list[i] = tensor_list[i].sizes()[1];
        h_mapping_width_list[i] = mapping_list[i].sizes()[0];
    }

    thrust::device_vector<int64_t*> d_tensor_in_list = h_tensor_in_list;
    thrust::device_vector<int64_t*> d_mapping_list = h_mapping_list;
    thrust::device_vector<int32_t> d_width_list = h_width_list;
    thrust::device_vector<int32_t> d_mapping_width_list = h_mapping_width_list;

    int thread_num = 128; 
    dim3 grid_size(length, (batch_size+thread_num-1)/thread_num);

    mapid_fused_kernel<<<grid_size, thread_num>>>(
        thrust::raw_pointer_cast(d_tensor_in_list.data()),
        thrust::raw_pointer_cast(d_mapping_list.data()),
        batch_size,
        thrust::raw_pointer_cast(d_width_list.data()),
        thrust::raw_pointer_cast(d_mapping_width_list.data()),
        length
    );

    cudaDeviceSynchronize();
}

std::tuple<int64_t, int64_t, int64_t> mapid_ptr_prepare(
    std::vector<torch::Tensor> tensor_list,
    std::vector<torch::Tensor> mapping_list
)
{
    int length = tensor_list.size();
    std::vector<int64_t*> h_mapping_list(length);
    std::vector<int32_t> width_list(length);
    std::vector<int32_t> mapping_width_list(length);

    for(int i=0;i<length;i++){
        h_mapping_list[i] = (int64_t*)mapping_list[i].data_ptr();
        width_list[i] = tensor_list[i].sizes()[1];
        mapping_width_list[i] = mapping_list[i].sizes()[0];
    }

    int64_t mapping_list_ptr = copy_vector_to_GPU_int64_ptr(h_mapping_list);
    int64_t width_list_ptr = copy_vector_to_GPU_int32(width_list);
    int64_t mapping_width_list_ptr = copy_vector_to_GPU_int32(mapping_width_list);

    return std::make_tuple(mapping_list_ptr, width_list_ptr, mapping_width_list_ptr);
}


void mapid_list_fused_prepared(
    std::vector<torch::Tensor> tensor_list,
    int64_t mapping_list_ptr,
    int64_t width_list,
    int64_t mapping_width_list,
    int if_kernel=1
)
{
    int length = tensor_list.size();
    int batch_size = tensor_list[0].sizes()[0];

    // ================== thrust implementation (support multi-dim) ==================

    thrust::host_vector<int64_t*> h_tensor_in_list(length);

    for(int i=0;i<length;i++){
        h_tensor_in_list[i] = (int64_t*)tensor_list[i].data_ptr();
    }

    thrust::device_vector<int64_t*> d_tensor_in_list = h_tensor_in_list;

    int thread_num = 128; 
    dim3 grid_size(length, (batch_size+thread_num-1)/thread_num);

    if(if_kernel){
        mapid_fused_kernel<<<grid_size, thread_num>>>(
            thrust::raw_pointer_cast(d_tensor_in_list.data()),
            reinterpret_cast<int64_t**>(mapping_list_ptr),
            batch_size,
            reinterpret_cast<int32_t*>(width_list),
            reinterpret_cast<int32_t*>(mapping_width_list),
            length
        );
    }

    cudaDeviceSynchronize();
}

std::tuple<int64_t, int32_t> mapid_cpu_part(
    std::vector<torch::Tensor> tensor_list
)
{
    int length = tensor_list.size();
    int batch_size = tensor_list[0].sizes()[0];

    // ================== thrust implementation (support multi-dim) ==================

    thrust::host_vector<int64_t*> h_tensor_in_list(length);

    for(int i=0;i<length;i++){
        h_tensor_in_list[i] = (int64_t*)tensor_list[i].data_ptr();
    }

    thrust::device_vector<int64_t*> d_tensor_in_list = h_tensor_in_list;

    int64_t d_tensor_in_ptr_int64 = reinterpret_cast<int64_t>(thrust::raw_pointer_cast(d_tensor_in_list.data()));
    
    return std::make_tuple(d_tensor_in_ptr_int64, length);
}

std::tuple<int64_t, int32_t> mapid_cpu_part_base(
    std::vector<torch::Tensor> tensor_list
)
{
    int length = tensor_list.size();
    int batch_size = tensor_list[0].sizes()[0];

    // ================== thrust implementation (support multi-dim) ==================

    std::vector<int64_t*> h_tensor_in_list(length);

    for(int i=0;i<length;i++){
        h_tensor_in_list[i] = (int64_t*)tensor_list[i].data_ptr();
    }

    int64_t d_tensor_in_ptr_int64 = copy_vector_to_GPU(h_tensor_in_list);
    
    return std::make_tuple(d_tensor_in_ptr_int64, length);
}


void mapid_gpu_part(
    int64_t d_tensor_in_ptr_int64,
    int64_t mapping_list_ptr,
    int64_t width_list,
    int64_t mapping_width_list,
    int length,
    int batch_size
){
    int thread_num = 128; 
    dim3 grid_size(length, (batch_size+thread_num-1)/thread_num);

    mapid_fused_kernel<<<grid_size, thread_num>>>(
        reinterpret_cast<int64_t**>(d_tensor_in_ptr_int64),
        reinterpret_cast<int64_t**>(mapping_list_ptr),
        batch_size,
        reinterpret_cast<int32_t*>(width_list),
        reinterpret_cast<int32_t*>(mapping_width_list),
        length
    );

    cudaDeviceSynchronize();
}