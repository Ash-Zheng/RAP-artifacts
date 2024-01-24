#include<cuda.h>
#include<stdio.h>
#include <functional>

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/ATen.h>

using namespace std;
#include <chrono>
using namespace std::chrono;

using uint128_t = unsigned __int128;
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

// __global__
// void sigrid_hash(uint64_t*tensor_in, int tensor_size,const int64_t salt,const int64_t maxValue,const uint64_t multiplier,const int shift)
// {
//     int idx=blockDim.x*blockIdx.x+threadIdx.x;
//     if(idx<tensor_size)
//     {
//         uint64_t result;
//         uint64_t upper = hashfunction(salt);
//         uint64_t lower = hashfunction(twang_mix64(tensor_in[idx]));
//         int64_t hashed = hash_128_to_64(upper,lower);
        
//         if (maxValue > 1) {
//             int64_t sign = hashed >> (64 - 1);
//             __int128 left=((__int128)(multiplier));
//             uint64_t right=sign ^ hashed;
//             int64_t q = sign^((left * right)>>(64 + shift));    
//             result = hashed - q * maxValue;
//         }
//         else
//         {
//             result=hashed;
//         }
//         tensor_in[idx]=result;
//     }
// }

__global__
void sigrid_hash(uint64_t*tensor_in, int tensor_size,const int64_t salt,const int64_t maxValue,const uint64_t multiplier,const int shift)
{
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
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
void sigrid_hash_fused(uint64_t*tensor_in, int tensor_size,const int64_t salt,const int64_t maxValue,const uint64_t multiplier,const int shift)
{
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    if(idx<tensor_size)
    {
        for(int j=0; j<15; j++){
            uint64_t result;
            uint64_t upper = hashfunction(salt);
            uint64_t lower = hashfunction(twang_mix64(tensor_in[idx]));
            int64_t hashed = hash_128_to_64(upper,lower);
            
            if (maxValue > 1) {
                int64_t sign = hashed >> (64 - 1);
                __int128 left=((__int128)(multiplier));
                uint64_t right=sign ^ hashed;
                int64_t q = sign^((left * right)>>(64 + shift));    
                result = hashed - q * maxValue;
            }
            else
            {
                result=hashed;
            }
            tensor_in[idx]=result;
        }
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


    // (const int64_t*)index.data_ptr()
    // tensor.accessor<uint64_t,1>()
    // sigrid_hash<<<(tensor_size+31)/32,32>>>(
    
    int thread_num = 32; // 32 512
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

void sigrid_hash_main_fused(
    torch::Tensor tensor,
    int64_t salt,
    int64_t maxValue
)
{
    uint64_t multiplier_;
    int shift_;
    computeMultiperAndShift(maxValue, 63, multiplier_, shift_);    
    
    int tensor_size = tensor.numel();


    // (const int64_t*)index.data_ptr()
    // tensor.accessor<uint64_t,1>()
    // sigrid_hash_fused<<<(tensor_size+31)/32,32>>>(
    
    int thread_num = 32; // 32 512
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



// =======================================================================================================================================
// ======================================================== Bucketize ====================================================================
// =======================================================================================================================================
__global__
void bucketize(float*tensor_in,int*tensor_out,int tensor_size, float*borders,int borders_size)
{
    int idx=blockDim.x*blockIdx.x+threadIdx.x;
    if(idx<tensor_size)
    {
        float data=tensor_in[idx];
        //lower_bound
        int index=0;
        int i;
        for(i=0;i<borders_size;++i)
        {
            if(data<=borders[i])
            {
                break;
            }
        }
        index=i;
        int result;
        //index
        if (index >= borders_size - 1) 
        {
            result=index;
        }
        result = data < borders[index + 1] ? index : index + 1;
        tensor_out[idx]=result;
    }
}


torch::Tensor bucketize_main(
    torch::Tensor tensor,
    torch::Tensor borders
    // const std::vector<int>&  borders
)
{
    // torch::Tensor output_tensor = torch::zeros(tensor.shape);
    torch::Tensor output_tensor = torch::zeros({tensor.sizes()}, at::kInt).to(at::kCUDA);
    int tensor_size = tensor.numel();

    bucketize<<<(tensor_size+31)/32,32>>>(
        (float*)tensor.data_ptr(),
        (int*)output_tensor.data_ptr(),
        tensor_size, 
        (float*)borders.data_ptr(), 
        borders.numel()
        // borders.size()
    );
    return output_tensor;
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
        int* d_data_ptr = (int*)d_data_ptrs[row_idx];
        int32_t data = d_data_ptr[idx];
        tensor_ptr[row_idx*batch_size+idx] =  __int2float_rd(data);
    }
}


torch::Tensor fill_null_float(
    int64_t data_ptr,
    int32_t length
)
{
    torch::Tensor output_tensor = torch::zeros({length}, at::kFloat).to(at::kCUDA);
    int* d_data_ptr = (int*)data_ptr;
 
    fill_null_float_kernel<<<(length+31)/32,32>>>(
    // fill_null_float_kernel<<<(length+1024)/1024,1024>>>(
        d_data_ptr,
        (float*)output_tensor.data_ptr(),
        length
    );
    
    return output_tensor;
}

std::vector<torch::Tensor> fill_null_float_fused(
// torch::Tensor fill_null_float_fused(
    torch::Tensor data_ptrs,
    int32_t batch_size
)
{
    int length = data_ptrs.size(0);
    std::vector<torch::Tensor> output_tensors;
    // printf("length: %d, batch_size: %d\n", length, batch_size);
    // auto aggregated_tensor = torch::zeros({length, batch_size}, at::kFloat).to(at::kCUDA);
    torch::Tensor aggregated_tensor = torch::empty({length, batch_size}, at::kFloat).to(at::kCUDA);
    
    dim3 block_size(32);
    dim3 grid_size(length, (batch_size+31)/32); // gridDim.x = nTable, gridDim.y = batch_size/32, blockDim.x = 32.
 
    fill_null_float_kernel_fused<<<grid_size, block_size>>>(
        (int64_t*)data_ptrs.data_ptr(),
        (float*)aggregated_tensor.data_ptr(),
        length,
        batch_size
    );

    for(int i=0; i<length; i++)
        output_tensors.push_back(aggregated_tensor[i]);

    return output_tensors;
}


void fill_null_float_inplace(
    torch::Tensor output_tensor,
    int64_t data_ptr,
    int32_t length
)
{
    int* d_data_ptr = (int*)data_ptr;
 
    fill_null_float_kernel<<<(length+31)/32,32>>>(
    // fill_null_float_kernel<<<(length+1024)/1024,1024>>>(
        d_data_ptr,
        (float*)output_tensor.data_ptr(),
        length
    );
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

torch::Tensor fill_null_int64(
    int64_t data_ptr,
    int32_t length
)
{
    torch::Tensor output_tensor = torch::zeros({length}, at::kLong).to(at::kCUDA);
    int* d_data_ptr = (int*)data_ptr;
 
    fill_null_int64_kernel<<<(length+31)/32,32>>>(
        d_data_ptr,
        (int64_t*)output_tensor.data_ptr(),
        length
    );
    
    return output_tensor;
}


std::vector<torch::Tensor> fill_null_int64_fused(
    torch::Tensor data_ptrs,
    int32_t batch_size
)
{
    int length = data_ptrs.size(0);
    std::vector<torch::Tensor> output_tensors;
   
    auto aggregated_tensor = torch::empty({length, batch_size}, at::kLong).to(at::kCUDA);
    

    dim3 block_size(32);
    dim3 grid_size(length, (batch_size+31)/32); // gridDim.x = nTable, gridDim.y = batch_size/32, blockDim.x = 32.
 
    fill_null_int64_kernel_fused<<<grid_size, block_size>>>(
        (int64_t*)data_ptrs.data_ptr(),
        (int64_t*)aggregated_tensor.data_ptr(),
        length,
        batch_size
    );

    for(int i=0; i<length; i++)
        output_tensors.push_back(aggregated_tensor[i]);
    
    return output_tensors;
}


void fill_null_int64_inplace(
    torch::Tensor output_tensor,
    int64_t data_ptr,
    int32_t length
)
{
    int* d_data_ptr = (int*)data_ptr;
 
    fill_null_int64_kernel<<<(length+31)/32,32>>>(
        d_data_ptr,
        (int64_t*)output_tensor.data_ptr(),
        length
    );
}


