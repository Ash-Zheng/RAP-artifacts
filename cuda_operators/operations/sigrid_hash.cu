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
void sigrid_hash(uint64_t*tensor_in,uint64_t*tensor_out,int tensor_size,const int64_t salt,const int64_t maxValue,const uint64_t multiplier,const int shift)
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
            __int128 left=((__int128)(multiplier));
            uint64_t right=sign ^ hashed;
            int64_t q = sign^((left * right)>>(64 + shift));    
            result = hashed - q * maxValue;
        }
        else
        {
            result=hashed;
        }
        tensor_out[idx]=result;
    }
}


Tensor sigrid_hash_main(
    Tensor tensor,
    int64_t salt,
    int64_t maxValue,
    int kPrecision
)
{
    uint64_t multiplier_;
    int shift_;
    computeMultiperAndShift(maxValue, kPrecision, multiplier_, shift_);    
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

class StdHasher {
public:
 // The standard requires all explicit and partial specializations of std::hash
 // supplied by either the standard library or by users to be default
 // constructible.
 template <typename T>
 size_t operator()(const T& t) const noexcept(noexcept(std::hash<T>()(t))) {
   return std::hash<T>()(t);
 }
};

constexpr uint64_t twang_mix64_cpu(uint64_t key) noexcept {
    key = (~key) + (key << 21); // key *= (1 << 21) - 1; key -= 1;
    key = key ^ (key >> 24);
    key = key + (key << 3) + (key << 8); // key *= 1 + (1 << 3) + (1 << 8)
    key = key ^ (key >> 14);
    key = key + (key << 2) + (key << 4); // key *= 1 + (1 << 2) + (1 << 4)
    key = key ^ (key >> 28);
    key = key + (key << 31); // key *= 1 + (1 << 31)
    return key;
  }

  constexpr uint32_t twang_32from64(uint64_t key) noexcept {
    key = (~key) + (key << 18);
    key = key ^ (key >> 31);
    key = key * 21;
    key = key ^ (key >> 11);
    key = key + (key << 6);
    key = key ^ (key >> 22);
    return static_cast<uint32_t>(key);
  }

  constexpr uint64_t hash_128_to_64_cpu(
    const uint64_t upper, const uint64_t lower) noexcept {
  // Murmur-inspired hashing.
  const uint64_t kMul = 0x9ddfea08eb382d69ULL;
  uint64_t a = (lower ^ upper) * kMul;
  a ^= (a >> 47);
  uint64_t b = (upper ^ a) * kMul;
  b ^= (b >> 47);
  b *= kMul;
  return b;
}
using Byte = uint8_t;

constexpr uint64_t c_offsetBasis = 14695981039346656037;
constexpr uint64_t c_FNVPrime = 1099511628211;

uint64_t fnv1a(uint64_t value) 
{
    Byte*bs=(Byte*)(&value);
    int len=8;
    uint64_t h = c_offsetBasis;
    for (size_t i = 0; i < len; ++i) {
        h = h ^ bs[i];
        h = h * c_FNVPrime; 
    }
    return h;
}
  namespace detail {
    using c_array_size_t = size_t[];
    } // namespace detail
    
    // Never used, but gcc demands it.
    template <class Hasher>
    inline size_t hash_combine_generic(const Hasher&) noexcept {
      return 0;
    }
    
    template <class Hasher, typename T, typename... Ts>
    size_t hash_combine_generic(
        const Hasher& h,
        const T& t,
        const Ts&... ts) noexcept(noexcept(detail::c_array_size_t{
        h(t), h(ts)...})) {
      //size_t seed = h(t);
      size_t seed = fnv1a(t);
      //printf("hash:%ld %ld\n",t, seed);
      if (sizeof...(ts) == 0) {
        return seed;
      }
      size_t remainder = hash_combine_generic(h, ts...);
      if /* constexpr */ (sizeof(size_t) == sizeof(uint32_t)) {
        //printf("get789 ");
        return twang_32from64((uint64_t(seed) << 32) | remainder);
      } else {
        //printf("get456 ");
        //printf("input:%ld %ld\n",seed, remainder);
        //printf("128_to_64:%ld",hash_128_to_64_cpu(seed, remainder));
        return static_cast<size_t>(hash_128_to_64_cpu(seed, remainder));
      }
    }
    
    template <typename T, typename... Ts>
    size_t hash_combine_cpu(const T& t, const Ts&... ts) noexcept(
        noexcept(hash_combine_generic(StdHasher{}, t, ts...))) {
      return hash_combine_generic(StdHasher{}, t, ts...);
    }

template <typename TInput>
inline int64_t computeSigridHash_CPU(
    const TInput& input,
    const int64_t salt,
    const int64_t maxValue,
    const uint64_t multiplier,
    const int shift) {
  if (maxValue == 1) {
    return 0;
  }
  int64_t hashed = hash_combine_cpu(salt, twang_mix64_cpu(input));
  //printf("hasded:%lx ",hashed);
  if (maxValue > 1) {
    int64_t sign = hashed >> (64 - 1);
    __int128 left=(uint128_t)(multiplier);
    uint64_t right = (sign ^ hashed);

    int64_t q = sign ^((left * right) >>(64 + shift));
    //printf("q:%lx \n",sign);

    int64_t output = hashed - q * maxValue;
    return output;
  }
  return hashed;
}

int main()
{
    //4096 8192 16384(*20)
    int size_list[]={4096*200,8192*200,16384*200};
    int64_t salt=123;
    int64_t maxValue=456;
    const static int kPrecision = 63;

    uint64_t multiplier_;
    int shift_;
    computeMultiperAndShift(maxValue, kPrecision,multiplier_, shift_);

    uint64_t*tensor_in_h;
    uint64_t*tensor_out_h;
    uint64_t*tensor_out_cpu;
    uint64_t*tensor_in_d;
    uint64_t*tensor_out_d;
    int tensor_size;
    
    for(int idx=0;idx<3;++idx)
    {
    tensor_size=size_list[idx];
    tensor_in_h=(uint64_t*)malloc(sizeof(uint64_t)*tensor_size);
    tensor_out_h=(uint64_t*)malloc(sizeof(uint64_t)*tensor_size);
    tensor_out_cpu=(uint64_t*)malloc(sizeof(uint64_t)*tensor_size);
    for(int i=0;i<tensor_size;++i)
    {
        tensor_in_h[i]=i;
    }


    auto t1_tmp=std::chrono::steady_clock::now();
    for(int iter=0;iter<100;++iter)
    {
        for(int i=0;i<tensor_size;++i)
        {
            tensor_out_cpu[i]=computeSigridHash_CPU<uint64_t>(tensor_in_h[i],salt,maxValue,multiplier_,shift_);
        }
    }
    auto t2_tmp=std::chrono::steady_clock::now();
    double cpu_time=std::chrono::duration<double,std::milli>(t2_tmp-t1_tmp).count();




    cudaMalloc((void**)&tensor_in_d,sizeof(uint64_t)*tensor_size);
    cudaMalloc((void**)&tensor_out_d,sizeof(uint64_t)*tensor_size);
    cudaMemset(tensor_out_d,0,sizeof(uint64_t)*tensor_size);
    cudaMemcpy(tensor_in_d,tensor_in_h,sizeof(uint64_t)*tensor_size,cudaMemcpyHostToDevice);
    // int64_t*flag;
    // cudaMallocManaged(&flag,sizeof(int64_t)*10);
    // for(int i=0;i<10;++i)flag[i]=0;


    auto t3_tmp=std::chrono::steady_clock::now();
    for(int iter=0;iter<100;++iter)
    {
        sigrid_hash<<<(tensor_size+31)/32,32>>>(tensor_in_d,tensor_out_d,tensor_size,salt,maxValue,multiplier_,shift_);
    }
    cudaDeviceSynchronize();
    auto t4_tmp=std::chrono::steady_clock::now();
    double cuda_time=std::chrono::duration<double,std::milli>(t4_tmp-t3_tmp).count();

    printf("tensor_size:%d CPU:%f GPU:%f\n",tensor_size,cpu_time/100,cuda_time/100);



    cudaMemcpy(tensor_out_h,tensor_out_d,sizeof(uint64_t)*tensor_size,cudaMemcpyDeviceToHost);
    
    free(tensor_in_h);
    free(tensor_out_h);
    free(tensor_out_cpu);
    cudaFree(tensor_in_d);
    cudaFree(tensor_out_d);
    }
}