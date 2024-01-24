#include<cuda.h>
#include<stdio.h>
#include <chrono>
using namespace std::chrono;

__global__
void bucketize(float*tensor_in,int*tensor_out,int tensor_size,float*borders,int borders_size)
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
void bucketize_cpu(float*tensor_in,int*tensor_out,int tensor_size,float*borders,int borders_size)
{
    for(int idx=0;idx<tensor_size;++idx)
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
 
int main()
{
    float*tensor_in_h,*borders_h;
    int*tensor_out_h;
    float*tensor_in_d,*borders_d;
    int*tensor_out_d;
    int tensor_size,borders_size;
    int size_list[]={4096000,8192000,16384000};
    for(int idx=0;idx<3;++idx)
    {
    tensor_size=size_list[idx];
    borders_size=32;
    tensor_in_h=(float*)malloc(sizeof(float)*tensor_size);
    tensor_out_h=(int*)malloc(sizeof(int)*tensor_size);
    borders_h=(float*)malloc(sizeof(float)*borders_size);
    for(int i=0;i<tensor_size;++i)
    {
        tensor_in_h[i]=i;
    }
    for(int i=0;i<borders_size;++i)
    {
        borders_h[i]=tensor_size/borders_size*i;
    }
    //4096 8192 16384
    cudaMalloc((void**)&tensor_in_d,sizeof(float)*tensor_size);
    cudaMalloc((void**)&borders_d,sizeof(float)*borders_size);
    cudaMalloc((void**)&tensor_out_d,sizeof(int)*tensor_size);
    cudaMemcpy(tensor_in_d,tensor_in_h,sizeof(float)*tensor_size,cudaMemcpyHostToDevice);
    cudaMemcpy(borders_d,borders_h,sizeof(float)*borders_size,cudaMemcpyHostToDevice);


    auto t1_tmp=std::chrono::steady_clock::now();
    for(int iter=0;iter<100;++iter)
    {
    bucketize<<<(tensor_size+31)/32,32>>>(tensor_in_d,tensor_out_d,tensor_size,borders_d,borders_size);
    }
    cudaDeviceSynchronize();
    auto t2_tmp=std::chrono::steady_clock::now();
    double cuda_time=std::chrono::duration<double,std::milli>(t2_tmp-t1_tmp).count();

    auto t3_tmp=std::chrono::steady_clock::now();
    for(int iter=0;iter<100;++iter)
    {
    bucketize_cpu(tensor_in_h,tensor_out_h,tensor_size,borders_h,borders_size);
    }
    auto t4_tmp=std::chrono::steady_clock::now();
    double cpu_time=std::chrono::duration<double,std::milli>(t4_tmp-t3_tmp).count();
    

    printf("tensor_size:%d CPU:%f GPU:%f\n",tensor_size,cpu_time/100,cuda_time/100);
    cudaMemcpy(tensor_out_h,tensor_out_d,sizeof(int)*tensor_size,cudaMemcpyDeviceToHost);
    free(tensor_in_h);
    free(tensor_out_h);
    free(borders_h);
    cudaFree(tensor_in_d);
    cudaFree(tensor_out_d);
    cudaFree(borders_d);
    }
}