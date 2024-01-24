import cudf
import torch
import torch.utils.dlpack
from torch.utils.cpp_extension import load

# cuda_preprocess = load(name="gpu_operators", sources=[
# "/workspace/torchrec/examples/dlrm/cuda_operators/cuda_wrap.cpp", 
# "/workspace/torchrec/examples/dlrm/cuda_operators/gpu_operators.cu", 
# ], verbose=False, build_directory="/workspace/torchrec/examples/dlrm/cuda_operators/build")

cuda_preprocess = load(name="gpu_operators", sources=[
"/workspace/RAP/cuda_operators/cuda_wrap.cpp", 
"/workspace/RAP/cuda_operators/gpu_operators.cu", 
], verbose=True)

cuda_preprocess.init_cuda(0)
border = torch.tensor([1,2,3]).to(0)

batch_size = 4096
# file_name = "/workspace/RAP/generated_data/first_{}.parquet".format(batch_size)
file_name = "/workspace/RAP/sample_data.parquet".format(batch_size)
df = cudf.read_parquet(file_name)
keys = df.keys().to_list()


label_tensor = cuda_preprocess.fill_null_float(df[keys[0]].data.ptr, batch_size)
dense_tensors = []
sparse_tensors = []
for i in range(1,14):
    dense_tensors.append(cuda_preprocess.fill_null_float(df[keys[i]].data.ptr, batch_size)) 
for i in range(14,len(keys)):
    sparse_tensors.append(cuda_preprocess.fill_null_int64(df[keys[i]].data.ptr, batch_size))
for sparse_tensor in sparse_tensors:
    cuda_preprocess.sigrid_hash(sparse_tensor, 0, 65536)
new_sparse = cuda_preprocess.bucketize(dense_tensors[0], border)

print(sparse_tensors[0].device)