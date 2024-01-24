# RAP: Resource-aware Automated GPU Sharing for Multi-GPU Recommendation Model Training and Input Preprocessing

This repo is for the ASPLOS 2024 artifacts evaluation.

## Hardware Requirments
To reproduce the results in the paper, we recommend to use a machine with 8 NVIDIA A100 GPUs (e.g. AWS p4d.24xlarge instance).

## Environment Setup
```
docker pull zhengwang0122/zheng_torchrec_cudf:latest
docker run --gpus all --name=RAP_exp -it -v <dir_of_RAP>:/workspace/RAP --ipc=host --cap-add=SYS_ADMIN zhengwang0122/zheng_torchrec_cudf:latest /bin/bash

apt-get update
apt-get install graphviz
apt-get install git 
pip install graphviz
pip install jinja2
pip install xgboost
python -m pip install gurobipy
pip install torchdata


# Set up gurobi solver: please apply for a free academic license from: https://www.gurobi.com/academia/academic-program-and-licenses/
# And put the license file (gurobi.lic) in the directory: /opt/gurobi
mkdir /opt/gurobi
cp gurobi.lic /opt/gurobi


# install torcharrow
cd /workspace/RAP/torcharrow
apt install -y g++ cmake ccache ninja-build checkinstall \
    libssl-dev libboost-all-dev libdouble-conversion-dev libgoogle-glog-dev \
    libgflags-dev libevent-dev libre2-dev libfl-dev libbison-dev
# Build and install folly and fmt
scripts/setup-ubuntu.sh
python setup.py install

# install torchrec
cd torchrec/
python setup.py install develop 
cd examples/test_installation/ 
./test.sh # this is used for testing the installation of torchrec
```

### RAP Code Generation
Run RAP to generate the end-to-end training code. The generated code is located in `/workspace/RAP-artifacts/RAP_end_to_end/combined_code`
```
cd /workspace/RAP/RAP_end_to_end
./RAP_code_gen.sh
```

### Figure-9
Get the TorchArrow results (the result of TorchArrow in Figure-9 will be stored in the directory: `/workspace/RAP/baseline_end_to_end/CPU_based_baseline/result`):
```
cd /workspace/RAP/baseline_end_to_end/CPU_based_baseline
./run_cpu_baseline.sh
```

Get the CUDA stream results (the result of CUDA stream in Figure-9 will be stored in the directory: `/workspace/RAP/baseline_end_to_end/CUDA_stream/result`):
```
cd /workspace/RAP/baseline_end_to_end/CUDA_stream
./run_stream.sh
```

Get the MPS results (the result of MPS in Figure-9 will be stored in the directory: `/workspace/RAP/baseline_end_to_end/MPS/result`):
```
cd /workspace/RAP/baseline_end_to_end/MPS
./run_mps.sh
```

Get the RAP results
```
cd /workspace/RAP/baseline_end_to_end/RAP
./run_RAP.sh
```


### Figure-10
Get the Sequential results (the result of Sequential in Figure-10 will be stored in the directory: `/workspace/RAP/breakdown_study/MPS_and_sequential/result` with the prefix "Sequential"):
```
cd /workspace/RAP/breakdown_study/MPS_and_sequential
./get_sequential_result.sh
```

Get the MPS results (the result of MPS in Figure-10 will be stored in the directory: `/workspace/RAP/breakdown_study/MPS_and_sequential/result` with the prefix "MPS"):
```
cd /workspace/RAP/breakdown_study/MPS_and_sequential
./get_MPS_result.sh
```

Get the RAP w/o Mapping results (the result of RAP w/o Mapping in Figure-10 will be stored in the directory: `/workspace/RAP/breakdown_study/no_mapping/result`):
```
cd /workspace/RAP/breakdown_study/no_mapping
./get_no_fusion_result.sh
```

Get the RAP w/o Fusion results (the result of RAP w/o Fusion in Figure-10 will be stored in the directory: `/workspace/RAP/breakdown_study/no_fusion/result`):
```
cd /workspace/RAP/breakdown_study/no_fusion
./get_no_fusion_result.sh
```

Get the ideal result (the result of ideal in Figure-10 will be stored in the directory: `/workspace/RAP/breakdown_study/ideal/result`):
```
cd /workspace/RAP/breakdown_study/ideal
./get_ideal_result.sh
```

Get the RAP results (the result of RAP in Figure-10 will be stored in the directory: `/workspace/RAP/breakdown_study/RAP/result`):
```
cd /workspace/RAP/breakdown_study/RAP
./run_RAP.sh
```


## Reference
We use some codes from the following repos:
* [TorchRec](https://github.com/pytorch/torchrec)
* [TorchArrow](https://github.com/pytorch/torcharrow)
