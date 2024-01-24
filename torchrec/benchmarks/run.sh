# CUDA_VISIBLE_DEVICES=4,5,6,7 python my_ebc_benchmark.py --mode=my_benchmark


# CUDA_VISIBLE_DEVICES=4,5,6,7 python my_ebc_benchmark.py --mode=my_benchmark --type=embeddingbag
# CUDA_VISIBLE_DEVICES=4,5,6,7 python my_ebc_benchmark.py --mode=my_benchmark --type=fused_table

# nsys profile -t cuda,nvtx,cublas,cusparse,cudnn --gpu-metrics-device=all --capture-range cudaProfilerApi -o embeddingbag_train_1 -f true -x true python my_ebc_benchmark.py --mode=my_benchmark --type=embeddingbag --pooling_factor=1
# nsys profile -t cuda,nvtx,cublas,cusparse,cudnn --gpu-metrics-device=all --capture-range cudaProfilerApi -o fused_table_train_1 -f true -x true python my_ebc_benchmark.py --mode=my_benchmark --type=fused_table --pooling_factor=1

# nsys profile -t cuda,nvtx,cublas,cusparse,cudnn --gpu-metrics-device=all --capture-range cudaProfilerApi -o embeddingbag_train_64 -f true -x true python my_ebc_benchmark.py --mode=my_benchmark --type=embeddingbag --pooling_factor=64
# nsys profile -t cuda,nvtx,cublas,cusparse,cudnn --gpu-metrics-device=all --capture-range cudaProfilerApi -o fused_table_train_64 -f true -x true python my_ebc_benchmark.py --mode=my_benchmark --type=fused_table --pooling_factor=64


# nsys profile -t cuda,nvtx,cublas,cusparse,cudnn --gpu-metrics-device=all --capture-range cudaProfilerApi -o embeddingbag_inference_1 -f true -x true python my_ebc_benchmark.py --mode=my_benchmark --type=embeddingbag --pooling_factor=1
# nsys profile -t cuda,nvtx,cublas,cusparse,cudnn --gpu-metrics-device=all --capture-range cudaProfilerApi -o fused_table_inference_1 -f true -x true python my_ebc_benchmark.py --mode=my_benchmark --type=fused_table --pooling_factor=1

# nsys profile -t cuda,nvtx,cublas,cusparse,cudnn --gpu-metrics-device=all --capture-range cudaProfilerApi -o embeddingbag_inference_64 -f true -x true python my_ebc_benchmark.py --mode=my_benchmark --type=embeddingbag --pooling_factor=64
# nsys profile -t cuda,nvtx,cublas,cusparse,cudnn --gpu-metrics-device=all --capture-range cudaProfilerApi -o fused_table_inference_64 -f true -x true python my_ebc_benchmark.py --mode=my_benchmark --type=fused_table --pooling_factor=64


nsys profile -t cuda,nvtx,cublas,cusparse,cudnn --gpu-metrics-device=all --capture-range cudaProfilerApi -o embeddingbag_train_1 -f true -x true python my_ebc_benchmark.py --mode=my_benchmark --type=embeddingbag --pooling_factor=1

nsys profile -t cuda,nvtx,cublas,cusparse,cudnn --gpu-metrics-device=all --capture-range cudaProfilerApi -o embeddingbag_train_2 -f true -x true python my_ebc_benchmark.py --mode=my_benchmark --type=embeddingbag --pooling_factor=2

nsys profile -t cuda,nvtx,cublas,cusparse,cudnn --gpu-metrics-device=all --capture-range cudaProfilerApi -o embeddingbag_train_4 -f true -x true python my_ebc_benchmark.py --mode=my_benchmark --type=embeddingbag --pooling_factor=4

nsys profile -t cuda,nvtx,cublas,cusparse,cudnn --gpu-metrics-device=all --capture-range cudaProfilerApi -o embeddingbag_train_8 -f true -x true python my_ebc_benchmark.py --mode=my_benchmark --type=embeddingbag --pooling_factor=8

nsys profile -t cuda,nvtx,cublas,cusparse,cudnn --gpu-metrics-device=all --capture-range cudaProfilerApi -o embeddingbag_train_16 -f true -x true python my_ebc_benchmark.py --mode=my_benchmark --type=embeddingbag --pooling_factor=16

nsys profile -t cuda,nvtx,cublas,cusparse,cudnn --gpu-metrics-device=all --capture-range cudaProfilerApi -o embeddingbag_train_32 -f true -x true python my_ebc_benchmark.py --mode=my_benchmark --type=embeddingbag --pooling_factor=32

nsys profile -t cuda,nvtx,cublas,cusparse,cudnn --gpu-metrics-device=all --capture-range cudaProfilerApi -o embeddingbag_train_64 -f true -x true python my_ebc_benchmark.py --mode=my_benchmark --type=embeddingbag --pooling_factor=64

# =========================================================================================================================================================================
# =========================================================================================================================================================================
# =========================================================================================================================================================================

nsys profile -t cuda,nvtx,cublas,cusparse,cudnn --gpu-metrics-device=all --capture-range cudaProfilerApi -o fused_table_train_1 -f true -x true python my_ebc_benchmark.py --mode=my_benchmark --type=fused_table --pooling_factor=1

nsys profile -t cuda,nvtx,cublas,cusparse,cudnn --gpu-metrics-device=all --capture-range cudaProfilerApi -o fused_table_train_2 -f true -x true python my_ebc_benchmark.py --mode=my_benchmark --type=fused_table --pooling_factor=2

nsys profile -t cuda,nvtx,cublas,cusparse,cudnn --gpu-metrics-device=all --capture-range cudaProfilerApi -o fused_table_train_4 -f true -x true python my_ebc_benchmark.py --mode=my_benchmark --type=fused_table --pooling_factor=4

nsys profile -t cuda,nvtx,cublas,cusparse,cudnn --gpu-metrics-device=all --capture-range cudaProfilerApi -o fused_table_train_8 -f true -x true python my_ebc_benchmark.py --mode=my_benchmark --type=fused_table --pooling_factor=8

nsys profile -t cuda,nvtx,cublas,cusparse,cudnn --gpu-metrics-device=all --capture-range cudaProfilerApi -o fused_table_train_16 -f true -x true python my_ebc_benchmark.py --mode=my_benchmark --type=fused_table --pooling_factor=16

nsys profile -t cuda,nvtx,cublas,cusparse,cudnn --gpu-metrics-device=all --capture-range cudaProfilerApi -o fused_table_train_32 -f true -x true python my_ebc_benchmark.py --mode=my_benchmark --type=fused_table --pooling_factor=32

nsys profile -t cuda,nvtx,cublas,cusparse,cudnn --gpu-metrics-device=all --capture-range cudaProfilerApi -o fused_table_train_64 -f true -x true python my_ebc_benchmark.py --mode=my_benchmark --type=fused_table --pooling_factor=64