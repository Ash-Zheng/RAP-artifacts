export LOCAL_WORLD_SIZE=4
export WORLD_SIZE=4
export GLOBAL_BATCH_SIZE=32768 # 8192*4
export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_VISIBLE_DEVICES=2,3,4,5


python ddp_train.py --nDev=4
python my_sharding.py --nDev=4
python only_emb.py --nDev=4
python BatchedFusedEmbedding_test.py --nDev=4

nsys profile -t cuda,nvtx,cublas,cusparse,cudnn --gpu-metrics-device=all --capture-range cudaProfilerApi -o torchrec_terabyte -f true -x true python nsight_profiling.py --nDev=4
nsys profile -t cuda,nvtx,cublas,cusparse,cudnn --gpu-metrics-device=4,5,6,7 --capture-range cudaProfilerApi -o torchrec_terabyte_full -f true -x true python nsight_profiling.py --nDev=4
nsys profile -t cuda,nvtx,cublas,cusparse,cudnn --gpu-metrics-device=4,5,6,7 --capture-range cudaProfilerApi -o torchrec_only_embedding_sharding -f true -x true python only_emb.py --nDev=4
