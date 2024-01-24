mkdir result
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
nvidia-smi -i 0,1,2,3,4,5,6,7 -c EXCLUSIVE_PROCESS

nvidia-cuda-mps-control -d
ps -ef | grep mps
echo "Finish enabling MPS"


python data_prepare.py --preprocessing_plan=0 --nDev=4 --batch_size=4096
python /workspace/RAP/RAP_end_to_end/combined_code/GPU_4_plan_0_all.py --nDev=4 --batch_size=4096 --preprocessing_plan=0 --num_embeddings_per_feature="1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194,27,14992,5461306,10,5652,2173,4,7046547,18,15,286181,105,142572"  --over_arch_layer_sizes="512,512,256,1"

python data_prepare.py --preprocessing_plan=1 --nDev=4 --batch_size=4096
python /workspace/RAP/RAP_end_to_end/combined_code/GPU_4_plan_1_all.py --nDev=4 --batch_size=4096 --preprocessing_plan=1

# =====================================================================================================
python data_prepare.py --preprocessing_plan=0 --nDev=4 --batch_size=8192
python /workspace/RAP/RAP_end_to_end/combined_code/GPU_4_plan_0_all.py --nDev=4 --batch_size=8192 --preprocessing_plan=0 --num_embeddings_per_feature="1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194,27,14992,5461306,10,5652,2173,4,7046547,18,15,286181,105,142572"  --over_arch_layer_sizes="512,512,256,1"

python data_prepare.py --preprocessing_plan=1 --nDev=4 --batch_size=8192
python /workspace/RAP/RAP_end_to_end/combined_code/GPU_4_plan_1_all.py --nDev=4 --batch_size=8192 --preprocessing_plan=1

echo quit | nvidia-cuda-mps-control
nvidia-smi -i 0,1,2,3,4,5,6,7 -c DEFAULT

echo "Finish disabling MPS"