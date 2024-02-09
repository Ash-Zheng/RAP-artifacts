mkdir result
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
nvidia-smi -i 0,1,2,3,4,5,6,7 -c EXCLUSIVE_PROCESS

nvidia-cuda-mps-control -d
ps -ef | grep mps
echo "Finish enabling MPS"


for BATCH_SIZE in 4096 8192
do
    python data_prepare.py --preprocessing_plan=0 --nDev=2 --batch_size=$BATCH_SIZE
    python /workspace/RAP/RAP_end_to_end/combined_code/GPU_2_plan_0_all.py --nDev=2 --batch_size=$BATCH_SIZE --preprocessing_plan=0 --num_embeddings_per_feature="1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194,27,14992,5461306,10,5652,2173,4,7046547,18,15,286181,105,142572"  --over_arch_layer_sizes="512,512,256,1"

    python data_prepare.py --preprocessing_plan=0 --nDev=4 --batch_size=$BATCH_SIZE
    python /workspace/RAP/RAP_end_to_end/combined_code/GPU_4_plan_0_all.py --nDev=4 --batch_size=$BATCH_SIZE --preprocessing_plan=0 --num_embeddings_per_feature="1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194,27,14992,5461306,10,5652,2173,4,7046547,18,15,286181,105,142572"  --over_arch_layer_sizes="512,512,256,1"

    python data_prepare.py --preprocessing_plan=0 --nDev=8 --batch_size=$BATCH_SIZE
    python /workspace/RAP/RAP_end_to_end/combined_code/GPU_8_plan_0_all.py --nDev=8 --batch_size=$BATCH_SIZE --preprocessing_plan=0 --num_embeddings_per_feature="1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194,27,14992,5461306,10,5652,2173,4,7046547,18,15,286181,105,142572"  --over_arch_layer_sizes="512,512,256,1"



    python data_prepare.py --preprocessing_plan=1 --nDev=2 --batch_size=$BATCH_SIZE
    python /workspace/RAP/RAP_end_to_end/combined_code/GPU_2_plan_1_all.py --nDev=2 --batch_size=$BATCH_SIZE --preprocessing_plan=1

    python data_prepare.py --preprocessing_plan=1 --nDev=4 --batch_size=$BATCH_SIZE
    python /workspace/RAP/RAP_end_to_end/combined_code/GPU_4_plan_1_all.py --nDev=4 --batch_size=$BATCH_SIZE --preprocessing_plan=1 

    python data_prepare.py --preprocessing_plan=1 --nDev=8 --batch_size=$BATCH_SIZE
    python /workspace/RAP/RAP_end_to_end/combined_code/GPU_8_plan_1_all.py --nDev=8 --batch_size=$BATCH_SIZE --preprocessing_plan=1 
done


echo quit | nvidia-cuda-mps-control
nvidia-smi -i 0,1,2,3,4,5,6,7 -c DEFAULT

echo "Finish disabling MPS"
