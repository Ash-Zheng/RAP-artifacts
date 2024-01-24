mkdir result
for BATCH_SIZE in 4096 8192
do
    python data_prepare.py --batch_size=$BATCH_SIZE --nDev=2 --plan=0
    python GPU_2_plan_0_stream.py --nDev=2 --batch_size=$BATCH_SIZE --preprocessing_plan=0 --num_embeddings_per_feature="1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194,27,14992,5461306,10,5652,2173,4,7046547,18,15,286181,105,142572"  --over_arch_layer_sizes="512,512,256,1"

    python data_prepare.py --batch_size=$BATCH_SIZE --nDev=4 --plan=0
    python GPU_4_plan_0_stream.py --nDev=4 --batch_size=$BATCH_SIZE --preprocessing_plan=0 --num_embeddings_per_feature="1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194,27,14992,5461306,10,5652,2173,4,7046547,18,15,286181,105,142572"  --over_arch_layer_sizes="512,512,256,1"

    python data_prepare.py --batch_size=$BATCH_SIZE --nDev=8 --plan=0
    python GPU_8_plan_0_stream.py --nDev=8 --batch_size=$BATCH_SIZE --preprocessing_plan=0 --num_embeddings_per_feature="1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194,27,14992,5461306,10,5652,2173,4,7046547,18,15,286181,105,142572"  --over_arch_layer_sizes="512,512,256,1"


    python data_prepare.py --batch_size=$BATCH_SIZE --nDev=2 --plan=1
    python GPU_2_plan_0_stream.py --nDev=2 --batch_size=$BATCH_SIZE --preprocessing_plan=1

    python data_prepare.py --batch_size=$BATCH_SIZE --nDev=4 --plan=1
    python GPU_4_plan_0_stream.py --nDev=4 --batch_size=$BATCH_SIZE --preprocessing_plan=1 

    python data_prepare.py --batch_size=$BATCH_SIZE --nDev=8 --plan=1
    python GPU_8_plan_0_stream.py --nDev=8 --batch_size=$BATCH_SIZE --preprocessing_plan=1 
done