mkdir result
python torchrec_nopreprocessing_baseline.py --nDense=13 --nSparse=26 --nDev=4 --batch_size=4096 --preprocessing_plan=0 --num_embeddings_per_feature="1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194,27,14992,5461306,10,5652,2173,4,7046547,18,15,286181,105,142572"  --over_arch_layer_sizes="512,512,256,1"

python torchrec_nopreprocessing_upperbound.py --nDense=13 --nSparse=26 --nDev=4 --batch_size=4096 --preprocessing_plan=1

python torchrec_nopreprocessing_upperbound.py --nDense=13 --nSparse=26 --nDev=4 --batch_size=8192 --preprocessing_plan=0 python torchrec_nopreprocessing_baseline.py --nDense=13 --nSparse=26 --nDev=4 --batch_size=4096 --preprocessing_plan=0 --num_embeddings_per_feature="1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194,27,14992,5461306,10,5652,2173,4,7046547,18,15,286181,105,142572"  --over_arch_layer_sizes="512,512,256,1"

python torchrec_nopreprocessing_upperbound.py --nDense=13 --nSparse=26 --nDev=4 --batch_size=8192 --preprocessing_plan=1

