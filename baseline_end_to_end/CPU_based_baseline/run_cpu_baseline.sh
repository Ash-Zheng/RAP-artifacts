mkdir data_4096
mkdir data_8192
mkdir result
python data_prepare.py --batch_size=4096
python data_prepare.py --batch_size=8192


python multi_worker_torchrec_cpu_baseline.py --nDense=13 --nSparse=26 --nDev=2 --batch_size=4096 --preprocessing_plan=0  --num_embeddings_per_feature="1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194,27,14992,5461306,10,5652,2173,4,7046547,18,15,286181,105,142572"  --over_arch_layer_sizes="1024,1024,512,1"
python multi_worker_torchrec_cpu_baseline.py --nDense=13 --nSparse=26 --nDev=4 --batch_size=4096 --preprocessing_plan=0  --num_embeddings_per_feature="1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194,27,14992,5461306,10,5652,2173,4,7046547,18,15,286181,105,142572"  --over_arch_layer_sizes="1024,1024,512,1"
python multi_worker_torchrec_cpu_baseline.py --nDense=13 --nSparse=26 --nDev=8 --batch_size=4096 --preprocessing_plan=0  --num_embeddings_per_feature="1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194,27,14992,5461306,10,5652,2173,4,7046547,18,15,286181,105,142572"  --over_arch_layer_sizes="1024,1024,512,1"

python multi_worker_torchrec_cpu_baseline.py --nDense=13 --nSparse=26 --nDev=2 --batch_size=4096 --preprocessing_plan=1 --nWorker=4
python multi_worker_torchrec_cpu_baseline.py --nDense=13 --nSparse=26 --nDev=4 --batch_size=4096 --preprocessing_plan=1 --nWorker=4
python multi_worker_torchrec_cpu_baseline.py --nDense=13 --nSparse=26 --nDev=8 --batch_size=4096 --preprocessing_plan=1 --nWorker=4

python multi_worker_torchrec_cpu_baseline.py --nDense=13 --nSparse=26 --nDev=2 --batch_size=8192 --preprocessing_plan=0 --num_embeddings_per_feature="1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194,27,14992,5461306,10,5652,2173,4,7046547,18,15,286181,105,142572"  --over_arch_layer_sizes="1024,1024,512,1"
python multi_worker_torchrec_cpu_baseline.py --nDense=13 --nSparse=26 --nDev=4 --batch_size=8192 --preprocessing_plan=0 --num_embeddings_per_feature="1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194,27,14992,5461306,10,5652,2173,4,7046547,18,15,286181,105,142572"  --over_arch_layer_sizes="1024,1024,512,1"
python multi_worker_torchrec_cpu_baseline.py --nDense=13 --nSparse=26 --nDev=8 --batch_size=8192 --preprocessing_plan=0 --num_embeddings_per_feature="1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194,27,14992,5461306,10,5652,2173,4,7046547,18,15,286181,105,142572"  --over_arch_layer_sizes="1024,1024,512,1"

python multi_worker_torchrec_cpu_baseline.py --nDense=13 --nSparse=26 --nDev=2 --batch_size=8192 --preprocessing_plan=1 --nWorker=2 
python multi_worker_torchrec_cpu_baseline.py --nDense=13 --nSparse=26 --nDev=4 --batch_size=8192 --preprocessing_plan=1 --nWorker=2 
python multi_worker_torchrec_cpu_baseline.py --nDense=13 --nSparse=26 --nDev=8 --batch_size=8192 --preprocessing_plan=1 --nWorker=4 
