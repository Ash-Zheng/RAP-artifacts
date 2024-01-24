# ==================================================================================================
# =====================================     Plan 0     =============================================
# ==================================================================================================
# 2 GPU gen code for plan 0
python RAP_pipeline.py --nDev=2 --preprocessing_plan=0 --batch_size=4096 --nPartition=1 --num_embeddings_per_feature="1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194,27,14992,5461306,10,5652,2173,4,7046547,18,15,286181,105,142572"

# 4 GPU gen code for plan 0
python RAP_pipeline.py --nDev=4 --preprocessing_plan=0 --batch_size=4096 --nPartition=1 --num_embeddings_per_feature="1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194,27,14992,5461306,10,5652,2173,4,7046547,18,15,286181,105,142572"

# 8 GPU gen code for plan 0
python RAP_pipeline.py --nDev=8 --preprocessing_plan=0 --batch_size=4096 --nPartition=1 --num_embeddings_per_feature="1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194,27,14992,5461306,10,5652,2173,4,7046547,18,15,286181,105,142572"


# ==================================================================================================
# =====================================     Plan 1     =============================================
# ==================================================================================================
# 2 GPU gen code for plan 1
python RAP_pipeline.py --nDev=2 --preprocessing_plan=1 --batch_size=4096 --nPartition=1 

# 4 GPU gen code for plan 1
python RAP_pipeline.py --nDev=4 --preprocessing_plan=1 --batch_size=4096 --nPartition=1 

# 8 GPU gen code for plan 1
python RAP_pipeline.py --nDev=8 --preprocessing_plan=1 --batch_size=4096 --nPartition=1 

