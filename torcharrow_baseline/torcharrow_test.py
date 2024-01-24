import pyarrow.parquet as pq
import numpy as np

# ============================== Generate a small sample parquet file for quick iteration =============================
# file_name = "/home/zhengw/dlrm_dataset/terabyte/output/criteo_parquet/test_parquet/day_2.part0.parquet"
# df = pq.read_table(source=file_name).to_pandas()

# df_1 = df.iloc[:8192 * 16,:]
# print(df_1.shape)
# df_1.to_parquet("parquet_data/{}.parquet".format("sample_data"))
# ====================================================================================================================



# ============================== Generate a parquet file with specific batch size =============================
# length = 4096
# file_name = "parquet_data/{}.parquet".format("sample_data")
# output_file = "test_data_folder/first1_{}.parquet".format(length)

# df = pq.read_table(source=file_name).to_pandas()
# df_1 = df.iloc[:length,:]
# print(df_1.shape)
# df_1.to_parquet(output_file)
# =============================================================================================================

length = 10
file_name = "parquet_data/{}.parquet".format("sample_data")
output_file = "test_data_folder/first1_{}.parquet".format(length)

df = pq.read_table(source=file_name).to_pandas()
df_1 = df.iloc[:length,:]
# print(df_1)
