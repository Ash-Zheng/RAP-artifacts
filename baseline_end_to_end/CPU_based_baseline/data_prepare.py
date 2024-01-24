import pyarrow.parquet as pq
import numpy as np

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--batch_size', type=int, default=4096)

# ============================== Generate a small sample parquet file for quick iteration =============================
# file_name = "/home/zhengw/dlrm_dataset/terabyte/output/criteo_parquet/test_parquet/day_2.part0.parquet"
# df = pq.read_table(source=file_name).to_pandas()

# df_1 = df.iloc[:8192 * 16,:]
# print(df_1.shape)
# df_1.to_parquet("parquet_data/{}.parquet".format("sample_data"))
# ====================================================================================================================


# ============================== Generate a parquet file with specific batch size =============================
args = parser.parse_args()
length = args.batch_size
file_name = "/workspace/RAP/{}.parquet".format("sample_data")
output_file = "/workspace/RAP/baseline_end_to_end/CPU_based_baseline/data_{}/first_{}.parquet".format(length, length)

df = pq.read_table(source=file_name).to_pandas()
df_1 = df.iloc[:length,:]
print(df_1.shape)
df_1.to_parquet(output_file)
