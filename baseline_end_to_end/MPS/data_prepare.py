import pyarrow.parquet as pq
import numpy as np
import os

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--batch_size', type=int, default=4096)
parser.add_argument('--nDev', type=int, default=2)
parser.add_argument('--plan', type=int, default=0)

def generate_parquet_file_both(rank, nDev, batch_size, out_dir, plan):
    file_name = "/workspace/RAP/{}.parquet".format("sample_data")

    nDense = 13
    nSparse = 26

    df = pq.read_table(source=file_name).to_pandas()

    total_length = batch_size * nDev
    df_all_GPU = df.iloc[:total_length,:]
    df_local = df_all_GPU.iloc[rank * batch_size:(rank + 1) * batch_size,:]

    data_parallel_name = ["label"] + ["int_{}".format(i) for i in range(nDense)]
    data_parallel_data = df_local[data_parallel_name]

    sparse_names = ["cat_{}".format(i) for i in range(nSparse)]
    both_data = df_all_GPU[data_parallel_name +  sparse_names]
    both_output_file = out_dir + "GPU_{}_both_{}.parquet".format(rank, plan)
    both_data.to_parquet(both_output_file)


if not os.path.exists("/workspace/RAP/baseline_end_to_end/MPS/generated_data"):
    os.mkdir("/workspace/RAP/baseline_end_to_end/MPS/generated_data")
else:
    os.system("rm -rf /workspace/RAP/baseline_end_to_end/MPS/generated_data")
    os.mkdir("/workspace/RAP/baseline_end_to_end/MPS/generated_data")

args = parser.parse_args()
length = args.batch_size

for i in range(args.nDev):
    generate_parquet_file_both(i, args.nDev, length, "/workspace/RAP/baseline_end_to_end/MPS/generated_data/", args.plan)
