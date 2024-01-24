import pyarrow.parquet as pq
import numpy as np

def generate_parquet_file(rank, nDev, batch_size, plan, nDense, nSparse):
    file_name = "/workspace/RAP/parquet_data/{}.parquet".format("sample_data")

    df = pq.read_table(source=file_name).to_pandas()

    total_length = batch_size * nDev
    df_all_GPU = df.iloc[:total_length,:]
    df_local = df_all_GPU.iloc[rank * batch_size:(rank + 1) * batch_size,:]

    data_parallel_name = ["label"] + ["int_{}".format(i) for i in range(nDense)]
    data_parallel_data = df_local[data_parallel_name]


    sparse_names = ["cat_{}".format(i) for i in range(nSparse)]

    model_parallel_name = []
    sharding_dict = list(plan.plan.values())[0]
    for table_name in sparse_names:
        table_mapping = sharding_dict["t_" + table_name].ranks
        if rank in table_mapping:
            model_parallel_name.append(table_name)
                
    model_parallel_data = df_all_GPU[model_parallel_name]

    both_data = df_all_GPU[data_parallel_name +  model_parallel_name]

    dense_output_file = "/workspace/RAP/generated_data/GPU_{}_dense.parquet".format(rank)
    sparse_output_file = "/workspace/RAP/generated_data/GPU_{}_sparse.parquet".format(rank)
    both_output_file = "/workspace/RAP/generated_data/GPU_{}_both.parquet".format(rank)

    data_parallel_data.to_parquet(dense_output_file)
    model_parallel_data.to_parquet(sparse_output_file)
    both_data.to_parquet(both_output_file)


def generate_parquet_file_based_on_mapping(rank, nDev, batch_size, nDense, nSparse, mapping):
    file_name = "/workspace/RAP/parquet_data/{}.parquet".format("sample_data")

    df = pq.read_table(source=file_name).to_pandas()

    total_length = batch_size * nDev
    df_all_GPU = df.iloc[:total_length,:]
    df_local = df_all_GPU.iloc[rank * batch_size:(rank + 1) * batch_size,:]

    data_parallel_name = ["label"] + ["int_{}".format(i) for i in range(nDense)]
    data_parallel_data = df_local[data_parallel_name]


    sparse_names = ["cat_{}".format(i) for i in range(nSparse)]

    model_parallel_name = mapping[rank]
    model_parallel_data = df_all_GPU[model_parallel_name]


    dense_output_file = "/workspace/RAP/generated_data/GPU_{}_dense.parquet".format(rank)
    sparse_output_file = "/workspace/RAP/generated_data/GPU_{}_sparse.parquet".format(rank)

    data_parallel_data.to_parquet(dense_output_file)
    model_parallel_data.to_parquet(sparse_output_file)