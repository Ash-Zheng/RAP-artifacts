load_new_batch = \
"""\
# load next_batch_file
df_sparse = cudf.read_parquet(sparse_file_name)
df_dense = cudf.read_parquet(dense_file_name)
dense_keys = list(df_dense.keys())
sparse_keys = list(df_sparse.keys())
int_ptr = []
cat_ptr = []
for i in range(len(dense_name)):
    idx = i % len(dense_keys)
    int_ptr.append(df_dense[dense_keys[idx]].data.ptr)
    int_ptr.append(df_dense[label_name].data.ptr)
for i in range(len(sparse_name)):
    idx = i % len(sparse_keys)
    cat_ptr.append(df_sparse[sparse_keys[idx]].data.ptr)

int_ptr_tensor = torch.tensor(int_ptr, dtype=torch.int64, device=device)
cat_ptr_tensor = torch.tensor(cat_ptr, dtype=torch.int64, device=device)\
"""