import os
import sys
sys.path.append('/workspace/RAP/torchrec_models')

import re
import pyarrow.parquet as pq


class table_mapping_processor:
    def __init__(self, nDev, preprocessing_plan):
        self.table_mapping = [1,1,3,2, 2,2,3,1, 1,3,1,3, 0,0,1,3, 1,2,3,0, 0,2,2,0, 0,2,0]
        if preprocessing_plan == 0 or preprocessing_plan == 1:
            if nDev == 2:
                self.table_mapping = [0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
            elif nDev == 4:
                self.table_mapping = [1,1,3,2, 2,2,3,1, 1,3,1,3, 0,0,1,3, 1,2,3,0, 0,2,2,0, 0,2,0] # preprocessing_plan_0
            elif nDev == 8:
                self.table_mapping = [1, 1, 3, 6, 2, 2, 7, 1, 5, 3, 5, 7, 0, 0, 5, 3, 1, 2, 7, 0, 4, 2, 6, 4, 4, 6, 0]
        elif preprocessing_plan == 2:
            if nDev == 2:
                self.table_mapping = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            elif nDev == 4:
                self.table_mapping = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
            elif nDev == 8:
                self.table_mapping = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3]
        elif preprocessing_plan == 3:
            if nDev == 2:
                self.table_mapping = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            elif nDev == 4:
                self.table_mapping = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
            elif nDev == 8:
                self.table_mapping = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]

        self.table_mapping_dict = {}
        self.table_placement = [[] for _ in range(nDev)]
        for idx, i in enumerate(range(len(self.table_mapping))):
            self.table_mapping_dict[i] = self.table_mapping[i]
            self.table_placement[self.table_mapping[i]].append("cat_{}".format(idx))

        self.local_dict_mapping = [[] for i in range(nDev)]
        for i in range(len(self.table_mapping)):
            table_name = "cat_" + str(i)
            self.local_dict_mapping[self.table_mapping[i]].append(table_name)


    def get_table_mapping(self):
        return self.table_mapping

    def get_table_mapping_dict(self):
        return self.table_mapping_dict
    
    def get_local_dict_mapping(self):
        return self.local_dict_mapping

    def get_table_placement(self):
        return self.table_placement


class input_splitter:
    def __init__(self, nDev, batch_size, preprocessing_plan, mapping_processor, parse_graph):
        self.nDev = nDev
        self.batch_size = batch_size
        self.preprocessing_plan = preprocessing_plan
        self.tmp = mapping_processor
        self.parse_graph = parse_graph


    def generate_parquet_file_both(self, rank, nDev, batch_size, nDense, nSparse, out_dir, plan):
        file_name = "/workspace/RAP/{}.parquet".format("sample_data")

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


    def generate_parquet_file_based_on_mapping(self, rank, nDev, batch_size, nDense, nSparse, mapping, out_dir, plan):
        file_name = "/workspace/RAP/{}.parquet".format("sample_data")

        df = pq.read_table(source=file_name).to_pandas()

        total_length = batch_size * nDev
        df_all_GPU = df.iloc[:total_length,:]
        df_local = df_all_GPU.iloc[rank * batch_size:(rank + 1) * batch_size,:]

        data_parallel_name = ["label"]
        for i in range(nDense):
            # ==== skip to avoid repeat load dense raw data ====
            # skip = False
            # for table in mapping[rank]:
            #     if "int_{}".format(i) == table:
            #         skip = True
            #         continue
            # if skip: 
            #     continue
            data_parallel_name += ["int_{}".format(i)]
        data_parallel_data = df_local[data_parallel_name]

        model_parallel_name = mapping[rank]
        model_parallel_data = df_all_GPU[model_parallel_name]

        dense_output_file = out_dir + "GPU_{}_dense_{}.parquet".format(rank, plan)
        sparse_output_file = out_dir + "GPU_{}_sparse_{}.parquet".format(rank, plan)

        data_parallel_data.to_parquet(dense_output_file)
        model_parallel_data.to_parquet(sparse_output_file)
    
    def extract_number(self, s):
        """Extract the number from the string."""
        match = re.search(r'\d+', s)
        return int(match.group()) if match else 0

    def split_input(self, output_file_dir):
        # partition the graph
        # table_mapping = tmp.get_table_mapping()
        table_mapping_dict = self.tmp.get_table_mapping_dict()
        table_placement = self.tmp.get_table_placement()
        # local_dict_mapping = tmp.get_local_dict_mapping()

        all_nodes_dense, all_nodes_sparse = self.parse_graph.partition_graph_no_dup(self.nDev, table_mapping_dict) 

        # update fill null dense index
        dense_input_name_list = []
        for node in all_nodes_dense:
            if node.op_name == "fill_null_dense" and node.input_data[0].content not in dense_input_name_list:
                dense_input_name_list.append(node.input_data[0].content)
        # dense_input_name_list.sort()
        dense_input_name_list = sorted(dense_input_name_list, key=self.extract_number)
        for node in all_nodes_dense:
            if node.op_name == "fill_null_dense":
                idx = dense_input_name_list.index(node.input_data[0].content)
                node.fill_null_index = idx
        
        # update fill null sparse index
        for gpu_id in range(len(all_nodes_sparse)):
            sparse_input_name_list = []
            for node in all_nodes_sparse[gpu_id]:
                if node.op_name == "fill_null_sparse" and node.input_data[0].content not in sparse_input_name_list:
                    sparse_input_name_list.append(node.input_data[0].content)
            # sparse_input_name_list.sort()
            sparse_input_name_list = sorted(sparse_input_name_list, key=self.extract_number)
            for node in all_nodes_sparse[gpu_id]:
                if node.op_name == "fill_null_sparse":
                    idx = sparse_input_name_list.index(node.input_data[0].content)
                    node.fill_null_index = idx

        raw_data_mapping = []
        for i in range(self.nDev):
            visited = {}
            this_list = []
            for node in all_nodes_sparse[i]:
                if not node.predecessor: # root node
                    for data in node.input_data:
                        data_name = re.findall(r'\["(.*?)"\]', data.content)[0]
                        data_id = int(re.findall(r'\d+', data.content)[0])
                        if data_name not in visited:
                            visited[data_name] = 1
                            this_list.append((data_name, data_id))
            # sort by id
            this_list.sort(key=lambda x: x[1])
            sorted_list = [x[0] for x in this_list]
            raw_data_mapping.append(sorted_list)


        print("table_placement: ")
        for line in table_placement:
            print(line)

        for i in range(self.nDev):
            for j in range(len(raw_data_mapping[i])):
                if raw_data_mapping[i][j][0] == "r":
                    raw_data_mapping[i][j] = raw_data_mapping[i][j][4:]

        # ============================== Generate a parquet file with specific batch size =============================
        for i in range(self.nDev):
            self.generate_parquet_file_based_on_mapping(i, self.nDev, self.batch_size, 13, 26, raw_data_mapping, output_file_dir, self.preprocessing_plan)
            self.generate_parquet_file_both(i, self.nDev, self.batch_size, 13, 26, output_file_dir, self.preprocessing_plan)

        return all_nodes_dense, all_nodes_sparse