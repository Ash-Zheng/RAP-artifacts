import os
import sys
import torch
import pickle   
# sys.path.append('/workspace/RAP/torchrec_models')

class capacity_estimator:
    def __init__(self, batch_size, nDev, preprocessing_plan, kernel_model):
        self.nDev = nDev
        self.preprocessing_plan = preprocessing_plan
        self.train_op_list = ["emb_fwd", "mlp_fwd", "mlp_bwd", "emb_bwd", "grad_comm"]
        
        self.input_path = "/workspace/RAP/RAP_end_to_end/utils/model_capacity/plan_{}_GPU_{}_batch_{}.txt".format(preprocessing_plan, nDev, batch_size)
        capacity_matrix = []
        # Open the file and read line by line
        with open(self.input_path, 'r') as file:
            for line in file:
                if ':' in line:
                    _, numbers = line.split(':')
                    numbers = numbers.strip().split(',')
                    capacity_matrix.append([int(num) for num in numbers])


        # [TODO]: load capacity record
        # currently direct assign some previous result:
        # 
        self.capacity_dict = {}
        for i in range(self.nDev):
            gpu_dic = {}
            for idx, op in enumerate(self.train_op_list):
                gpu_dic[op] = capacity_matrix[idx][i]
            self.capacity_dict[i] = gpu_dic
        
        self.base_latency_dict = {}
        if preprocessing_plan == 0:
            if nDev == 2:
                self.base_latency_dict = {"emb_fwd": 0.90, "mlp_fwd": 1.96, "mlp_bwd": 2.88, "emb_bwd": 0.85, "grad_comm": 1.61}
            elif nDev == 4:
                self.base_latency_dict = {"emb_fwd": 0.93, "mlp_fwd": 1.92, "mlp_bwd": 2.86, "emb_bwd": 0.99, "grad_comm": 1.37}
            elif nDev == 8:
                self.base_latency_dict = {"emb_fwd": 1.01, "mlp_fwd": 1.92, "mlp_bwd": 2.91, "emb_bwd": 1.05, "grad_comm": 1.41}
        elif preprocessing_plan == 1:
            if nDev == 2:
                self.base_latency_dict = {"emb_fwd": 1.12, "mlp_fwd": 2.71, "mlp_bwd": 4.38, "emb_bwd": 1.01, "grad_comm": 1.53}
            elif nDev == 4:
                self.base_latency_dict = {"emb_fwd": 0.90, "mlp_fwd": 1.92, "mlp_bwd": 2.86, "emb_bwd": 0.97, "grad_comm": 1.71}
            elif nDev == 8:
                self.base_latency_dict = {"emb_fwd": 1.01, "mlp_fwd": 1.92, "mlp_bwd": 2.91, "emb_bwd": 1.12, "grad_comm": 1.73}
        elif preprocessing_plan == 2:
            if nDev == 2:
                self.base_latency_dict = {"emb_fwd": 1.23, "mlp_fwd": 3.41, "mlp_bwd": 7.34, "emb_bwd": 1.41, "grad_comm": 1.58}
            elif nDev == 4:
                self.base_latency_dict = {"emb_fwd": 1.31, "mlp_fwd": 3.41, "mlp_bwd": 7.38, "emb_bwd": 1.51, "grad_comm": 1.71}
            elif nDev == 8:
                self.base_latency_dict = {"emb_fwd": 1.40, "mlp_fwd": 3.41, "mlp_bwd": 7.38, "emb_bwd": 1.61, "grad_comm": 1.73}
        elif preprocessing_plan == 3:
            if nDev == 2:
                self.base_latency_dict = {"emb_fwd": 2.25, "mlp_fwd": 5.98, "mlp_bwd": 18.72, "emb_bwd": 2.61, "grad_comm": 1.58}
            elif nDev == 4:
                self.base_latency_dict = {"emb_fwd": 2.34, "mlp_fwd": 5.98, "mlp_bwd": 18.72, "emb_bwd": 2.77, "grad_comm": 1.58}
            elif nDev == 8:
                self.base_latency_dict = {"emb_fwd": 2.43, "mlp_fwd": 5.98, "mlp_bwd": 18.72, "emb_bwd": 2.85, "grad_comm": 1.59}


        # ==========================================================================================
        # Get latency-based capacity abstraction
        self.latency_capacity_list = [[0 for _ in range(len(self.train_op_list))] for _ in range(self.nDev)]
        self.latency_capacity_dict = [{} for _ in range(self.nDev)]
        for i in range(self.nDev):
            for idx, op in enumerate(self.train_op_list):
                n_batch = self.capacity_dict[i][op]
                latency = kernel_model.predict_ngram_capacity(n_batch)  
                self.latency_capacity_list[i][idx] = latency
                self.latency_capacity_dict[i][op] = latency

        # get intensity order on GPU_0
        l_list = self.latency_capacity_list[0]
        base_l_list = self.base_latency_dict.values()
        self.capacity_intensity_order = zip(self.train_op_list, l_list, base_l_list)
        self.capacity_intensity_order = sorted(self.capacity_intensity_order, key=lambda x: x[1]/x[2], reverse=True)
        

    def get_capacity(self):
        return self.latency_capacity_list, self.latency_capacity_dict, self.capacity_intensity_order
    
    def get_exec_latency(self):
        return self.base_latency_dict
    
    def layer_index(self, layer_name):
        return self.train_op_list.index(layer_name)



if __name__ == "__main__":
    # c_obj = capacity_estimator(4096, 2, 0, None)


    all_op_list =  ["fill_null", "sigrid_hash", "bucketize", "logit", "firstx", "boxcox", "clamp", "onehot", "ngram", "mapid"]
    # op = "sigrid_hash"
    # file_name = "/workspace/RAP/003_RAP_result/preprocessing_latency_predict_model/kernel_latency/{}/{}_latency.pt".format(4096, op)
    # data_and_result = torch.load(file_name)
    # print(data_and_result)

    # load_dict = pickle.load(open("kernel_latency/op_scaling_factor_dict.pkl", "rb"))
    # load_dict["firstx"] = (0.7737, 0.2686)
    # print(load_dict)

    # # save scaling factor dic
    # with open("kernel_latency/op_scaling_factor_dict.pkl", "wb") as f:
    #     pickle.dump(load_dict, f)