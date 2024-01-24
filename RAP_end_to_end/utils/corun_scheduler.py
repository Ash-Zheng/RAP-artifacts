import copy
import torch
import cudf


class corun_scheduler:
    def __init__(self, nDev, plan, fused_kernels, latency_capacity_list, latency_capacity_dict, capacity_intensity_order, dlrm_exec_latency_dict, kernel_latency_predictor, data_prepare_latency_predictor=None):
        self.nDev = nDev
        self.plan = plan
        self.fused_kernels = fused_kernels
        self.latency_capacity_list = latency_capacity_list
        self.latency_capacity_dict = latency_capacity_dict
        self.capacity_intensity_order = capacity_intensity_order
        self.dlrm_exec_latency_dict = dlrm_exec_latency_dict
        self.kernel_latency_predictor = kernel_latency_predictor
        self.data_prepare_latency_predictor = data_prepare_latency_predictor
        self.data_loading_latency = []
        self.nLayer = len(latency_capacity_list[0])
        self.overflow_sign = [0 for _ in range(nDev)]
        self.dlrm_layers = ["emb_fwd", "mlp_fwd", "mlp_bwd", "emb_bwd", "grad_comm"]

    def schedule(self):
        schedule_plan = [[[] for _ in range(self.nLayer)] for _ in range(self.nDev)]
        for i in range(self.nDev):
            # 1. compute the latency of all kernels:
            total_latency = 0
            kernels_on_GPU = self.fused_kernels[i]
            for kernel in kernels_on_GPU:
                total_latency += self.kernel_latency_predictor.predict_fused_kernel(kernel)

            # 2. select layers to overlap
            layer_total_latency = 0
            select_layer_list = []
            for layer in self.capacity_intensity_order:
                layer_name = layer[0]
                layer_latency = self.latency_capacity_dict[i][layer_name]
                layer_total_latency += layer_latency
                select_layer_list.append(layer_name)
                if layer_total_latency > total_latency:
                    break # select enough layer
            if total_latency > sum(self.latency_capacity_list[0]): # the kernel latency exceeds the capacity
                self.overflow_sign[i]=1
            
            # 3. get schedule plan:
            remained_layer_latency = copy.deepcopy(self.latency_capacity_dict[i])
            kernel_idx = 0
            for idx, layer_name in enumerate(self.dlrm_layers):
                if layer_name in select_layer_list: # if selected layer
                    remained_latency = remained_layer_latency[layer_name]
                    while kernel_idx < len(kernels_on_GPU):
                        kernel_latency = self.kernel_latency_predictor.predict_fused_kernel(kernels_on_GPU[kernel_idx])
                        if remained_latency > kernel_latency: # sufficient overlapping capacity
                            schedule_plan[i][idx].append(kernels_on_GPU[kernel_idx])
                            remained_latency -= kernel_latency
                            kernel_idx += 1
                        elif kernels_on_GPU[kernel_idx].nOp > 1: # Resource-ware kernel sharding
                            ori_nOp = kernels_on_GPU[kernel_idx].nOp
                            for k in range(ori_nOp-1, 1, -1):
                                sharded_latency = self.kernel_latency_predictor.predict_fused_kernel(kernels_on_GPU[kernel_idx], nOp=k)
                                if remained_latency > sharded_latency: # shard
                                    kernel_1, kernel_2 = kernels_on_GPU[kernel_idx].kernel_shard(k)
                                    schedule_plan[i][idx].append(kernel_1)
                                    remained_latency -= sharded_latency
                                    kernels_on_GPU[kernel_idx] = kernel_2 # set the current kernel to kernel-2
                                break
                            break
                        else: # add to next layer
                            break

            # if still remained
            while kernel_idx < len(kernels_on_GPU):
                schedule_plan[i][-1].append(kernels_on_GPU[kernel_idx])
                kernel_idx += 1

        self.schedule_plan = schedule_plan
        return schedule_plan
    

    def data_loading_latency_measurement(self):
        self.data_loading_latency = []
        for gpu_id in range(self.nDev):
            data_dir = "/workspace/RAP/RAP_end_to_end/splitted_input/"
            dense_file_name = data_dir + "GPU_{}_dense_{}.parquet".format(gpu_id, self.plan)
            sparse_file_name = data_dir + "GPU_{}_sparse_{}.parquet".format(gpu_id, self.plan)
            both_file_name = data_dir + "GPU_{}_both_{}.parquet".format(gpu_id, self.plan)

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            for i in range(10):
                df_sparse = cudf.read_parquet(sparse_file_name)
                df_dense = cudf.read_parquet(dense_file_name)
            torch.cuda.synchronize()
            
            start_event.record()
            for i in range(10):
                # df_sparse = cudf.read_parquet(sparse_file_name)
                # df_dense = cudf.read_parquet(dense_file_name)
                df_both = cudf.read_parquet(both_file_name)
            end_event.record()
            torch.cuda.synchronize()

            data_loading_latency = start_event.elapsed_time(end_event)
            avg_loading_latency = data_loading_latency / 10
            print("GPU_{} data loading latency: {}".format(gpu_id, avg_loading_latency))
            self.data_loading_latency.append(avg_loading_latency)

    def print_schedule(self):
        for i in range(self.nDev):
            print("GPU-{}:".format(i))
            for idx in range(len(self.dlrm_layers)):
                info = "{}:[".format(self.dlrm_layers[idx])
                for kernel in self.schedule_plan[i][idx]:
                    info += "({},{})".format(kernel.op_name, kernel.nOp)
                info += "]\n"
                print(info)
    