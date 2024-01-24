import torch
import numpy as np
import gurobipy as gp
from gurobipy import GRB

def get_node_name(node):
    return node.op_name + "_" + str(node.batch_size)

class MILP_solver:
    def __init__(self, nDev, nPartition, op_list, partition_type=0):
        self.nDev = nDev
        self.partition_type = partition_type # 0 for horizontal, 1 for vertical
        self.nPartition = nPartition
        self.op_list = []
        for op in op_list:
            self.op_list.append(op+"_1")
            self.op_list.append(op+"_2")

    def MILP_formulation(self, all_nodes_on_GPUs, all_nodes_idx_on_GPUs):
        # Start Solving for each GPU
        fused_kernel_list_on_GPUs = []
        
        for gpu_id in range(self.nDev):
            all_nodes = all_nodes_on_GPUs[gpu_id]
            all_nodes_idx = all_nodes_idx_on_GPUs[gpu_id]

            nNode = len(all_nodes)
            # nLayer = nNode  # maximum number of layers, it can potentially reduced by fusing
            nLayer = 8

            per_op_cnt = {op:0 for op in self.op_list}

            for node in all_nodes:
                if get_node_name(node) in self.op_list:
                    per_op_cnt[get_node_name(node)] += 1
            
            # print(all_nodes_idx)
            # print(self.op_list)
            print(per_op_cnt)
            print("=====================================================")
            print("=====================================================")
            print("=====================================================")


            try:
                # Create a new model
                m = gp.Model("mip1")

                m.setParam(GRB.Param.MIPGap, 0.003)

                # Create variables
                layer_val_list = np.arange(nLayer)
                op_layer_list = [] # nNode * nLayer, binary matrix for each node
                op_layer_val_list = [] # 1 * nNode, dot product value for each node
                op_layer_sum_list = [] # 1 * nNode, sum of each row for each node, must equal to 1
                for node in all_nodes:
                    this_op_list = []
                    layer_val = 0
                    layer_sum = 0
                    for i in range(nLayer): 
                        op_var = m.addVar(vtype=GRB.BINARY, name="op_" + str(node.id) + "_" + str(i))
                        this_op_list.append(op_var)
                        layer_val += op_var * layer_val_list[i] # dot product value indicates the layer number
                        layer_sum += op_var
                    op_layer_list.append(this_op_list) # 1 * nLayer, binary vector for each node
                    op_layer_val_list.append(layer_val) 
                    op_layer_sum_list.append(layer_sum)

                # Set constraints
                # 1. data dependency constraints
                #     op_layer_val < successor.op_layer_val
                constraint_cnt = 0
                for node in all_nodes:
                    for pre in node.predecessor:
                        if pre in all_nodes:
                            m.addConstr(op_layer_val_list[all_nodes_idx[node.id]] >= op_layer_val_list[all_nodes_idx[pre.id]] + 1, "c_{}".format(constraint_cnt))
                            constraint_cnt += 1

                
                # 2. placement constraints
                #    Each op only placed in one layer
                constraint_cnt = 0
                for i, node in enumerate(all_nodes):
                    m.addConstr(op_layer_sum_list[i] == 1, "c_{}".format(constraint_cnt))
                    constraint_cnt += 1

                # Set objective
                per_op_objective = {}
                for op in self.op_list:
                    per_op_objective[op] = [0 for i in range(nLayer)] # 1 * nLayer, objective value for each op
                for idx, node in enumerate(all_nodes):
                    node_name = get_node_name(node)
                    this_node_placement_list = op_layer_list[idx] # 1 * nLayer, binary vector for each node
                    this_node_objective = per_op_objective[node_name]

                    for i in range(len(this_node_objective)):
                        this_node_objective[i] += this_node_placement_list[i]
                
                total_obj = 0
                for op in self.op_list:
                    if per_op_cnt[op] == 0: # if there is no such op in the graph, skip
                        continue
                    this_objective = per_op_objective[op]
                    for i in range(len(this_node_objective)):
                        total_obj += this_objective[i] * this_objective[i] 
                        # [TODO]: consider the impact of fusion for different operations (fuse_reduction_ration[op_list.index(op)])
                
                m.setObjective(total_obj, GRB.MAXIMIZE)
                m.optimize()

                result_record = torch.zeros(nNode, nLayer, dtype=torch.int32)
                for i, layer_list in enumerate(op_layer_list): 
                    for j, var in enumerate(layer_list):
                        if var.x == 1:
                            result_record[i][j] = 1
                
                # for i in range(nNode):
                    # print(result_record[i], all_nodes[i].op_name, all_nodes[i].id)

                # torch.save(result_record, "result_record.pt")

                # Parse the fusion schedule
                fuse_schedule = []
                for i in range(result_record.shape[1]):
                    this_layer_fuse = []
                    placement_layer = result_record[:,i]
                    for j in range(len(placement_layer)):
                        if placement_layer[j] == 1:
                            op_node = all_nodes[j]
                            # op_name = op_node.op_name
                            op_name = get_node_name(op_node)
                            this_layer_fuse.append((op_name, op_node.id))
                            
                    this_layer_fuse.sort(key=lambda x: x[0])
                    if len(this_layer_fuse) > 0:
                        fuse_schedule.append(this_layer_fuse)

                fused_kernel_list = []
                for i, layer in enumerate(fuse_schedule):
                    this_fused_kernel = []
                    this_kernel_op = ""
                    for op in layer:
                        if len(this_fused_kernel) == 0:
                            this_fused_kernel.append(op)
                            this_kernel_op = op[0]
                        elif op[0] == this_kernel_op: # same op
                            this_fused_kernel.append(op)
                        else: # different op
                            fused_kernel_list.append(this_fused_kernel)
                            this_fused_kernel = [op]
                            this_kernel_op = op[0]

                    if len(this_fused_kernel) > 0:
                        fused_kernel_list.append(this_fused_kernel)
                
                for i, kernel in enumerate(fused_kernel_list):
                    print("Kernel {}: {}".format(i, kernel))

                fused_kernel_list_on_GPUs.append(fused_kernel_list)

            except gp.GurobiError as e:
                print('Error code ' + str(e.errno) + ': ' + str(e))

            except AttributeError:
                print('Encountered an attribute error')  

        return fused_kernel_list_on_GPUs

            
    def solve(self, table_mapping_dict, parse_graph):
        fused_kernel_list = [[] for i in range(self.nDev)]

        if self.partition_type == 0: # Partitioning the graph for large input preprocessing graph
            partition_nodes_dense, partition_nodes_sparse = parse_graph.partition_horizontal(self.nDev, table_mapping_dict, self.nPartition)
        else:
            partition_nodes_dense, partition_nodes_sparse = parse_graph.partition_vertical(self.nDev, table_mapping_dict, self.nPartition)

        for group in range(self.nPartition):
            all_nodes_on_GPUs = [[] for i in range(self.nDev)]
            all_nodes_idx_on_GPUs = [{} for i in range(self.nDev)]
            for i in range(self.nDev):
                visited = {}
                cnt = 0
                for node in partition_nodes_dense[group]:
                    if node not in visited:
                        visited[node] = True
                        all_nodes_on_GPUs[i].append(node)
                        all_nodes_idx_on_GPUs[i][node.id] = cnt
                        cnt += 1
                for node in partition_nodes_sparse[i][group]:
                    if node not in visited:
                        visited[node] = True
                        all_nodes_on_GPUs[i].append(node)
                        all_nodes_idx_on_GPUs[i][node.id] = cnt
                        cnt += 1
            
            fused_kernel_list_on_GPUs = self.MILP_formulation(all_nodes_on_GPUs, all_nodes_idx_on_GPUs)

            print(len(fused_kernel_list_on_GPUs), self.nDev)
            for gpu_id in range(self.nDev):
                for kernel in fused_kernel_list_on_GPUs[gpu_id]:
                    fused_kernel_list[gpu_id].append(kernel)

        
        return fused_kernel_list

    def solve_seperate(self, table_mapping_dict, parse_graph):
        fused_kernel_list = [[] for i in range(self.nDev)]

        if self.partition_type == 0: # Partitioning the graph for large input preprocessing graph
            partition_nodes_dense, partition_nodes_sparse = parse_graph.partition_horizontal(self.nDev, table_mapping_dict, self.nPartition)
        else:
            partition_nodes_dense, partition_nodes_sparse = parse_graph.partition_vertical(self.nDev, table_mapping_dict, self.nPartition)

        # solve dense nodes
        for group in range(self.nPartition):
            all_nodes_on_GPUs = [[] for i in range(self.nDev)]
            all_nodes_idx_on_GPUs = [{} for i in range(self.nDev)]

            for i in range(self.nDev):
                visited = {}
                cnt = 0
                for node in partition_nodes_dense[group]:
                    if node not in visited:
                        visited[node] = True
                        all_nodes_on_GPUs[i].append(node)
                        all_nodes_idx_on_GPUs[i][node.id] = cnt
                        cnt += 1
            
            fused_kernel_list_on_GPUs = self.MILP_formulation(all_nodes_on_GPUs, all_nodes_idx_on_GPUs)

            for gpu_id in range(self.nDev):
                for kernel in fused_kernel_list_on_GPUs[gpu_id]:
                    fused_kernel_list[gpu_id].append(kernel)
        
        # solve sparse nodes
        for group in range(self.nPartition):
            all_nodes_on_GPUs = [[] for i in range(self.nDev)]
            all_nodes_idx_on_GPUs = [{} for i in range(self.nDev)]

            for i in range(self.nDev):
                visited = {}
                cnt = 0
                for node in partition_nodes_sparse[i][group]:
                    if node not in visited:
                        visited[node] = True
                        all_nodes_on_GPUs[i].append(node)
                        all_nodes_idx_on_GPUs[i][node.id] = cnt
                        cnt += 1
            
            fused_kernel_list_on_GPUs = self.MILP_formulation(all_nodes_on_GPUs, all_nodes_idx_on_GPUs)

            for gpu_id in range(self.nDev):
                for kernel in fused_kernel_list_on_GPUs[gpu_id]:
                    fused_kernel_list[gpu_id].append(kernel)

        return fused_kernel_list

