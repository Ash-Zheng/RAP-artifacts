import os
import sys
sys.path.append('/workspace/RAP/torchrec_models')
# sys.path.append('/workspace/RAP/003_RAP_result')

from dlrm_parser import get_args
import graphviz
import jinja2

import time

import torch
import numpy as np
import pickle

import re
import pyarrow.parquet as pq

import torch.optim as optim
import torchmetrics as metrics
from pyre_extensions import none_throws
from torch import distributed as dist
from torch.utils.data import DataLoader
from torchrec import EmbeddingBagCollection
from torchrec.datasets.criteo import DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES
from torchrec.datasets.utils import Batch
from torchrec.distributed import TrainPipelineSparseDist
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.optim.keyed import KeyedOptimizerWrapper
import torch.multiprocessing as mp

from torchrec.distributed.comm import get_local_size
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.distributed.planner.storage_reservations import (
    HeuristicalStorageReservation,
)
from torchrec.distributed.model_parallel import (
    DistributedModelParallel,
    get_default_sharders,
)
from dlrm_models import DLRM_MLP
from dataloader.dlrm_dataloader import get_dataloader
from torchrec_utils import *
from sharded_embedding_table import ShardedEmbeddingTable

from torchrec.distributed.shard import shard
from torchrec.distributed.types import (
    ParameterSharding,
    ShardingPlan,
    EnumerableShardingSpec,
    ShardMetadata,
    ShardingEnv,
    ShardingType
)
from tqdm import tqdm

from torchrec.distributed.dist_data import (
    KJTAllToAll,
    PooledEmbeddingsAllToAll,
)

# import RAP utils
from utils.input_splitter import table_mapping_processor, input_splitter
from utils.graph_parser import parse_file
from utils.kernel_latency_predictor import latency_perdictor
from utils.capacity_estimator import capacity_estimator
from utils.MILP_solver import MILP_solver
from utils.fused_kernel_decoder import fused_kernel_decoder
from utils.corun_scheduler import corun_scheduler
from utils.code_generator import code_generator



def draw_the_graph(parse_graph, args):
    graph = graphviz.Digraph(comment="The Round Table")
    for node in parse_graph.nodes:
        # add node
        graph.attr('node', shape='ellipse', style='filled', color='lightblue')
        if node.raw_input != None:
            graph.node("node" + str(node.id), node.op_name + "\ninput: " + node.raw_input)
        else:
            graph.node("node" + str(node.id), node.op_name + "_" +str(node.id))

        param_content = "params:\n"
        for param in node.parameters:
            param_content += param + "\n"
        graph.edge("node" + str(node.id), "node" + str(node.id), label=param_content)

        # add input data and link to current node
        graph.attr('node', shape='diamond', style='filled', color='lightgrey')
        for data in node.input_data:
            graph.node("data" + str(data.id), data.content + "\ndim: " + str(data.dim))
            graph.edge("data" + str(data.id), "node" + str(node.id))
        
        # add output data and link to current node
        graph.node("data" + str(node.output_data.id), node.output_data.content + "\ndim: " + str(node.output_data.dim))
        graph.edge("node" + str(node.id), "data" + str(node.output_data.id))

    graph.format = 'png'
    # check whether dir exist:
    if not os.path.exists("/workspace/RAP/RAP_end_to_end/drawed_graph"):
        os.makedirs("/workspace/RAP/RAP_end_to_end/drawed_graph")
    graph.render("/workspace/RAP/RAP_end_to_end/drawed_graph/tree_plan_{}".format(args.preprocessing_plan))

def draw_subgraph_func(all_nodes, args, GPU_id):
    graph = graphviz.Digraph(comment="The Round Table")
    for node in all_nodes:
        # add node
        graph.attr('node', shape='ellipse', style='filled', color='lightblue')
        if node.raw_input != None:
            graph.node("node" + str(node.id), node.op_name + "\ninput: " + node.raw_input)
        else:
            graph.node("node" + str(node.id), node.op_name + "_" +str(node.id))

        param_content = "params:\n"
        for param in node.parameters:
            param_content += param + "\n"
        graph.edge("node" + str(node.id), "node" + str(node.id), label=param_content)

        # add input data and link to current node
        graph.attr('node', shape='diamond', style='filled', color='lightgrey')
        for data in node.input_data:
            graph.node("data" + str(data.id), data.content + "\ndim: " + str(data.dim))
            graph.edge("data" + str(data.id), "node" + str(node.id))
        
        # add output data and link to current node
        graph.node("data" + str(node.output_data.id), node.output_data.content + "\ndim: " + str(node.output_data.dim))
        graph.edge("node" + str(node.id), "data" + str(node.output_data.id))

    graph.format = 'png'
    # check whether dir exist:
    if not os.path.exists("/workspace/RAP/RAP_end_to_end/drawed_graph"):
        os.makedirs("/workspace/RAP/RAP_end_to_end/drawed_graph")
    graph.render("/workspace/RAP/RAP_end_to_end/drawed_graph/tree_plan_{}_subgraph_{}".format(args.preprocessing_plan, GPU_id))

def draw_subgraph_for_GPUs(args, all_nodes_dense, all_nodes_sparse):
    # ===== Draw sub graphs for debugging =====
    all_nodes_on_GPUs = [[] for i in range(args.nDev)]
    dense_op_cnt = {"fill_null_dense": 0, "fill_null_sparse": 0, "sigrid_hash": 0, "bucketize":0, "logit":0, "firstx":0, "boxcox":0, "clamp":0, "onehot":0, "ngram":0, "mapid":0}
    sparse_op_cnt = [{"fill_null_dense": 0, "fill_null_sparse": 0, "sigrid_hash": 0, "bucketize":0, "logit":0, "firstx":0, "boxcox":0, "clamp":0, "onehot":0, "ngram":0, "mapid":0} for _ in range(args.nDev)]
    for i in range(args.nDev):
        visited = {}
        cnt = 0
        for node in all_nodes_dense:
            if node not in visited:
                dense_op_cnt[node.op_name] += 1/args.nDev
                visited[node] = True
                all_nodes_on_GPUs[i].append(node)
                cnt += 1
        
        for node in all_nodes_sparse[i]:
            sparse_op_cnt[i][node.op_name] += 1
            if node not in visited:
                visited[node] = True
                all_nodes_on_GPUs[i].append(node)
                cnt += 1

    print("n_op on GPUs:", [len(all_nodes_on_GPUs[i]) for i in range(args.nDev)])
    for i in range(args.nDev):
        draw_subgraph_func(all_nodes_on_GPUs[i], args, i)

    print("dense_op_cnt: ", dense_op_cnt)
    print("sparse_op_cnt: ")
    for i in range(args.nDev):
        print("GPU_{}: ".format(i), sparse_op_cnt[i])


if __name__ == "__main__":
    
    args = get_args()
    file_name = "preprocessing_plans/processing_plan_{}.txt".format(args.preprocessing_plan)
    print("file_name: ", file_name)

    # # ====== Partition the graph (the table mapping is dumped from the default schedule of TorchRec) ======
    output_file_dir = "/workspace/RAP/RAP_end_to_end/splitted_input/"
    op_list =  ["fill_null_dense", "fill_null_sparse", "sigrid_hash", "bucketize", "logit", "firstx", "boxcox", "clamp", "onehot", "ngram", "mapid"]
    parse_graph = parse_file(file_name) # draw_the_graph(parse_graph, args) # visualize the preprocessing graph
    table_mp = table_mapping_processor(args.nDev, args.preprocessing_plan)
    input_split = input_splitter(args.nDev, args.batch_size, args.preprocessing_plan, table_mp, parse_graph)
    all_nodes_dense, all_nodes_sparse = input_split.split_input(output_file_dir)
    placement = table_mp.get_table_placement()
    print("finished splitting input!")


    # # # ================================
    # # # ======== Estimate Cost =========
    # # # ================================
    kernel_model = latency_perdictor(args)
    print("finish load kerel latency prediction model!")

    # # ===============================
    # # ====== Estimate Capacity ======
    # # ===============================
    cap_est = capacity_estimator(args.batch_size, args.nDev, args.preprocessing_plan, kernel_model)
    latency_capacity_list, latency_capacity_dict, capacity_intensity_order = cap_est.get_capacity()
    dlrm_exec_latency_dict = cap_est.get_exec_latency()

    # print(latency_capacity_list)
    # print(capacity_intensity_order)

    # # # ===============================
    # # # ========= MILP Solver =========
    # # # ===============================
    table_mapping_dict = table_mp.get_table_mapping_dict()

    # if no fusion plan, do MILP solver
    if os.path.exists("searched_fused_kernels/plan-{}_nGPU-{}.pkl".format(args.preprocessing_plan, args.nDev)):
        with open("searched_fused_kernels/plan-{}_nGPU-{}.pkl".format(args.preprocessing_plan, args.nDev), "rb") as f:
            fused_kernels = pickle.load(f)
    else: 
        solver = MILP_solver(args.nDev, args.nPartition, op_list, partition_type=0)
        # fused_kernel_list = solver.solve(table_mapping_dict, parse_graph)
        fused_kernel_list = solver.solve_seperate(table_mapping_dict, parse_graph)

        kernel_decoder = fused_kernel_decoder(args.nDev, fused_kernel_list, parse_graph, args.preprocessing_plan)
        kernel_decoder.print_fused_kernel_info()
        kernel_decoder.save_fused_kernel(args.preprocessing_plan)
        fused_kernels = kernel_decoder.get_kernel_on_GPUs()

    # # # # ===============================
    # # # # ======= Corun Schedule ========
    # # # # ===============================
    scheduler = corun_scheduler(args.nDev, args.preprocessing_plan ,fused_kernels, latency_capacity_list, latency_capacity_dict, capacity_intensity_order, dlrm_exec_latency_dict, kernel_model)
    schedule = scheduler.schedule()
    scheduler.print_schedule()

    # # # # ===============================
    # # # # ========== Code Gen ===========
    # # # # ===============================
    all_node = parse_graph.nodes
    dlrm_layers = ["emb_fwd", "mlp_fwd", "mlp_bwd", "emb_bwd", "grad_comm"]
    c_gen = code_generator(args.nDev, args.preprocessing_plan, dlrm_layers, folder_name="generated_code")
    c_gen.init_all_codes(schedule, all_node)
    c_gen.RAP_gen_code_all()
