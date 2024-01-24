import argparse
import itertools
import os
import sys
sys.path.append('/workspace/RAP/torchrec_models')

from typing import Iterator, List

import torch
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
from dlrm_parser import get_args
from torchrec_utils import *
from sharded_embedding_table import ShardedEmbeddingTable
from torch.utils.cpp_extension import load
import cudf
from torchrec_prepare_data import generate_parquet_file_based_on_mapping
import random

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
import time

from torchrec.distributed.dist_data import (
    KJTAllToAll,
    PooledEmbeddingsAllToAll,
)

import warnings

def free_memory(a, b, c):
    del a
    del b
    del c


def train_and_preprocessing_process(rank, nDev, nProcess, args, queue_list, input_queue_list, if_train) -> None:
    warnings.filterwarnings('ignore')
    random.seed(rank)
    torch.manual_seed(rank)

    if rank < nDev:
        device = torch.device(f"cuda:{rank}")
        pair_rank = rank + nDev
    else:
        device = torch.device(f"cuda:{rank-nDev}")
        this_rank = rank - nDev
        pair_rank = rank - nDev
    
    backend = "gloo"  # synchronization between preprocessing and training process using gloo
    nccl_backend = "nccl"  # synchronization between training process using nccl

    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=nProcess)
    dist.barrier()
    print_once(rank, "finish init global dist")

    train_rank_list = [i for i in range(nDev)]
    preprocess_rank_list = [i+nDev for i in range(nDev)]

    train_group = torch.distributed.new_group(backend='nccl', ranks=train_rank_list)  # group of sub-process for training
    preprocess_group = torch.distributed.new_group(backend='nccl', ranks=preprocess_rank_list)  # group of sub-process for training
    global_group = dist.group.WORLD
    if if_train:
        dist.barrier(group=train_group)
    else:
        dist.barrier(group=preprocess_group)
    print_once(rank, "finish init train and preprocess dist")

    # Updata parameter setting

    if args.preprocessing_plan == 0 or args.preprocessing_plan == 1:
        args.num_embeddings_per_feature = args.num_embeddings_per_feature + [65536]
        args.cat_name = args.cat_name + ["bucketize_int_0"]
        args.nSparse = args.nSparse + 1
    elif args.preprocessing_plan == 2:
        args.nSparse = 26 * 2
        args.nDense = 13 * 2
        args.cat_name = ["cat_{}".format(i) for i in range(args.nSparse)]
        args.int_name = ["int_" + str(i) for i in range(args.nDense)]
        args.num_embeddings_per_feature = [65536 for _ in range(args.nSparse)]
    elif args.preprocessing_plan == 3:
        args.nSparse = 26 * 4
        args.nDense = 13 * 4
        args.cat_name = ["cat_{}".format(i) for i in range(args.nSparse)]
        args.int_name = ["int_" + str(i) for i in range(args.nDense)]
        args.num_embeddings_per_feature = [65536 for _ in range(args.nSparse)]

    if if_train:
        torch.cuda.set_device(rank)

        eb_configs = [
            EmbeddingBagConfig(
                name=f"t_{feature_name}",
                embedding_dim=args.embedding_dim,
                num_embeddings = args.num_embeddings_per_feature[feature_idx],
                feature_names=[feature_name],
            )
            for feature_idx, feature_name in enumerate(args.cat_name)
        ]

        embedding_tables = EmbeddingBagCollection(
            tables=eb_configs,
            device=torch.device("meta"), # "meta" model will not allocate memory until sharding
        )

        sharding_constraints = {
            f"t_{feature_name}": ParameterConstraints(
            sharding_types=[ShardingType.TABLE_WISE.value],  # TABLE_WISE, ROW_WISE, COLUMN_WISE, DATA_PARALLEL
            ) for feature_idx, feature_name in enumerate(args.cat_name)
        }

        planner = EmbeddingShardingPlanner(
            topology=Topology(
                local_world_size=nDev,
                world_size=nDev,
                compute_device=device.type,
            ),
            batch_size=args.batch_size,
            storage_reservation=HeuristicalStorageReservation(percentage=0.01),
            constraints=sharding_constraints,
        )

        plan = planner.collective_plan(
            module=embedding_tables, sharders=get_default_sharders(), pg=train_group
        )

        # =================== Make sure sharding is correct ===================
        nTable = len(args.cat_name)
        table_mapping = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
        
        if nTable != len(table_mapping):
            raise ValueError("nTable != len(table_mapping)")
        table_names = plan.plan[''].keys()

        for idx, t_name in enumerate(table_names):
            plan.plan[''][t_name].ranks = [table_mapping[idx]]
            plan.plan[''][t_name].sharding_spec.shards[0].placement._device = torch.device("cuda:{}".format(table_mapping[idx]))
            plan.plan[''][t_name].sharding_spec.shards[0].placement._rank = table_mapping[idx]

        # if rank == 0:
            # print(plan)  
        #====================================================

        sharded_emts = ShardedEmbeddingTable(
            embedding_tables=embedding_tables, 
            plan=plan, 
            device=device, 
            group=train_group, 
            dim=args.embedding_dim,
            batch_size=args.batch_size, 
            table_names=args.cat_name,
            input_queue=queue_list[rank],
        )

        mlp_layers = DLRM_MLP(
            embedding_dim=args.embedding_dim,
            num_sparse_features=args.nSparse,
            dense_in_features=args.nDense,
            dense_arch_layer_sizes=args.dense_arch_layer_sizes,
            over_arch_layer_sizes=args.over_arch_layer_sizes,
            dense_device=device,
        )

        sparse_optimizer = KeyedOptimizerWrapper(
            dict(sharded_emts.named_parameters()),
            lambda params: torch.optim.SGD(params, lr=args.learning_rate),
        )
        dense_optimizer = optim.SGD(mlp_layers.parameters(), lr=args.learning_rate)

        loss_fn = torch.nn.BCEWithLogitsLoss()

        train_dataloader = get_dataloader(args, backend, "train")
        in_mem_dataloader = InMemoryDataLoader(train_dataloader, rank, nDev, 16)

        batch = in_mem_dataloader.next()
        dist_input = sharded_emts.input_comm(batch.sparse_features) # dist_input[0] is JaggedTensor
        # print("rank", rank, "dist_input[0]._values.shape: ", dist_input[0]._values.shape)

        dist.barrier(group=train_group)
        print_once(rank, "finish model initalization")

    else:
        # generate_parquet_file_based_on_mapping(this_rank, nDev, args.batch_size, args.nDense, args.nSparse, raw_data_mapping_list)

        table_length_dic = {}
        for idx, table_name in enumerate(args.cat_name):
            table_length_dic[table_name] = args.num_embeddings_per_feature[idx]

        cuda_preprocess = load(name="gpu_operators", sources=[
        "/workspace/RAP/cuda_operators/cuda_wrap.cpp", 
        "/workspace/RAP/cuda_operators/gpu_operators.cu", 
        ], verbose=False)

        cuda_preprocess.init_cuda(this_rank)

        label_name = "label"
        sparse_name = args.cat_name
        dense_name = args.int_name

        border = torch.tensor([1,2,3]).to(this_rank)

        data_dir = "/workspace/RAP/breakdown_study/MPS_and_sequential/generated_data/"
        both_file_name = data_dir + "GPU_{}_both_{}.parquet".format(this_rank, args.preprocessing_plan)

        local_table_length_list = [table_length_dic[table_name] for table_name in sparse_name]

        # ===============================  Pointer Prepare  ====================================================
        df_both = cudf.read_parquet(both_file_name)

        # ===============================  Gnerated Code started  ==============================================
        # ==================== load data and fill null  ====================
        int_ptr = []
        cat_ptr = []
        for i in range(len(dense_name)):
            idx = i % 13
            int_name = "int_{}".format(idx)
            int_ptr.append(df_both[int_name].data.ptr)
        int_ptr.append(df_both[label_name].data.ptr)
        for i in range(len(sparse_name)):
            idx = i % 26
            cat_name = "cat_{}".format(idx)
            cat_ptr.append(df_both[cat_name].data.ptr)

        sparse_tensor_list = []
        for i in range(len(sparse_name)):
            sparse_tensor_list.append(cuda_preprocess.fill_null_int64(cat_ptr[i], args.batch_size))
        dense_label_tensor_list = []
        for i in range(len(int_ptr)):
            dense_label_tensor_list.append(cuda_preprocess.fill_null_float(int_ptr[i], args.batch_size))
        label_tensor = dense_label_tensor_list[-1].squeeze(1)
        dense_tensor_list = dense_label_tensor_list[0:-1]
        dense_output_tensor_list = [torch.rand((args.batch_size,1), dtype=torch.float32, device=device)* 100000 for _ in range(args.nDense)] # only dense
        sparse_output_tensor_list = [torch.randint(0, 65536, (args.batch_size,1), dtype=torch.int64, device=device) for _ in range(len(cat_ptr))] # only sparse
        # ===============================================================
        logit_input_list_1 = [torch.rand((args.batch_size,1), dtype=torch.float32, device=device)* 100000 for _ in range(7)]
        eps_list_1 = [1e-5 for _ in range(7)]
        for i in range(len(logit_input_list_1)):
            cuda_preprocess.logit(logit_input_list_1[i], eps_list_1[i])
        boxcox_input_list_2 = [torch.rand((args.batch_size,1), dtype=torch.float32, device=device)* 100000 for _ in range(10)]
        lambda_list_2 = [0.5 for _ in range(10)]
        for i in range(len(boxcox_input_list_2)):
            cuda_preprocess.boxcox(boxcox_input_list_2[i], lambda_list_2[i])
        onehot_input_list_3 = [torch.randint(0, 100000, (args.batch_size,1), dtype=torch.float, device=device) for _ in range(9)]
        low_list_3 = [0.0 for _ in range(9)]
        high_list_3 = [16.0 for _ in range(9)]
        num_classes_list_3 = [5 for _ in range(9)]
        onehot_output_list_3 = []
        for i in range(len(onehot_input_list_3)):
            onehot_output_list_3.append(cuda_preprocess.onehot(onehot_input_list_3[i], low_list_3[i], high_list_3[i], num_classes_list_3[i]))
        first_input_list_4 = [torch.randint(0, 100000, (args.batch_size, 1), dtype=torch.int64, device=device) for _ in range(10)]
        x_list_4 = [1 for _ in range(10)]
        firstx_output_list_4 = []
        for i in range(len(first_input_list_4)):
            firstx_output_list_4.append(cuda_preprocess.firstx(first_input_list_4[i], x_list_4[i]))
        # sparse ===============================================================
        onehot_input_list_5 = [torch.randint(0, 100000, (args.batch_size,1), dtype=torch.float, device=device) for _ in range(13)]
        low_list_5 = [0.0 for _ in range(13)]
        high_list_5 = [16.0 for _ in range(13)]
        num_classes_list_5 = [5 for _ in range(13)]
        onehot_output_list_5 = []
        for i in range(len(onehot_input_list_5)):
            onehot_output_list_5.append(cuda_preprocess.onehot(onehot_input_list_5[i], low_list_5[i], high_list_5[i], num_classes_list_5[i]))
        ngram_input_list_6 = [torch.randint(0, 100000, (args.batch_size, 8), dtype=torch.int64, device=device) for _ in range(13)]
        ngram_n_list_6 = [41 for _ in range(13)]
        ngram_output_list_6 = []
        for i in range(len(ngram_input_list_6)):
            ngram_output_list_6.append(cuda_preprocess.ngram(ngram_input_list_6[i], ngram_n_list_6[i]))
        first_input_list_7 = [torch.randint(0, 100000, (args.batch_size, 1), dtype=torch.int64, device=device) for _ in range(13)]
        x_list_7 = [1 for _ in range(13)]
        firstx_output_list_7 = []
        for i in range(len(first_input_list_7)):
            firstx_output_list_7.append(cuda_preprocess.firstx(first_input_list_7[i], x_list_7[i]))
        sigrid_hash_input_list_8 = [torch.randint(0, 100000, (args.batch_size,1), dtype=torch.int64, device=device) for _ in range(2)]
        sigrid_hash_length_list_8 = [65536 for _ in range(2)]
        for i in range(len(sigrid_hash_input_list_8)):
            cuda_preprocess.sigrid_hash(sigrid_hash_input_list_8[i], 0, sigrid_hash_length_list_8[i])
        clamp_input_list_9 = [torch.randint(0, 100000, (args.batch_size,1), dtype=torch.int64, device=device) for _ in range(7)]
        clamp_min_list_9 = [0 for _ in range(7)]
        clamp_max_list_9 = [65536 for _ in range(7)]
        for i in range(len(clamp_input_list_9)):
            cuda_preprocess.clamp(clamp_input_list_9[i], clamp_min_list_9[i], clamp_max_list_9[i])
        first_input_list_10 = [torch.randint(0, 100000, (args.batch_size, 1), dtype=torch.int64, device=device) for _ in range(4)]
        x_list_10 = [1 for _ in range(4)]
        firstx_output_list_10 = []
        for i in range(len(first_input_list_10)):
            firstx_output_list_10.append(cuda_preprocess.firstx(first_input_list_10[i], x_list_10[i]))
        sigrid_hash_input_list_11 = [torch.randint(0, 100000, (args.batch_size,1), dtype=torch.int64, device=device) for _ in range(6)]
        sigrid_hash_length_list_11 = [65536 for _ in range(6)]
        for i in range(len(sigrid_hash_input_list_11)):
            cuda_preprocess.sigrid_hash(sigrid_hash_input_list_11[i], 0, sigrid_hash_length_list_11[i])
        clamp_input_list_12 = [torch.randint(0, 100000, (args.batch_size,1), dtype=torch.int64, device=device) for _ in range(3)]
        clamp_min_list_12 = [0 for _ in range(3)]
        clamp_max_list_12 = [65536 for _ in range(3)]
        for i in range(len(clamp_input_list_12)):
            cuda_preprocess.clamp(clamp_input_list_12[i], clamp_min_list_12[i], clamp_max_list_12[i])
        first_input_list_13 = [torch.randint(0, 100000, (args.batch_size, 1), dtype=torch.int64, device=device) for _ in range(4)]
        x_list_13 = [1 for _ in range(4)]
        firstx_output_list_13 = []
        for i in range(len(first_input_list_13)):
            firstx_output_list_13.append(cuda_preprocess.firstx(first_input_list_13[i], x_list_13[i]))
        sigrid_hash_input_list_14 = [torch.randint(0, 100000, (args.batch_size,1), dtype=torch.int64, device=device) for _ in range(3)]
        sigrid_hash_length_list_14 = [65536 for _ in range(3)]
        for i in range(len(sigrid_hash_input_list_14)):
            cuda_preprocess.sigrid_hash(sigrid_hash_input_list_14[i], 0, sigrid_hash_length_list_14[i])
        clamp_input_list_15 = [torch.randint(0, 100000, (args.batch_size,1), dtype=torch.int64, device=device) for _ in range(6)]
        clamp_min_list_15 = [0 for _ in range(6)]
        clamp_max_list_15 = [65536 for _ in range(6)]
        for i in range(len(clamp_input_list_15)):
            cuda_preprocess.clamp(clamp_input_list_15[i], clamp_min_list_15[i], clamp_max_list_15[i])
        first_input_list_16 = [torch.randint(0, 100000, (args.batch_size, 1), dtype=torch.int64, device=device) for _ in range(4)]
        x_list_16 = [1 for _ in range(4)]
        firstx_output_list_16 = []
        for i in range(len(first_input_list_16)):
            firstx_output_list_16.append(cuda_preprocess.firstx(first_input_list_16[i], x_list_16[i]))

        # insert one batch training data into the queue
        new_dist_input = torch.cat(sparse_output_tensor_list, dim=0).squeeze(1)
        dense_input = torch.cat(dense_output_tensor_list, dim=1)
        input_queue_list[this_rank].put((dense_input, new_dist_input, label_tensor))
        # =========================================== Generated Code End ===========================================
        dist.barrier(group=preprocess_group)
        print_once(this_rank, "finish cudf loading initalization")


    n_warm_up = 512
    n_loop = 1024
    loop_range = range(n_loop)
    if rank == 0:
        loop_range = tqdm(loop_range)
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for i in loop_range:
        if i == n_warm_up: 
            start_event.record()
        if if_train:
            dense_input, new_dist_input, label_tensor = input_queue_list[rank].get()
            batch.sparse_features._values = new_dist_input.clone()
            training_iter(sparse_optimizer, dense_optimizer, loss_fn, batch.sparse_features, dense_input, label_tensor, sharded_emts, mlp_layers, rank, train_group)
        else:
            # =========================================== Generated Code Start ===========================================
            df_both = cudf.read_parquet(both_file_name)
            for _ in range(2): # 4 times preprocessing intensity for plan 4
                # load next_batch_file
                int_ptr = []
                cat_ptr = []
                for i in range(len(dense_name)):
                    idx = i % 13
                    int_name = "int_{}".format(idx)
                    int_ptr.append(df_both[int_name].data.ptr)
                int_ptr.append(df_both[label_name].data.ptr)
                for i in range(len(sparse_name)):
                    idx = i % 26
                    cat_name = "cat_{}".format(idx)
                    cat_ptr.append(df_both[cat_name].data.ptr)
                # fill null
                sparse_tensor_list = []
                for i in range(len(sparse_name)):
                    sparse_tensor_list.append(cuda_preprocess.fill_null_int64(cat_ptr[i], args.batch_size))
                dense_label_tensor_list = []
                for i in range(len(int_ptr)):
                    dense_label_tensor_list.append(cuda_preprocess.fill_null_float(int_ptr[i], args.batch_size))
                label_tensor = dense_label_tensor_list[-1].squeeze(1)
                dense_tensor_list = dense_label_tensor_list[0:-1]
                # dense GPU part ==================
                for i in range(len(logit_input_list_1)):
                    cuda_preprocess.logit(logit_input_list_1[i], eps_list_1[i])
                for i in range(len(boxcox_input_list_2)):
                    cuda_preprocess.boxcox(boxcox_input_list_2[i], lambda_list_2[i])
                onehot_output_list_3 = []
                for i in range(len(onehot_input_list_3)):
                    onehot_output_list_3.append(cuda_preprocess.onehot(onehot_input_list_3[i], low_list_3[i], high_list_3[i], num_classes_list_3[i]))
                firstx_output_list_4 = []
                for i in range(len(first_input_list_4)):
                    firstx_output_list_4.append(cuda_preprocess.firstx(first_input_list_4[i], x_list_4[i]))
                # sparse GPU part ==================
                onehot_output_list_5 = []
                for i in range(len(onehot_input_list_5)):
                    onehot_output_list_5.append(cuda_preprocess.onehot(onehot_input_list_5[i], low_list_5[i], high_list_5[i], num_classes_list_5[i]))
                ngram_output_list_6 = []
                for i in range(len(ngram_input_list_6)):
                    ngram_output_list_6.append(cuda_preprocess.ngram(ngram_input_list_6[i], ngram_n_list_6[i]))
                firstx_output_list_7 = []
                for i in range(len(first_input_list_7)):
                    firstx_output_list_7.append(cuda_preprocess.firstx(first_input_list_7[i], x_list_7[i]))
                for i in range(len(sigrid_hash_input_list_8)):
                    cuda_preprocess.sigrid_hash(sigrid_hash_input_list_8[i], 0, sigrid_hash_length_list_8[i])
                for i in range(len(clamp_input_list_9)):
                    cuda_preprocess.clamp(clamp_input_list_9[i], clamp_min_list_9[i], clamp_max_list_9[i])
                firstx_output_list_10 = []
                for i in range(len(first_input_list_10)):
                    firstx_output_list_10.append(cuda_preprocess.firstx(first_input_list_10[i], x_list_10[i]))
                for i in range(len(sigrid_hash_input_list_11)):
                    cuda_preprocess.sigrid_hash(sigrid_hash_input_list_11[i], 0, sigrid_hash_length_list_11[i])
                for i in range(len(clamp_input_list_12)):
                    cuda_preprocess.clamp(clamp_input_list_12[i], clamp_min_list_12[i], clamp_max_list_12[i])
                firstx_output_list_13 = []
                for i in range(len(first_input_list_13)):
                    firstx_output_list_13.append(cuda_preprocess.firstx(first_input_list_13[i], x_list_13[i]))
                for i in range(len(sigrid_hash_input_list_14)):
                    cuda_preprocess.sigrid_hash(sigrid_hash_input_list_14[i], 0, sigrid_hash_length_list_14[i])
                for i in range(len(clamp_input_list_15)):
                    cuda_preprocess.clamp(clamp_input_list_15[i], clamp_min_list_15[i], clamp_max_list_15[i])
                firstx_output_list_16 = []
                for i in range(len(first_input_list_16)):
                    firstx_output_list_16.append(cuda_preprocess.firstx(first_input_list_16[i], x_list_16[i]))


                # =========================================== Generated Code End ===========================================
        
            # insert one batch training data into the queue
            new_dist_input = torch.cat(sparse_output_tensor_list, dim=0).squeeze(1)
            dense_input = torch.cat(dense_output_tensor_list, dim=1)
            input_queue_list[this_rank].put((dense_input, new_dist_input, label_tensor))
            
        dist.barrier()

    end_event.record()
    torch.cuda.synchronize()
    total_latency = start_event.elapsed_time(end_event)
    avg_latency = total_latency / (n_loop-n_warm_up) # ms
    throughput = 1000 / avg_latency # batch/s

    # free_memory
    if if_train:
        dense_input, new_dist_input, label_tensor = input_queue_list[rank].get()
        free_memory(dense_input, new_dist_input, label_tensor)

    dist.barrier()
    print_once(rank, "Clear all data in queue")
    print_once(rank, "avg_latency:{:.3f} ms, throughput:{:.3f} iter/s".format(avg_latency, throughput))
    if rank == 0:
        if args.MPS == True:
            file_name = "result/MPS-result_GPU-{}_Plan-{}_Batch-{}_{}.log".format(nDev, args.preprocessing_plan, args.batch_size, os.getpid())
        else:
            file_name = "result/Sequential-result_GPU-{}_Plan-{}_Batch-{}_{}.log".format(nDev, args.preprocessing_plan, args.batch_size, os.getpid())

        with open(file_name, 'w') as file:
            file.write("rank:{}, avg_latency:{:.3f} ms/iter, throughput:{:.3f} iters/s, total_throughput:{:.3f}".format(rank, avg_latency, throughput, throughput*nDev))


def training_iter(sparse_optimizer, dense_optimizer, loss_fn, sparse_features, dense_input, label_tensor, sharded_emts, mlp_layers, rank, work_group):
    sparse_optimizer.zero_grad()
    dense_optimizer.zero_grad()

    dist_input = sharded_emts.input_comm(sparse_features) # input communication
    sparse_feature = sharded_emts.forward_on_dist_input(dist_input) # forward
    logits = mlp_layers(dense_input, sparse_feature)
    
    loss = loss_fn(logits, label_tensor)
    loss.backward()
    average_gradients(mlp_layers, work_group)
    
    sparse_optimizer.step()
    dense_optimizer.step()


if __name__ == "__main__":
    random.seed(123)
    torch.manual_seed(123)

    args = get_args()
    nDev = args.nDev

    processes = []
    mp.set_start_method("spawn")

    queue_list = []
    for i in range(nDev):
        queue_list.append(mp.SimpleQueue())

    input_queue_list = []
    for i in range(nDev):
        input_queue_list.append(mp.Queue())

    nProcess = nDev * 2

    for i in range(nProcess):
        if i < nDev:
            if_train = True
        else:
            if_train = False

        p = mp.Process(target=train_and_preprocessing_process, args=(i, nDev, nProcess, args, queue_list, input_queue_list, if_train))
        processes.append(p)

    for p in processes:
        p.start()

    for p in processes:
        p.join()