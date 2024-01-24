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

from torcharrow_dataloader import torcharrow_dataloader

def data_process(args, queue, niter, device):
    parquet_directory = "/workspace/RAP/baseline_end_to_end/CPU_based_baseline/data_{}".format(args.batch_size)

    # train_dataloader = get_dataloader(args, backend, "train")
    # in_mem_dataloader = InMemoryDataLoader(train_dataloader, rank, nDev, 512)
    # batch = in_mem_dataloader.next()

    dataloader = torcharrow_dataloader(
        parquet_directory=parquet_directory,
        args=args,    
        device=device,
        preprocessing_plan=args.preprocessing_plan, # default plan
    )
    (dense_features, kjt, labels) = dataloader.next()

    # in_mem_dataloader.reset()
    for _ in range(niter):
        new_batch = dataloader.next_skip_convert_kjt()
        queue.put(1)
        # for _ in range(10):
        #     dense_features, kjt, labels = dataloader.next_skip_convert_kjt()


def train_process(rank, nDev, args) -> None:
    print_once(rank, "start init dist")
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    backend = 'nccl'
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend, rank=rank, world_size=nDev)
    dist.barrier()
    # get the default group
    default_group = dist.group.WORLD
    print_once(rank, "finish init dist")

    parquet_directory = "/workspace/RAP/baseline_end_to_end/CPU_based_baseline/data_{}".format(args.batch_size)

    if args.preprocessing_plan == 0 or args.preprocessing_plan == 1:
        args.num_embeddings_per_feature = args.num_embeddings_per_feature + [65536]
        args.cat_name = args.cat_name + ["bucketize_int_0"]
        args.nSparse = args.nSparse + 1
    elif args.preprocessing_plan == 2:
        args.nSparse = 26 * 4
        args.nDense = 13 * 4
        args.cat_name = ["cat_{}".format(i) for i in range(args.nSparse)]
        args.int_name = ["int_" + str(i) for i in range(args.nDense)]
        args.num_embeddings_per_feature = [65536 for _ in range(args.nSparse)]
    elif args.preprocessing_plan == 3:
        args.nSparse = 26 * 4
        args.nDense = 13 * 4
        args.cat_name = ["cat_{}".format(i) for i in range(args.nSparse)]
        args.int_name = ["int_" + str(i) for i in range(args.nDense)]
        args.num_embeddings_per_feature = [65536 for _ in range(args.nSparse)]

    train_dataloader = get_dataloader(args, backend, "train")
    in_mem_dataloader = InMemoryDataLoader(train_dataloader, rank, nDev, 512)
    batch = in_mem_dataloader.next()

    dataloader = torcharrow_dataloader(
        parquet_directory=parquet_directory,
        args=args,    
        device=device,
        preprocessing_plan=args.preprocessing_plan, # default plan
    )
    (dense_features, kjt, labels) = dataloader.next()

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
            local_world_size=get_local_size(),
            world_size=dist.get_world_size(),
            compute_device=device.type,
        ),
        batch_size=args.batch_size,
        storage_reservation=HeuristicalStorageReservation(percentage=0.01),
        constraints=sharding_constraints,
    )

    plan = planner.collective_plan(
        embedding_tables, get_default_sharders(), default_group
    )

    sharded_emts = ShardedEmbeddingTable(
        embedding_tables=embedding_tables, 
        plan=plan, 
        device=device, 
        group=default_group, 
        dim=args.embedding_dim,
        batch_size=args.batch_size, 
        table_names=args.cat_name
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

    # loss_fn = torch.nn.BCELoss(reduction="mean")
    loss_fn = torch.nn.BCEWithLogitsLoss()


    dist.barrier()
    print_once(rank, "finish model initalization")

    for _ in range(10):
        training_iter(sparse_optimizer, dense_optimizer, loss_fn, in_mem_dataloader, sharded_emts, mlp_layers, rank, default_group, batch)

    dist.barrier()
    print_once(rank, "finish warm up")

    total_niter = 128
    # nWorker = 16
    # nWorker_per_GPU = int(nWorker / nDev)
    nWorker_per_GPU = args.nWorker
    niter = int(total_niter / nWorker_per_GPU)

    queue = mp.Queue(64)
    processes = [mp.Process(target=data_process, args=(args, queue, niter, device)) for _ in range(nWorker_per_GPU)]

    # Start the processes
    for p in processes:
        p.start()

    iter_range = tqdm(range(total_niter)) if rank == 0 else range(total_niter)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    start_iter = 64

    for i in iter_range:
        if i == start_iter:
            start_event.record()
        get_data = queue.get()
        training_iter(sparse_optimizer, dense_optimizer, loss_fn, in_mem_dataloader, sharded_emts, mlp_layers, rank, default_group, batch)
    
    end_event.record()
    torch.cuda.synchronize()
    total_latency = start_event.elapsed_time(end_event)
    avg_latency = total_latency / (total_niter-start_iter) # ms
    throughput = 1000 / avg_latency # batch/s
    print("rank:{}, avg_latency:{:.3f} ms/iter, throughput:{:.3f} iters/s".format(rank, avg_latency, throughput))

    if rank == 0:
        file_name = "result/result_GPU-{}_Plan-{}_Batch-{}_{}.log".format(nDev, args.preprocessing_plan, args.batch_size, os.getpid())
        with open(file_name, 'w') as file:
            file.write("rank:{}, avg_latency:{:.3f} ms/iter, throughput:{:.3f} iters/s, total_throughput:{:.3f}".format(rank, avg_latency, throughput, throughput*nDev))
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    while not queue.empty():
        try:
            queue.get_nowait()
        except queue.Empty:
            break






def training_iter(sparse_optimizer, dense_optimizer, loss_fn, dataloader, sharded_emts, mlp_layers, rank, default_group, batch):
    sparse_optimizer.zero_grad()
    dense_optimizer.zero_grad()
    # dense_features, kjt, labels = dataloader.next_skip_convert_kjt() # Online data preprocessing (skip convert to kjt, reuse kjt from previous iteration)
    dense_features, kjt, labels = batch.dense_features, batch.sparse_features, batch.labels.float()

    sparse_feature = sharded_emts(kjt) # forward
    logits = mlp_layers(dense_features, sparse_feature)
    
    loss = loss_fn(logits, labels)
    loss.backward()
    average_gradients(mlp_layers, default_group)
    
    sparse_optimizer.step()
    dense_optimizer.step()
    dist.barrier()


if __name__ == "__main__":
    args = get_args()
    nDev = args.nDev

    processes = []
    mp.set_start_method("spawn")

    for rank in range(nDev):
        p = mp.Process(target=train_process, args=(rank, nDev, args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
