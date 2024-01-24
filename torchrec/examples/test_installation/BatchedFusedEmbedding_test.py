#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import itertools
import os
import sys
from typing import Iterator, List

import torch
import torchmetrics as metrics
from pyre_extensions import none_throws
from torch import distributed as dist
from torch.utils.data import DataLoader
# from torchrec import EmbeddingBagCollection
from torchrec.datasets.criteo import DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES
from torchrec.datasets.utils import Batch
from torchrec.distributed import TrainPipelineSparseDist
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.optim.keyed import KeyedOptimizerWrapper
import torch.multiprocessing as mp

from torchrec.distributed.comm import get_local_size
from torchrec.distributed.shard import shard
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.storage_reservations import (
    HeuristicalStorageReservation,
)
from torchrec.distributed.model_parallel import (
    DistributedModelParallel,
    get_default_sharders,
)
from dlrm_models import DLRMTrain, DLRM

from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.distributed.planner.types import ParameterConstraints

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
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


parser = argparse.ArgumentParser(description="torchrec dlrm example trainer")
parser.add_argument(
    "--nDev",
    type=int,
    default=4,
    help="number of GPUs",
)


def print_once(rank, msg):
    if rank == 0:
        print(msg)


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
    print_once(rank, "finish init dist")


    table_name = ["table_0", "table_1", "table_2", "table_3"]
    num_embeddings_per_feature = [4, 4, 4, 4]


    eb_configs = [
        EmbeddingBagConfig(
            name=f"t_{feature_name}",
            embedding_dim=4,
            num_embeddings = num_embeddings_per_feature[feature_idx],
            feature_names=[feature_name],
        )
        for feature_idx, feature_name in enumerate(table_name)
    ]

    # print_once(rank, eb_configs)

    # print_once(rank, dist.group)


    model = EmbeddingBagCollection(
        tables=eb_configs,
        device=device,
    )

    sharding_constraints = {
        f"t_{feature_name}": ParameterConstraints(
        sharding_types=[ShardingType.DATA_PARALLEL.value],  # TABLE_WISE, ROW_WISE, COLUMN_WISE, DATA_PARALLEL
        ) for feature_idx, feature_name in enumerate(table_name)
    }

    planner = EmbeddingShardingPlanner(
        topology=Topology(
            local_world_size=get_local_size(),
            world_size=dist.get_world_size(),
            compute_device=device.type,
        ),
        batch_size=3,
        storage_reservation=HeuristicalStorageReservation(percentage=0.01),
        constraints=sharding_constraints,
    )

    plan = planner.collective_plan(
        model, get_default_sharders(), dist.GroupMember.WORLD
    )

    # get the default group
    default_group = dist.group.WORLD

    # print the group size
    # print(default_group.size())

    sharded_model = shard(
        module=model,
        env=ShardingEnv.from_process_group(default_group),
        plan=plan.get_plan_for_module(""),
        # sharder=sharder,
        device=device,
    )

    # if rank == 0:
        # print(sharded_model)

    sharded_model_optimizer = torch.optim.SGD(
        sharded_model.parameters(), lr=0.5
    )

    kjt_input_per_rank = [  # noqa
        KeyedJaggedTensor.from_lengths_sync(
            keys=["table_0", "table_1", "table_2", "table_3"],
            values=torch.LongTensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]),
            lengths=torch.LongTensor([2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1]),
        ),
        KeyedJaggedTensor.from_lengths_sync(
            keys=["table_0", "table_1", "table_2", "table_3"],
            values=torch.LongTensor([3, 2, 1, 2, 0, 1, 2, 3, 2, 3, 2, 3, 2, 1, 2, 0, 1, 2, 3, 2, 3, 2]),
            lengths=torch.LongTensor([2, 2, 4, 2, 0, 1, 2, 2, 4, 2, 0, 1]),
        ),
        KeyedJaggedTensor.from_lengths_sync(
            keys=["table_0", "table_1", "table_2", "table_3"],
            values=torch.LongTensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]),
            lengths=torch.LongTensor([2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1]),
        ),
        KeyedJaggedTensor.from_lengths_sync(
            keys=["table_0", "table_1", "table_2", "table_3"],
            values=torch.LongTensor([3, 2, 1, 2, 0, 1, 2, 3, 2, 3, 2, 3, 2, 1, 2, 0, 1, 2, 3, 2, 3, 2]),
            lengths=torch.LongTensor([2, 2, 4, 2, 0, 1, 2, 2, 4, 2, 0, 1]),
        ),
    ]

    kjt_input_per_rank = [kjt.to(device) for kjt in kjt_input_per_rank]

    sharded_model_optimizer.zero_grad()
    sharded_model_pred_jts = sharded_model(kjt_input_per_rank[rank]).wait()

    # sharded_model_pred = torch.cat(
    #     [sharded_model_pred_jts[feature].values() for feature in table_name]
    # )

    B = kjt_input_per_rank[rank].stride()
    sparse = sharded_model_pred_jts.to_dict()
    sparse_values = []
    for name in table_name:
        sparse_values.append(sparse[name])

    result = torch.cat(sparse_values, dim=1).reshape(B, len(table_name), 4)

    if rank == 0:
        print(result.sum())

    loss_fn = torch.nn.MSELoss(reduction="mean")
    label = torch.tensor([0.0]).to(rank)
    loss = loss_fn(result.sum(), label[0]) 
    loss.backward() 

    sharded_model_optimizer.step()

    # result.sum().backward()

    # if rank == 0:
    #     params = sharded_model.parameters()
    #     for param in params:
    #         print(param.grad)
        # print(*sharded_model.parameters())

    # if rank == 0:
        # print(plan)
        # print("original input:")
        # print(kjt_input_per_rank[rank])

        # print("result:")
        # print(sharded_model_pred_jts)

        # print("sum_result:")
        # print(result)
        
        # print("updated parameters:")
        # print(*sharded_model.parameters())


    sharded_model_pred_jts = sharded_model(kjt_input_per_rank[rank]).wait()
    B = kjt_input_per_rank[rank].stride()
    sparse = sharded_model_pred_jts.to_dict()
    sparse_values = []
    for name in table_name:
        sparse_values.append(sparse[name])
    result = torch.cat(sparse_values, dim=1).reshape(B, len(table_name), 4)

    if rank == 0:
        print(result.sum())
   
  


if __name__ == "__main__":
    args = parser.parse_args()
    nDev = args.nDev

    processes = []
    mp.set_start_method("spawn")

    for rank in range(nDev):
        p = mp.Process(target=train_process, args=(rank, nDev, args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
