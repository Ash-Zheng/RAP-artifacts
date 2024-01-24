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
from dlrm_models import DLRMTrain, DLRM
from data.dlrm_dataloader import get_dataloader

from torchrec.distributed.types import (
    ParameterSharding,
    ShardingPlan,
    EnumerableShardingSpec,
    ShardMetadata,
    ShardingType
)
from tqdm import tqdm
import time

from torchrec.distributed.dist_data import (
    KJTAllToAll,
    PooledEmbeddingsAllToAll,
)



parser = argparse.ArgumentParser(description="torchrec dlrm example trainer")
parser.add_argument(
    "--epochs", type=int, default=1, help="number of epochs to train"
)
parser.add_argument(
    "--batch_size", type=int, default=8192, help="local batch size to use for training"
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=1,
    help="number of dataloader workers",
)
parser.add_argument(
    "--num_embeddings",
    type=int,
    default=30000001,
    help="max_ind_size. The number of embeddings in each embedding table. Defaults"
    " to 100_000 if num_embeddings_per_feature is not supplied.",
)
parser.add_argument(
    "--num_embeddings_per_feature",
    type=str,
    # default=None,
    default="45833188,36746,17245,7413,20243,3,7114,1441,62,29275261,1572176,345138,10,2209,11267,128,4,974,14,48937457,11316796,40094537,452104,12606,104,35",
    help="Comma separated max_ind_size per sparse feature. The number of embeddings"
    " in each embedding table. 26 values are expected for the Criteo dataset.",
)
# Reduced_Terabyte: "10000000,36746,17245,7413,20243,3,7114,1441,62,10000000,1572176,345138,10,2209,11267,128,4,974,14,10000000,10000000,10000000,452104,12606,104,35"
# Terabyte: "45833188,36746,17245,7413,20243,3,7114,1441,62,29275261,1572176,345138,10,2209,11267,128,4,974,14,48937457,11316796,40094537,452104,12606,104,35"
# Kaggle: "1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194,27,14992,5461306,10,5652,2173,4,7046547,18,15,286181,105,142572"

parser.add_argument(
    "--dense_arch_layer_sizes",
    type=str,
    default="512,256,128",
    help="Comma separated layer sizes for dense arch.",
)
parser.add_argument(
    "--over_arch_layer_sizes",
    type=str,
    default="1024,1024,512,256,1",
    help="Comma separated layer sizes for over arch.",
)
parser.add_argument(
    "--embedding_dim",
    type=int,
    default=128,
    help="Size of each embedding.",
)
parser.add_argument(
    "--undersampling_rate",
    type=float,
    help="Desired proportion of zero-labeled samples to retain (i.e. undersampling zero-labeled rows)."
    " Ex. 0.3 indicates only 30pct of the rows with label 0 will be kept."
    " All rows with label 1 will be kept. Value should be between 0 and 1."
    " When not supplied, no undersampling occurs.",
)
parser.add_argument(
    "--seed",
    type=float,
    help="Random seed for reproducibility.",
)
parser.add_argument(
    "--pin_memory",
    dest="pin_memory",
    action="store_true",
    default=True,
    help="Use pinned memory when loading data.",
)
parser.add_argument(
    "--in_memory_binary_criteo_path",
    type=str,
    # default="/home/yuke_wang/zheng/torchrec/processed",
    default=None,
    help="Path to a folder containing the binary (npy) files for the Criteo dataset."
    " When supplied, InMemoryBinaryCriteoIterDataPipe is used.",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=0.1,
    help="Learning rate.",
)
parser.add_argument(
    "--shuffle_batches",
    type=bool,
    default=False,
    help="Shuffle each batch during training.",
)
parser.add_argument(
    "--nDev",
    type=int,
    default=4,
    help="number of GPUs",
)
parser.add_argument(
    "--single_table",
    type=bool,
    default=False,
    help="if_single_table",
)

# # OSS import
# try:
#     # pyre-ignore[21]
#     # @manual=//torchrec/github/examples/dlrm/data:dlrm_dataloader
#     from data.dlrm_dataloader import get_dataloader, STAGES

#     # pyre-ignore[21]
#     # @manual=//torchrec/github/examples/dlrm/modules:dlrm_train
#     from modules.dlrm_train import DLRMTrain
# except ImportError:
#     pass

# # internal import
# try:
#     from .data.dlrm_dataloader import (  # noqa F811
#         get_dataloader,
#         STAGES,
#     )
#     from .modules.dlrm_train import DLRMTrain  # noqa F811
# except ImportError:
#     pass

# from .data.dlrm_dataloader import (  # noqa F811
#     get_dataloader,
#     STAGES,
# )
# from .modules.dlrm_train import DLRMTrain  # noqa F811
# from data.dlrm_dataloader import get_dataloader, STAGES
# from modules.dlrm_train import DLRMTrain

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

    args.num_embeddings_per_feature = list(
        map(int, args.num_embeddings_per_feature.split(","))
    )
    args.dense_arch_layer_sizes = list(
        map(int, args.dense_arch_layer_sizes.split(","))
    )
    args.over_arch_layer_sizes = list(
        map(int, args.over_arch_layer_sizes.split(","))
    )
    train_dataloader = get_dataloader(args, backend, "train")
    dataiter = iter(train_dataloader)

    # load random data into GPU memory
    input_data = []
    nRandom = 500
    for i in range(nRandom):
        batch = next(dataiter).to(rank)
        input_data.append(batch)


    dist.barrier()
    print_once(rank, "finish load data into GPU memory")

    eb_configs = [
        EmbeddingBagConfig(
            name=f"t_{feature_name}",
            embedding_dim=args.embedding_dim,
            # num_embeddings=none_throws(args.num_embeddings_per_feature)[feature_idx]
            # if args.num_embeddings is None
            # else args.num_embeddings,
            num_embeddings = args.num_embeddings_per_feature[feature_idx],
            feature_names=[feature_name],
        )
        for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES)
    ]

    print_once(rank, "finish eb_configs")

    dlrm_model = DLRM(
        embedding_bag_collection=EmbeddingBagCollection(
            tables=eb_configs, device=torch.device("meta")
        ),
        dense_in_features=len(DEFAULT_INT_NAMES),
        dense_arch_layer_sizes=args.dense_arch_layer_sizes,
        over_arch_layer_sizes=args.over_arch_layer_sizes,
        dense_device=device,
    )

    train_model = DLRMTrain(dlrm_model)

    print_once(rank, "finish dlrm_model")

    sharding_constraints = {
        f"t_{feature_name}": ParameterConstraints(
        sharding_types=[ShardingType.COLUMN_WISE.value],  # TABLE_WISE, ROW_WISE, COLUMN_WISE, DATA_PARALLEL
        ) for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES)
    }

    planner = EmbeddingShardingPlanner(
        topology=Topology(
            local_world_size=get_local_size(),
            world_size=dist.get_world_size(),
            compute_device=device.type,
        ),
        batch_size=args.batch_size,
        # If experience OOM, increase the percentage. see
        # https://pytorch.org/torchrec/torchrec.distributed.planner.html#torchrec.distributed.planner.storage_reservations.HeuristicalStorageReservation
        storage_reservation=HeuristicalStorageReservation(percentage=0.01),
        constraints=sharding_constraints,
    )

    plan = planner.collective_plan(
        train_model, get_default_sharders(), dist.GroupMember.WORLD
    )

    print_once(rank, "finish planner")

    model = DistributedModelParallel(
        module=train_model,
        device=device,
        plan=plan,
    )

    if rank == 0:
        print(model)
        print(plan)

    # optimizer = KeyedOptimizerWrapper(
    #     dict(model.named_parameters()),
    #     lambda params: torch.optim.SGD(params, lr=args.learning_rate),
    # )

    # print_once(rank, "finish DistributedModelParallel")

    # optimizer = KeyedOptimizerWrapper(
    #     dict(model.named_parameters()),
    #     lambda params: torch.optim.SGD(params, lr=args.learning_rate),
    # )
    
    # print_once(rank, "start training")
    # dist.barrier()

    # input_iter = iter(input_data)
    # nIters = 1000
    # if rank == 0:
    #     for i in tqdm(range(nIters)):
    #         batch = next(input_iter)
    #         if i % nRandom == nRandom-1:
    #             input_iter = iter(input_data)

    #         optimizer.zero_grad()
    #         loss, (d_loss, logits, labels) = model.forward(batch)

    #         torch.sum(loss, dim=0).backward()
    #         optimizer.step()
    #         dist.barrier()
    # else:
    #     for i in range(nIters):
    #         batch = next(input_iter)
    #         if i % nRandom == nRandom-1:
    #             input_iter = iter(input_data)

    #         optimizer.zero_grad()
    #         loss, (d_loss, logits, labels) = model.forward(batch)

    #         torch.sum(loss, dim=0).backward()
    #         optimizer.step()
    #         dist.barrier()
  


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
