#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys
import os
from typing import List, Tuple

import torch
from fbgemm_gpu.split_table_batched_embeddings_ops import EmbeddingLocation
import ebc_benchmarks_utils
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.modules.fused_embedding_modules import FusedEmbeddingBagCollection
from torch.utils.cpp_extension import load
from tqdm import tqdm
import nvtx

import torch.distributed as dist
import torch.multiprocessing as mp

parser = argparse.ArgumentParser(description="TorchRec ebc benchmarks")
parser.add_argument(
    "--cpu_only",
    action="store_true",
    default=False,
    help="specify whether to use cpu",
)
parser.add_argument(
    "--mode",
    type=str,
    default="ebc_comparison_dlrm",
    help="specify 'ebc_comparison_dlrm', 'ebc_comparison_scaling' or 'fused_ebc_uvm'",
)
parser.add_argument(
    "--type",
    type=str,
    default="embeddingbag", # embeddingbag fused_table
    help="specify 'embeddingbag', 'fused_table'",
)
parser.add_argument(
    "--pooling_factor",
    type=int,
    default=1, 
    help="pooling factor: criteo=1",
)

# Reference: https://github.com/facebookresearch/dlrm/blob/main/torchrec_dlrm/README.MD
DLRM_NUM_EMBEDDINGS_PER_FEATURE = [
    45833188,
    36746,
    17245,
    7413,
    20243,
    3,
    7114,
    1441,
    62,
    29275261,
    1572176,
    345138,
    10,
    2209,
    11267,
    128,
    4,
    974,
    14,
    48937457,
    11316796,
    40094537,
    452104,
    12606,
    104,
    35,
]

def time_wrap():
    torch.cuda.synchronize()
    return time.time()

def get_shrunk_dlrm_num_embeddings(reduction_degree: int) -> List[int]:
    return [
        num_emb if num_emb < 10000000 else int(num_emb / reduction_degree)
        for num_emb in DLRM_NUM_EMBEDDINGS_PER_FEATURE
    ]

def queue_wait(queue):
     while queue.empty():
        time.sleep(0.000001)
        a = queue.get()

def main_fused_table(rank, nProcess, args, queue) -> None:
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("gloo", rank=rank, world_size=nProcess)
    dist.barrier()
    print("finish init:", rank)
    batch_size = 4096
    device = torch.device(0)

    if rank == 0: # embedding
        embedding_bag_configs: List[EmbeddingBagConfig] = [
                EmbeddingBagConfig(
                    name=f"ebc_{idx}",
                    embedding_dim=128,
                    num_embeddings=num_embeddings,
                    feature_names=[f"ebc_{idx}_feat_1"],
                )
                for idx, num_embeddings in enumerate(
                    get_shrunk_dlrm_num_embeddings(64)
                )
            ]

        pooling_factors = {} # pooling factors of DLRM, set to 64
        for table in embedding_bag_configs:
            for feature_name in table.feature_names:
                pooling_factors[feature_name] = args.pooling_factor

        dataset = ebc_benchmarks_utils.get_random_dataset(
            batch_size=batch_size,
            num_batches=10,
            num_dense_features=13,
            embedding_bag_configs=embedding_bag_configs,
            pooling_factors=pooling_factors
        )

        dataset_iter = iter(dataset)
        input_batch = next(dataset_iter)
        sparse_features = input_batch.sparse_features.to(device)
    
        # EBC with fused optimizer backed by fbgemm SplitTableBatchedEmbeddingBagsCodegen
        fused_ebc = FusedEmbeddingBagCollection(
            tables=embedding_bag_configs,
            optimizer_type=torch.optim.SGD,
            optimizer_kwargs={"lr": 0.02},
            device=device,
        )

        fused_pooled_embeddings = fused_ebc(sparse_features)
        fused_vals = []
        for _name, param in fused_pooled_embeddings.to_dict().items():
            fused_vals.append(param)
        torch.cat(fused_vals, dim=1).sum().backward()

    else: # preprocess
        file_name = "generated_data/first_{}.parquet".format(batch_size)
        cuda_preprocess = load(name="gpu_operators", sources=[
        "cuda_operators/cuda_wrap.cpp", 
        "cuda_operators/gpu_operators.cu", 
        ], verbose=False)

        cuda_preprocess.init_cuda(0)

        sparse_tensors = [torch.randint(100000000, (10, 4096), dtype=torch.int64).to(device) for _ in range(10)]
        for i in range(10):
            cuda_preprocess.sigrid_hash(sparse_tensors[i], 0, 65536)


    # ==================================== Lookup time profile =======================================
    for _ in range(100):
        if rank == 0:
            fused_pooled_embeddings = fused_ebc(sparse_features)
            fused_vals = []
            for _name, param in fused_pooled_embeddings.to_dict().items():
                fused_vals.append(param)
            torch.cat(fused_vals, dim=1).sum().backward()
        # else:
        #     for i in range(1):
        #         cuda_preprocess.sigrid_hash(sparse_tensors[i], 0, 65536)
        dist.barrier()
        torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    repeat = 1000

    latency_list = []
    for out_loop in range(20):
        start.record()
        for _ in range(repeat):
            if rank == 0:
                fused_pooled_embeddings = fused_ebc(sparse_features)
                fused_vals = []
                for _name, param in fused_pooled_embeddings.to_dict().items():
                    fused_vals.append(param)
                torch.cat(fused_vals, dim=1).sum().backward()
            # else:
            #     for i in range(1):
            #         cuda_preprocess.sigrid_hash(sparse_tensors[i], 0, 65536)
            dist.barrier()
        end.record()
        dist.barrier()
        torch.cuda.synchronize()
        total_time = start.elapsed_time(end)
        avg_latency_lookup = total_time/float(repeat)

        communicate_tensor = torch.tensor([avg_latency_lookup])
        tensor_list = [torch.tensor([0.0]) for _ in range(nProcess)]
        dist.all_gather(tensor_list, communicate_tensor)

        if tensor_list[0].item() > tensor_list[1].item():
            avg_latency_lookup = tensor_list[0].item()
        else:
            avg_latency_lookup = tensor_list[1].item()
        latency_list.append(avg_latency_lookup)

    latency_list.sort()
    latency_list = latency_list[5:15]
    avg_latency_lookup = sum(latency_list)/len(latency_list)
    print("avg_latency_lookup:{:.3f} ms".format(avg_latency_lookup))

    # ==================================== Max Preprocessing Iters profile =======================================
    SM_param = 1
    pre_iters = 1
    pre_iters_upper_bound = 256
    sparse_tensors = [torch.randint(100000000, (SM_param, batch_size), dtype=torch.int64).to(device) for _ in range(pre_iters_upper_bound)]

    # print("start search for max pre_iters")
    pre_iters_record = pre_iters
    result_list = []
    for out_loop in range(10):
        # if rank == 0:
            # print("iter:{} for max_pre_iters:{}".format(out_loop, pre_iters))
        pre_pre_iters = pre_iters_record
        pre_iters = pre_iters_record * 2
        communicate_tensor = torch.tensor([0])
        tensor_list = [torch.tensor([0]) for _ in range(nProcess)]
        while(communicate_tensor.item() == 0):
            if pre_iters >= pre_iters_upper_bound:
                break
            dist.barrier()
            start.record()
            for _ in range(repeat):
                if rank == 0:
                    fused_pooled_embeddings = fused_ebc(sparse_features)
                    fused_vals = []
                    for _name, param in fused_pooled_embeddings.to_dict().items():
                        fused_vals.append(param)
                    torch.cat(fused_vals, dim=1).sum().backward()
                else:
                    for i in range(pre_iters):
                        cuda_preprocess.sigrid_hash(sparse_tensors[i], 0, 65536)
                dist.barrier()
            end.record()
            dist.barrier()
            torch.cuda.synchronize()
            total_time = start.elapsed_time(end)
            avg_latency = total_time/float(repeat)
            performance_decrease = (avg_latency - avg_latency_lookup)/avg_latency_lookup
         
            if performance_decrease >= 0.07:
                communicate_tensor[0] = 1
            else:
                communicate_tensor[0] = 0
            dist.all_gather(tensor_list, communicate_tensor)
            comm_result = sum(tensor_list)

            if comm_result.item() > 0: # stop
                communicate_tensor[0] = 1
                break
            else: # continue increase pre_iters
                communicate_tensor[0] = 0
                pre_pre_iters = pre_iters
                pre_iters = pre_iters * 2
       
        # print("binary search for best pre_iters")
        up_bound = pre_iters
        lower_bound = pre_pre_iters
        while(up_bound - lower_bound > 2):
            pre_iters = int((up_bound + lower_bound)/2)
            dist.barrier()
            start.record()
            for _ in range(repeat):
                if rank == 0:
                    fused_pooled_embeddings = fused_ebc(sparse_features)
                    fused_vals = []
                    for _name, param in fused_pooled_embeddings.to_dict().items():
                        fused_vals.append(param)
                    torch.cat(fused_vals, dim=1).sum().backward()
                else:
                    for i in range(pre_iters):
                        cuda_preprocess.sigrid_hash(sparse_tensors[i], 0, 65536)
                dist.barrier()
            end.record()
            dist.barrier()
            torch.cuda.synchronize()
            total_time = start.elapsed_time(end)
            avg_latency = total_time/float(repeat)
            performance_decrease = (avg_latency - avg_latency_lookup)/avg_latency_lookup
         
            if performance_decrease >= 0.07:
                communicate_tensor[0] = 1
            else:
                communicate_tensor[0] = 0
            dist.all_gather(tensor_list, communicate_tensor)
            comm_result = sum(tensor_list)

            if comm_result.item() > 0: # stop
                up_bound = pre_iters - 1
            else:
                lower_bound = pre_iters + 1
        pre_iters = int(lower_bound * 0.8) 
        if pre_iters < 1:
            pre_iters = 1
        result_list.append(pre_iters)
       
    result_list.sort()
    result_list = result_list[2:8]
    pre_iters = int(sum(result_list)/6)
 
    start.record()
    for _ in range(repeat):
        if rank == 0:
            fused_pooled_embeddings = fused_ebc(sparse_features)
            fused_vals = []
            for _name, param in fused_pooled_embeddings.to_dict().items():
                fused_vals.append(param)
            torch.cat(fused_vals, dim=1).sum().backward()
        else:
            for i in range(pre_iters):
                cuda_preprocess.sigrid_hash(sparse_tensors[i], 0, 65536)
        dist.barrier()
    end.record()
    dist.barrier()
    torch.cuda.synchronize()
    total_time = start.elapsed_time(end)
    avg_latency = total_time/float(repeat)

    print("max_pre_iter avg_latency:{:.3f} ms, max_iters:{}".format(avg_latency, pre_iters))



    # ==================================== Max Preprocessing SM_param profile =======================================
    SM_param_record = SM_param
    result_list = []
    for out_loop in range(10):
        # if rank == 0:
            # print("iter:{} for max_SM:{}".format(out_loop, SM_param))

        pre_SM = SM_param_record
        SM_param = SM_param_record * 2
        communicate_tensor = torch.tensor([0])
        tensor_list = [torch.tensor([0]) for _ in range(nProcess)]
        while(communicate_tensor.item() == 0):
            if SM_param >= pre_iters_upper_bound:
                break
            sparse_tensors = [torch.randint(100000000, (SM_param, batch_size), dtype=torch.int64).to(device) for _ in range(pre_iters)]
            dist.barrier()
            start.record()
            for _ in range(repeat):
                if rank == 0:
                    fused_pooled_embeddings = fused_ebc(sparse_features)
                    fused_vals = []
                    for _name, param in fused_pooled_embeddings.to_dict().items():
                        fused_vals.append(param)
                    torch.cat(fused_vals, dim=1).sum().backward()
                else:
                    for i in range(pre_iters):
                        cuda_preprocess.sigrid_hash(sparse_tensors[i], 0, 65536)
                dist.barrier()
            end.record()
            dist.barrier()
            torch.cuda.synchronize()
            total_time = start.elapsed_time(end)
            avg_latency = total_time/float(repeat)
            performance_decrease = (avg_latency - avg_latency_lookup)/avg_latency_lookup
            # print("SM_param:{}, avg_latency:{:.3f} ms, performance_decrease:{:.3f}".format(SM_param, avg_latency, performance_decrease))
         
            if performance_decrease >= 0.07:
                communicate_tensor[0] = 1
            else:
                communicate_tensor[0] = 0
            dist.all_gather(tensor_list, communicate_tensor)
            comm_result = sum(tensor_list)

            if comm_result.item() > 0: # stop
                communicate_tensor[0] = 1
                break
            else: # continue increase pre_iters
                communicate_tensor[0] = 0
                pre_SM = SM_param
                SM_param = SM_param * 2
       
        # print("binary search for best pre_iters")
        up_bound = SM_param
        lower_bound = pre_SM
        while(up_bound - lower_bound > 2):
            SM_param = int((up_bound + lower_bound)/2)
            sparse_tensors = [torch.randint(100000000, (SM_param, batch_size), dtype=torch.int64).to(device) for _ in range(pre_iters)]
            dist.barrier()
            start.record()
            for _ in range(repeat):
                if rank == 0:
                    fused_pooled_embeddings = fused_ebc(sparse_features)
                    fused_vals = []
                    for _name, param in fused_pooled_embeddings.to_dict().items():
                        fused_vals.append(param)
                    torch.cat(fused_vals, dim=1).sum().backward()
                else:
                    for i in range(pre_iters):
                        cuda_preprocess.sigrid_hash(sparse_tensors[i], 0, 65536)
                dist.barrier()
            end.record()
            dist.barrier()
            torch.cuda.synchronize()
            total_time = start.elapsed_time(end)
            avg_latency = total_time/float(repeat)
            performance_decrease = (avg_latency - avg_latency_lookup)/avg_latency_lookup
            # print("SM_param:{}, avg_latency:{:.3f} ms, performance_decrease:{:.3f}".format(SM_param, avg_latency, performance_decrease))
         
            if performance_decrease >= 0.07:
                communicate_tensor[0] = 1
            else:
                communicate_tensor[0] = 0
            dist.all_gather(tensor_list, communicate_tensor)
            comm_result = sum(tensor_list)

            if comm_result.item() > 0: # stop
                up_bound = SM_param - 1
            else:
                lower_bound = SM_param + 1
        SM_param = int(lower_bound * 0.8) 
        if SM_param <= 0:
            SM_param = 1
        result_list.append(SM_param)
       
    result_list.sort()
    result_list = result_list[2:8]
    SM_param = int(sum(result_list)/6)
    
    sparse_tensors = [torch.randint(100000000, (SM_param, batch_size), dtype=torch.int64).to(device) for _ in range(pre_iters)]
    start.record()
    for _ in range(repeat):
        if rank == 0:
            fused_pooled_embeddings = fused_ebc(sparse_features)
            fused_vals = []
            for _name, param in fused_pooled_embeddings.to_dict().items():
                fused_vals.append(param)
            torch.cat(fused_vals, dim=1).sum().backward()
        else:
            for i in range(pre_iters):
                cuda_preprocess.sigrid_hash(sparse_tensors[i], 0, 65536)
        dist.barrier()
    end.record()
    dist.barrier()
    torch.cuda.synchronize()
    total_time = start.elapsed_time(end)
    avg_latency = total_time/float(repeat)
    
    print("max_SM_para avg_latency:{:.3f} ms, SM_param:{}".format(avg_latency, SM_param))

    performance_decrease = (avg_latency - avg_latency_lookup)/avg_latency_lookup
    exp_type = args.type
    pooling_factor = args.pooling_factor
    nTB = int(SM_param * batch_size * pre_iters / 128)
    if rank == 0:
        with open("output/overlapping.out", "a") as f:
            f.write("Criteo fused table: batch_size:{}, exp_type:{}, pooling_factor:{}, pre_iter:{}, SM_param:{}, performance_decrease:{:.3f}%, nTB:{}\n".format(batch_size, exp_type, pooling_factor, pre_iters, SM_param, performance_decrease*100, nTB))
    
def main_embedding_bag(rank, nProcess, args, queue) -> None:
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("gloo", rank=rank, world_size=nProcess)
    dist.barrier()
    print("finish init:", rank)
    batch_size = 4096
    device = torch.device(0)

    if rank == 0: # embedding
        embedding_bag_configs: List[EmbeddingBagConfig] = [
                EmbeddingBagConfig(
                    name=f"ebc_{idx}",
                    embedding_dim=128,
                    num_embeddings=num_embeddings,
                    feature_names=[f"ebc_{idx}_feat_1"],
                )
                for idx, num_embeddings in enumerate(
                    get_shrunk_dlrm_num_embeddings(64)
                )
            ]

        pooling_factors = {} # pooling factors of DLRM, set to 64
        for table in embedding_bag_configs:
            for feature_name in table.feature_names:
                pooling_factors[feature_name] = args.pooling_factor

        dataset = ebc_benchmarks_utils.get_random_dataset(
            batch_size=batch_size,
            num_batches=10,
            num_dense_features=13,
            embedding_bag_configs=embedding_bag_configs,
            pooling_factors=pooling_factors
        )

        dataset_iter = iter(dataset)
        input_batch = next(dataset_iter)
        sparse_features = input_batch.sparse_features.to(device)
    
        # EBC with fused optimizer backed by fbgemm SplitTableBatchedEmbeddingBagsCodegen
        ebc = EmbeddingBagCollection(
            tables=embedding_bag_configs,
            device=device,
            )
        optimizer = torch.optim.SGD(ebc.parameters(), lr=0.02)

        pooled_embeddings = ebc(sparse_features)
        optimizer.zero_grad()

        vals = []
        for _name, param in pooled_embeddings.to_dict().items():
            vals.append(param)
        torch.cat(vals, dim=1).sum().backward()
        optimizer.step()

    else: # preprocess
        file_name = "generated_data/first_{}.parquet".format(batch_size)
        cuda_preprocess = load(name="gpu_operators", sources=[
        "cuda_operators/cuda_wrap.cpp", 
        "cuda_operators/gpu_operators.cu", 
        ], verbose=False)

        cuda_preprocess.init_cuda(0)

        sparse_tensors = [torch.randint(100000000, (10, 4096), dtype=torch.int64).to(device) for _ in range(10)]
        for i in range(10):
            cuda_preprocess.sigrid_hash(sparse_tensors[i], 0, 65536)


    # ==================================== Lookup time profile =======================================
    for _ in range(100):
        if rank == 0:
            pooled_embeddings = ebc(sparse_features)
            optimizer.zero_grad()

            vals = []
            for _name, param in pooled_embeddings.to_dict().items():
                vals.append(param)
            torch.cat(vals, dim=1).sum().backward()
            optimizer.step()
        # else:
        #     for i in range(1):
        #         cuda_preprocess.sigrid_hash(sparse_tensors[i], 0, 65536)
        dist.barrier()
        torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    repeat = 1000

    latency_list = []
    for out_loop in range(20):
        start.record()
        for _ in range(repeat):
            if rank == 0:
                pooled_embeddings = ebc(sparse_features)
                optimizer.zero_grad()

                vals = []
                for _name, param in pooled_embeddings.to_dict().items():
                    vals.append(param)
                torch.cat(vals, dim=1).sum().backward()
                optimizer.step()
            # else:
            #     for i in range(1):
            #         cuda_preprocess.sigrid_hash(sparse_tensors[i], 0, 65536)
            dist.barrier()
        end.record()
        dist.barrier()
        torch.cuda.synchronize()
        total_time = start.elapsed_time(end)
        avg_latency_lookup = total_time/float(repeat)

        communicate_tensor = torch.tensor([avg_latency_lookup])
        tensor_list = [torch.tensor([0.0]) for _ in range(nProcess)]
        dist.all_gather(tensor_list, communicate_tensor)

        if tensor_list[0].item() > tensor_list[1].item():
            avg_latency_lookup = tensor_list[0].item()
        else:
            avg_latency_lookup = tensor_list[1].item()
        latency_list.append(avg_latency_lookup)

    latency_list.sort()
    latency_list = latency_list[5:15]
    avg_latency_lookup = sum(latency_list)/len(latency_list)
    print("avg_latency_lookup:{:.3f} ms".format(avg_latency_lookup))

    # ==================================== Max Preprocessing Iters profile =======================================
    SM_param = 1
    pre_iters = 1
    pre_iters_upper_bound = 256
    sparse_tensors = [torch.randint(100000000, (SM_param, batch_size), dtype=torch.int64).to(device) for _ in range(pre_iters_upper_bound)]

    # print("start search for max pre_iters")
    pre_iters_record = pre_iters
    result_list = []
    for out_loop in range(10):
        # if rank == 0:
            # print("iter:{} for max_pre_iters:{}".format(out_loop, pre_iters))
        pre_pre_iters = pre_iters_record
        pre_iters = pre_iters_record * 2
        communicate_tensor = torch.tensor([0])
        tensor_list = [torch.tensor([0]) for _ in range(nProcess)]
        while(communicate_tensor.item() == 0):
            if pre_iters >= pre_iters_upper_bound:
                break
            dist.barrier()
            start.record()
            for _ in range(repeat):
                if rank == 0:
                    pooled_embeddings = ebc(sparse_features)
                    optimizer.zero_grad()

                    vals = []
                    for _name, param in pooled_embeddings.to_dict().items():
                        vals.append(param)
                    torch.cat(vals, dim=1).sum().backward()
                    optimizer.step()
                else:
                    for i in range(pre_iters):
                        cuda_preprocess.sigrid_hash(sparse_tensors[i], 0, 65536)
                dist.barrier()
            end.record()
            dist.barrier()
            torch.cuda.synchronize()
            total_time = start.elapsed_time(end)
            avg_latency = total_time/float(repeat)
            performance_decrease = (avg_latency - avg_latency_lookup)/avg_latency_lookup
         
            if performance_decrease >= 0.07:
                communicate_tensor[0] = 1
            else:
                communicate_tensor[0] = 0
            dist.all_gather(tensor_list, communicate_tensor)
            comm_result = sum(tensor_list)

            if comm_result.item() > 0: # stop
                communicate_tensor[0] = 1
                break
            else: # continue increase pre_iters
                communicate_tensor[0] = 0
                pre_pre_iters = pre_iters
                pre_iters = pre_iters * 2
       
        # print("binary search for best pre_iters")
        up_bound = pre_iters
        lower_bound = pre_pre_iters
        while(up_bound - lower_bound > 2):
            pre_iters = int((up_bound + lower_bound)/2)
            dist.barrier()
            start.record()
            for _ in range(repeat):
                if rank == 0:
                    pooled_embeddings = ebc(sparse_features)
                    optimizer.zero_grad()

                    vals = []
                    for _name, param in pooled_embeddings.to_dict().items():
                        vals.append(param)
                    torch.cat(vals, dim=1).sum().backward()
                    optimizer.step()
                else:
                    for i in range(pre_iters):
                        cuda_preprocess.sigrid_hash(sparse_tensors[i], 0, 65536)
                dist.barrier()
            end.record()
            dist.barrier()
            torch.cuda.synchronize()
            total_time = start.elapsed_time(end)
            avg_latency = total_time/float(repeat)
            performance_decrease = (avg_latency - avg_latency_lookup)/avg_latency_lookup
         
            if performance_decrease >= 0.07:
                communicate_tensor[0] = 1
            else:
                communicate_tensor[0] = 0
            dist.all_gather(tensor_list, communicate_tensor)
            comm_result = sum(tensor_list)

            if comm_result.item() > 0: # stop
                up_bound = pre_iters - 1
            else:
                lower_bound = pre_iters + 1
        pre_iters = int(lower_bound * 0.8) 
        if pre_iters < 1:
            pre_iters = 1
        result_list.append(pre_iters)
       
    result_list.sort()
    result_list = result_list[2:8]
    pre_iters = int(sum(result_list)/6)
 
    start.record()
    for _ in range(repeat):
        if rank == 0:
            pooled_embeddings = ebc(sparse_features)
            optimizer.zero_grad()

            vals = []
            for _name, param in pooled_embeddings.to_dict().items():
                vals.append(param)
            torch.cat(vals, dim=1).sum().backward()
            optimizer.step()
        else:
            for i in range(pre_iters):
                cuda_preprocess.sigrid_hash(sparse_tensors[i], 0, 65536)
        dist.barrier()
    end.record()
    dist.barrier()
    torch.cuda.synchronize()
    total_time = start.elapsed_time(end)
    avg_latency = total_time/float(repeat)

    print("max_pre_iter avg_latency:{:.3f} ms, max_iters:{}".format(avg_latency, pre_iters))



    # ==================================== Max Preprocessing SM_param profile =======================================
    SM_param_record = SM_param
    result_list = []
    for out_loop in range(10):
        # if rank == 0:
            # print("iter:{} for max_SM:{}".format(out_loop, SM_param))

        pre_SM = SM_param_record
        SM_param = SM_param_record * 2
        communicate_tensor = torch.tensor([0])
        tensor_list = [torch.tensor([0]) for _ in range(nProcess)]
        while(communicate_tensor.item() == 0):
            if SM_param >= pre_iters_upper_bound:
                break
            sparse_tensors = [torch.randint(100000000, (SM_param, batch_size), dtype=torch.int64).to(device) for _ in range(pre_iters)]
            dist.barrier()
            start.record()
            for _ in range(repeat):
                if rank == 0:
                    pooled_embeddings = ebc(sparse_features)
                    optimizer.zero_grad()

                    vals = []
                    for _name, param in pooled_embeddings.to_dict().items():
                        vals.append(param)
                    torch.cat(vals, dim=1).sum().backward()
                    optimizer.step()
                else:
                    for i in range(pre_iters):
                        cuda_preprocess.sigrid_hash(sparse_tensors[i], 0, 65536)
                dist.barrier()
            end.record()
            dist.barrier()
            torch.cuda.synchronize()
            total_time = start.elapsed_time(end)
            avg_latency = total_time/float(repeat)
            performance_decrease = (avg_latency - avg_latency_lookup)/avg_latency_lookup
            # print("SM_param:{}, avg_latency:{:.3f} ms, performance_decrease:{:.3f}".format(SM_param, avg_latency, performance_decrease))
         
            if performance_decrease >= 0.07:
                communicate_tensor[0] = 1
            else:
                communicate_tensor[0] = 0
            dist.all_gather(tensor_list, communicate_tensor)
            comm_result = sum(tensor_list)

            if comm_result.item() > 0: # stop
                communicate_tensor[0] = 1
                break
            else: # continue increase pre_iters
                communicate_tensor[0] = 0
                pre_SM = SM_param
                SM_param = SM_param * 2
       
        # print("binary search for best pre_iters")
        up_bound = SM_param
        lower_bound = pre_SM
        while(up_bound - lower_bound > 2):
            SM_param = int((up_bound + lower_bound)/2)
            sparse_tensors = [torch.randint(100000000, (SM_param, batch_size), dtype=torch.int64).to(device) for _ in range(pre_iters)]
            dist.barrier()
            start.record()
            for _ in range(repeat):
                if rank == 0:
                    pooled_embeddings = ebc(sparse_features)
                    optimizer.zero_grad()

                    vals = []
                    for _name, param in pooled_embeddings.to_dict().items():
                        vals.append(param)
                    torch.cat(vals, dim=1).sum().backward()
                    optimizer.step()
                else:
                    for i in range(pre_iters):
                        cuda_preprocess.sigrid_hash(sparse_tensors[i], 0, 65536)
                dist.barrier()
            end.record()
            dist.barrier()
            torch.cuda.synchronize()
            total_time = start.elapsed_time(end)
            avg_latency = total_time/float(repeat)
            performance_decrease = (avg_latency - avg_latency_lookup)/avg_latency_lookup
            # print("SM_param:{}, avg_latency:{:.3f} ms, performance_decrease:{:.3f}".format(SM_param, avg_latency, performance_decrease))
         
            if performance_decrease >= 0.07:
                communicate_tensor[0] = 1
            else:
                communicate_tensor[0] = 0
            dist.all_gather(tensor_list, communicate_tensor)
            comm_result = sum(tensor_list)

            if comm_result.item() > 0: # stop
                up_bound = SM_param - 1
            else:
                lower_bound = SM_param + 1
        SM_param = int(lower_bound * 0.8) 
        if SM_param <= 0:
            SM_param = 1
        result_list.append(SM_param)
       
    result_list.sort()
    result_list = result_list[2:8]
    SM_param = int(sum(result_list)/6)
    
    sparse_tensors = [torch.randint(100000000, (SM_param, batch_size), dtype=torch.int64).to(device) for _ in range(pre_iters)]
    start.record()
    for _ in range(repeat):
        if rank == 0:
            pooled_embeddings = ebc(sparse_features)
            optimizer.zero_grad()

            vals = []
            for _name, param in pooled_embeddings.to_dict().items():
                vals.append(param)
            torch.cat(vals, dim=1).sum().backward()
            optimizer.step()
        else:
            for i in range(pre_iters):
                cuda_preprocess.sigrid_hash(sparse_tensors[i], 0, 65536)
        dist.barrier()
    end.record()
    dist.barrier()
    torch.cuda.synchronize()
    total_time = start.elapsed_time(end)
    avg_latency = total_time/float(repeat)
    
    print("max_SM_para avg_latency:{:.3f} ms, SM_param:{}".format(avg_latency, SM_param))

    performance_decrease = (avg_latency - avg_latency_lookup)/avg_latency_lookup
    exp_type = args.type
    pooling_factor = args.pooling_factor
    nTB = int(SM_param * batch_size * pre_iters / 128)
    if rank == 0:
        with open("output/overlapping.out", "a") as f:
            f.write("Criteo embeddingbag: batch_size:{}, exp_type:{}, pooling_factor:{}, pre_iter:{}, SM_param:{}, performance_decrease:{:.3f}%, nTB:{}\n".format(batch_size, exp_type, pooling_factor, pre_iters, SM_param, performance_decrease*100, nTB))


if __name__ == "__main__":
    args = parser.parse_args()
    processes = []
    mp.set_start_method("spawn")

    queue = mp.SimpleQueue()
    if args.type=="fused_table":
        print("fused table")
        p = mp.Process(target=main_fused_table, args=(0, 2, args, queue))
        processes.append(p)
        p = mp.Process(target=main_fused_table, args=(1, 2, args, queue))
        processes.append(p)
    elif args.type=="embeddingbag":
        print("embedding bag")
        p = mp.Process(target=main_embedding_bag, args=(0, 2, args, queue))
        processes.append(p)
        p = mp.Process(target=main_embedding_bag, args=(1, 2, args, queue))
        processes.append(p)

    for p in processes:
        p.start()

    for p in processes:
        p.join()