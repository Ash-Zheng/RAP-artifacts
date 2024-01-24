
import sys, os
import torch
import torch.distributed as dist
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from contextlib import contextmanager

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

def print_once(rank, msg):
    if rank == 0:
        print(msg)

def average_gradients(model, group=None):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        if param.grad != None:
            if group == None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.AVG)
            else:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.AVG, group=group)
            param.grad.data /= size

def convert_to_KeyedJaggedTensor_fixed_offset(keys, input, offset):
    kjt_tensor = KeyedJaggedTensor.from_lengths_sync(
        keys=keys,
        values=input,
        lengths=offset,
    )
    return kjt_tensor


def profiling_hook_func(module, input, output):
    # print("backward hook func")
    module.recorder_1.record()
    torch.cuda.synchronize()
    module.recorder_2.record()


def queue_hook_func(module, input, output):
    # print("backward hook func")
    module.queue.put(1)


class InMemoryDataLoader():
    def __init__(self, train_dataloader, rank, nDev, nRandom=512):
        self.input_data = []
        self.nRandom = nRandom
        self.batch_idx = rank
        self.rank = rank
        self.nDev = nDev

        dataiter = iter(train_dataloader)

        for i in range(nRandom * nDev):
            batch = next(dataiter)
            if i % nDev == rank:
                self.input_data.append(batch.to(rank))

        # for i in range(nRandom):
        #     batch = next(dataiter).to(rank)
        #     self.input_data.append(batch)
        self.input_iter = iter(self.input_data)


    def next(self):
        # batch = self.input_data[self.batch_idx]

        # # reset the index of the dataloader
        # self.batch_idx += self.nDev
        # if self.batch_idx >= self.nRandom - 1:
        #     self.batch_idx = self.rank 
     
        # return batch
        # ==========================

        self.batch_idx += 1
        if self.batch_idx >= self.nRandom:
            self.batch_idx = 0
            self.input_iter = iter(self.input_data)

        batch = next(self.input_iter)
        return batch


    def reset(self):
        self.batch_idx = 0
        self.input_iter = iter(self.input_data)