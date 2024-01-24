import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.distributed as dist
import torch.multiprocessing as mp
import time

def put_hook_func(module, input, output):
    print("put")
    module.queue.put(1)

def fc1_hook_func(module, input, output):
    print("fc1")

def fc2_hook_func(module, input, output):
    print("fc2")

class TwoLayerMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, queue=None):
        super(TwoLayerMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        # self.relu = nn.ReLU()
        self.queue = queue

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        return out

    def print_grad(self):
        if self.fc2.weight.grad != None:
            print(self.fc2.weight.grad[0])
        else:
            print("None")
    
    def print_queue(self):
        print(self.queue)

def train_process(rank, nDev, queue):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("nccl", rank=rank, world_size=nDev)
    dist.barrier()
    print("finish init:{}".format(rank))

    model = TwoLayerMLP(16, 16, 8, queue).to(rank)

    model.register_full_backward_hook(put_hook_func)
    model.fc1.register_full_backward_hook(fc1_hook_func)
    model.fc2.register_full_backward_hook(fc2_hook_func)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    label = torch.zeros((4,8)).to(rank)
    random_input = torch.zeros((4,16)).to(rank)

    output = model(random_input)

    loss = criterion(label, output)
    if rank == 0:
        loss.backward()
    if rank == 1:
        while queue.empty():
            time.sleep(0.01)
        res = queue.get()
        if queue.empty():
            print("get from queue: ", res)


if __name__ == "__main__":

    processes = []
    mp.set_start_method("spawn")
    

    queue = mp.SimpleQueue()

    nDev = 2

    for rank in range(nDev):
        p = mp.Process(target=train_process, args=(rank, nDev, queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

# class hook_function(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, queue):
#         ctx.queue = queue
    
#     @staticmethod
#     def backward(ctx, grad_output):
#         ctx.queue.append(1)

#         return grad_output

# def hook_func(module, input, output):
#     print("hook")
#     module.queue.append(2)




# def add_element(grad):
#     print("1")


# if __name__ == "__main__":

#     input_size = 16  # MNIST images are 28x28 = 784 pixels
#     hidden_size = 16
#     output_size = 8  # 10 classes for MNIST (digits 0-9)

#     queue = []
#     model_1 = TwoLayerMLP(16, 16, 8, queue).to(0)
#     model_2 = TwoLayerMLP(8, 8, 1, queue).to(0)

#     model_1.register_backward_hook(hook_func)

#     criterion = nn.MSELoss()
#     optimizer = optim.SGD(model_1.parameters(), lr=0.01)
#     label = torch.zeros((4,1)).to(0)
#     random_input = torch.zeros((4,16)).to(0)

#     # mid_value = output_1.detach().to(0)
#     # mid_variable = Variable(mid_value)
#     # mid_variable.requires_grad = True
#     # output_1.register_hook(add_element)
#     output_1 = model_1(random_input)
#     output_2 = model_2(output_1)

#     # model_1.print_queue()
#     # model_2.print_queue()

#     print(queue)
#     loss = criterion(label, output_2)
#     loss.backward()

#     print(queue)


#     # model_1.print_queue()
#     # model_2.print_queue()



    
    


