# Based on: https://github.com/pytorch/examples/blob/master/mnist/main.py
import os
import argparse
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import time


from torch.optim.lr_scheduler import StepLR
from transformers import BertModel, BertConfig

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from realfsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from realfsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

from utils.dist_train import setup, cleanup
from scripts.training import train, test
from models.exp_net import Net
from options.fsdp_options import args

# Dummy dataset for demonstration
class RandomDataset(Dataset):
    def __len__(self):
        return 100  # Number of data points

    def __getitem__(self, idx):
        # Generating random input ids and a random target label
        return torch.randint(0, 2000, (128,)), torch.randint(0, 2, (1,))

def print_fsdp_layer_memory(model):
    wrapping_layer = 0
    for name, module in model.named_modules():
        total_mem = 0
        for param in module.parameters(recurse=False):
            param_size = param.numel() * param.element_size()  # total number of elements times element size in bytes
            total_mem += param_size
        if total_mem > 0:  # Only print layers that have parameters
            wrapping_layer += 1
            print(f"Layer: {name}, Memory: {total_mem / (1024 ** 2):.2f} MB")  # Convert bytes to megabytes
    print('Total number of wrapping layers:', wrapping_layer)

# Training function
def train_one_epoch(model, dataloader):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()  # Start timing

    for inputs, labels in dataloader:
        inputs, labels = inputs.cuda(), labels.cuda()  # Move data to GPU
        optimizer.zero_grad()
        outputs = model(inputs)[0][:, 0]  # Use CLS token output for classification
        loss = criterion(outputs, labels.squeeze())
        loss.backward()
        optimizer.step()

    end_time = time.time()  # End timing
    print(f"Finished one epoch in {end_time - start_time:.2f} seconds")

def fsdp_main(rank, world_size, args):
    setup(rank, world_size)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Data loader setup
    dataset1 = RandomDataset()
    dataset2 = RandomDataset()

    sampler1 = DistributedSampler(dataset1, rank=rank, num_replicas=world_size, shuffle=True)
    sampler2 = DistributedSampler(dataset2, rank=rank, num_replicas=world_size)


    train_loader = DataLoader(dataset1, batch_size=8, shuffle=True)
    test_loader = DataLoader(dataset2, batch_size=8, shuffle=True)

    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=1e4
    )
    torch.cuda.set_device(rank)


    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)

    # Configure the BERT model
    config = BertConfig()
    print(config)
    model = BertModel(config).to(rank)

    #model = Net().to(rank)

    model = FSDP(model,auto_wrap_policy=my_auto_wrap_policy)

    print(model)
    print_fsdp_layer_memory(model)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    init_start_event.record()
    train_one_epoch(model, train_loader)

    init_end_event.record()

    if rank == 0:
        print(f"CUDA event elapsed time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
        print(f"{model}")

    if args.save_model:
        # use a barrier to make sure training is done on all ranks
        dist.barrier()
        states = model.state_dict()
        if rank == 0:
            torch.save(states, "mnist_cnn.pt")

    cleanup()

if __name__ == "__main__":
    
    torch.manual_seed(args.seed)
    WORLD_SIZE = torch.cuda.device_count()
    print('World size:', WORLD_SIZE)
    mp.spawn(fsdp_main,
        args=(WORLD_SIZE, args),
        nprocs=WORLD_SIZE,
        join=True)