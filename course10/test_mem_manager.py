# 代码可直接运行，用于测试 KVCacheMemoryManager 的结果

import unittest
import torch, os, sys
from mem_manager import KVCacheMemoryManager, ComputeMaxAvailableBlocks


def _get_max_avaliable_tokens(
    num_layers,
    head_size,
    num_heads,
    num_kv_heads,
    gpu_memory_utilization=0.4,
    block_size=1,
):
    avaliable_blocks = ComputeMaxAvailableBlocks(
        hidden_size=head_size,
        num_layers=num_layers,
        head_dim=head_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        gpu_memory_utilization=gpu_memory_utilization,
        block_size=block_size,
    )
    block_size = 1
    max_gpu_num_blocks = avaliable_blocks.compute_num_available_blocks()
    return max_gpu_num_blocks


if __name__ == "__main__":

    head_dim = 64
    num_kv_heads = 4
    num_layers = 2
    gpu_num_blocks = _get_max_avaliable_tokens(
        num_layers, head_dim, num_kv_heads, num_kv_heads
    )
    dtype = torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    manager = KVCacheMemoryManager(
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        num_layers=num_layers,
        gpu_num_blocks=gpu_num_blocks,
        dtype=dtype,
        device=device,
    )
    alloc_res1 = manager.alloc_contiguous_kvcache(3)
    print(alloc_res1)
