import torch, json, time, logging
from pathlib import Path
import torch.nn as nn

from transformers import LlavaConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from update_kv_indexes import update_kv_index
from mem_manager import ComputeMaxAvailableBlocks, KVCacheMemoryManager
from req_tokens_manager import ReqTokensManager

logger = logging.getLogger(__name__)

from dataclasses import dataclass
import torch
from typing import List


@dataclass
class ModelRunnerConfig:
    block_size = 1
    checkpoints_dir = "/gemini/code/Llama-3.2-1B-Instruct"
    max_batch_size = 16
    gpu_memory_utilization = 0.9


@dataclass
class AttentionInfo:
    # kv_cache = None # prefill 阶段的 context kv cache
    kv_buffer = List[torch.tensor([])]
    cur_select_index = torch.empty((0,), dtype=torch.int32)
    b_req_tokens_table = None
    b_start_loc = None
    b_req_idx = None


class ModelExecutor:
    # 定义类属性
    model_config = None
    model = None
    # model_runner_config = ModelRunnerConfig
    atten_info = AttentionInfo

    def __init__(self, max_gpu_num_blocks, device="cuda"):
        self.device = device

        self.max_seq_len = 512

        self.kv_mem_manager = self._init_mem_manager(max_gpu_num_blocks)
        self.max_gpu_num_tokens = max_gpu_num_blocks

        self.max_request_num = max_gpu_num_blocks // self.max_seq_len

        self.req_tokens_manager = ReqTokensManager(self.max_request_num, self.max_seq_len)
        self.atten_info = AttentionInfo()  # 创建 AttentionInfo 实例
        self.atten_info.kv_buffer = self.kv_mem_manager.gpu_kv_buffer
        self.atten_info.b_req_tokens_table = self.req_tokens_manager.b_req_tokens_table

    def _init_mem_manager(self, gpu_num_blocks, block_size=1, dtype=torch.float16, device="cuda"):
        num_layers = 4
        num_kv_heads = 8
        head_dim = 16
        kv_mem_manager = KVCacheMemoryManager(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,

            gpu_num_blocks=gpu_num_blocks,
            block_size=block_size,
            dtype=dtype,
            device=device
        )

        return kv_mem_manager

    def _init_req_to_tokens_table(self, b_req_tokens_table, b_req_idx, b_seq_len, alloc_mem_index):
        """
        初始化 prefill 阶段已分配的批次请求项的 kv cache 所用 tokens 索引
        """
        # TODO: 性能等待优化
        start_index = 0
        batch_size = len(b_seq_len)
        b_seq_len_numpy = b_seq_len.cpu().numpy()
        b_req_idx_numpy = b_req_idx.cpu().numpy()
        b_start_loc = torch.zeros((batch_size,), dtype=torch.int32, device=self.device)
        for i in range(batch_size):
            if i > 0:
                b_start_loc[i] = start_index
            cur_seq_len = b_seq_len_numpy[i]
            b_req_tokens_table[b_req_idx_numpy[i], :cur_seq_len] = alloc_mem_index[
                                                                   start_index: start_index + cur_seq_len]
            start_index += cur_seq_len

        return b_start_loc

    def prefill_alloc_kv_cache(self,
                               max_prompt_len, actual_prompt_lens, b_req_idx, debug_mode=False,
                               ):
        """
        start_index:        tensor([  0, 270, 540, 810], device='cuda:0', dtype=torch.int32)
        b_seq_len:          tensor([14, 12, 11, 11], device='cuda:0')
        Prefill Stage, cur_select_index: tensor([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
                                    270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283,
                                    540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553,
                                    810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823
                                ], device='cuda:0')
        Decode Stage, 0 step, cur_select_index: tensor([ 14, 282, 551, 821], device='cuda:0'), cur_b_seq_len: tensor([15, 13, 12, 12], device='cuda:0')
        Decode Stage, 1 step, cur_select_index: tensor([ 15, 283, 552, 822], device='cuda:0'), cur_b_seq_len: tensor([16, 14, 13, 13], device='cuda:0')
        """
        num_patch_indexs = None
        batch_size = len(actual_prompt_lens)
        self.atten_info.b_req_idx = b_req_idx

        context_num_tokens = max_prompt_len * batch_size
        # 一次性分配 bsz * seq_len + (number_patchs * number_patchs - 1) * img_batch_size 个索引
        self.atten_info.cur_select_index, _ = self.kv_mem_manager.alloc_kvcache_index(context_num_tokens)
        # 初始化每个批次项的实际提示词长度
        self.atten_info.b_seq_len = actual_prompt_lens  # 张量, 形状 [batch_size, 1]
        # 初始化批次请求的当前最大序列上下文长度(对应 kv cache 长度)
        self.atten_info.max_actual_seq_len = max_prompt_len  # int 类型

        self.atten_info.b_start_loc = self._init_req_to_tokens_table(
            self.atten_info.b_req_tokens_table, self.atten_info.b_req_idx,
            self.atten_info.b_seq_len, self.atten_info.cur_select_index
        )

        if debug_mode:
            print(f"context_num_tokens: {context_num_tokens}, max_prompt_len:{max_prompt_len}, \n \
                    self.atten_info.cur_select_index: {self.atten_info.cur_select_index},\n \
                    self.atten_info.max_actual_seq_len: {self.atten_info.max_actual_seq_len},\n \
                    self.atten_info.b_seq_len: {self.atten_info.b_seq_len}, \n \
                    self.atten_info.b_start_loc: {self.atten_info.b_start_loc}, "
                  )

        return self.atten_info.cur_select_index, num_patch_indexs

    def decode_alloc_kv_cache(self, batch_size):
        # TODO: torch.empty 创建的临时张量, 保存分配的非连续 kv_cache 索引空间
        self.atten_info.cur_select_index, _ = self.kv_mem_manager.alloc_kvcache_index(batch_size)
        update_kv_index(self.atten_info.b_req_tokens_table, self.atten_info.b_req_idx,
                        self.atten_info.b_seq_len, self.atten_info.cur_select_index)

        self.atten_info.b_seq_len += 1
        self.atten_info.max_actual_seq_len += 1

        return self.atten_info.cur_select_index  # shape [batch_size,]


if __name__ == '__main__':
    max_gpu_blocks = 512 * 512
    model_exe = ModelExecutor(max_gpu_blocks)
    b_req_idx = torch.arange(0, 4).to(torch.int32).cuda()
    actual_prompt_len = torch.Tensor([4, 4, 4, 5]).to(torch.int32).cuda()
    model_exe.prefill_alloc_kv_cache(5, actual_prompt_len, b_req_idx)
    model_exe.decode_alloc_kv_cache(4)
