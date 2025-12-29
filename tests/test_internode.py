import argparse
import time
import torch
import os
import sys
from utils import init_dist, bench

testdir = os.path.dirname(os.path.abspath(__file__))
rootdir = os.path.join(testdir, "..")
sys.path.append(os.path.join(rootdir, "python"))
sys.path.append(os.path.join(rootdir, "lib"))

from moe_dispatcher import Fusco2DMoEDispatcher
from fusco import FUSCO

global_fusco = None
intra_fusco = None

def test_main(
    args: argparse.Namespace, 
    local_rank: int, 
    num_ranks: int,
    num_local_ranks: int,
    global_rank: int, 
    world_group: torch.distributed.ProcessGroup
):
    # Settings
    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts
    num_local_experts = num_experts // num_ranks
    local_expert_indices = list(range(global_rank * num_local_experts, (global_rank + 1) * num_local_experts))

    assert num_experts % num_ranks == 0
    if local_rank == 0:
        print(f'[config] num_tokens={num_tokens}, hidden={hidden}, num_topk={num_topk}', flush=True)

    # Random data
    hidden_states = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device=torch.device('cuda'))
    scores = torch.randn((num_tokens, num_experts), dtype=torch.bfloat16, device=torch.cuda.current_device()).abs() + 1
    probs, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
    topk_idx = topk_idx.to(torch.int64)

    dispatcher = Fusco2DMoEDispatcher(num_local_experts, local_expert_indices, num_ranks, world_group, global_fusco, intra_fusco, num_local_ranks)

    dispatched_inputs, _ = dispatcher.dispatch(hidden_states, probs, topk_idx)

    outputs = dispatcher.combine(dispatched_inputs)

    t_dispatch = bench(lambda: dispatcher.dispatch(hidden_states, probs, topk_idx))
    t_combine = bench(lambda: dispatcher.combine(hidden_states))

    if local_rank == 0:
        print(f'Dispatch performance: average={t_dispatch[0] * 1000:.3f} ms, min={t_dispatch[1] * 1000:.3f} ms, max={t_dispatch[2] * 1000:.3f} ms', flush=True)
        print(f'Combine performance: average={t_combine[0] * 1000:.3f} ms, min={t_combine[1] * 1000:.3f} ms, max={t_combine[2] * 1000:.3f} ms', flush=True)

def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    global_rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    global global_fusco, intra_fusco
    group_ranks = [
        list(range(start, start + num_local_ranks))
        for start in range(0, num_ranks, num_local_ranks)
    ]
    global_fusco = FUSCO(group_ranks=[list(range(num_ranks))], library_path=os.path.join(rootdir, "lib", "libfusco.so"))
    intra_fusco = FUSCO(group_ranks=group_ranks, library_path=os.path.join(rootdir, "lib", "libfusco.so"))
    torch.manual_seed(global_rank)

    test_main(args, local_rank, num_ranks, num_local_ranks, global_rank, group)

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test EP')
    parser.add_argument('--num-processes', type=int, default=8, help='Number of processes to spawn')
    parser.add_argument('--num-tokens', type=int, default=16384, help='Number of tokens')
    parser.add_argument('--hidden', type=int, default=7168, help='Hidden dimension size')
    parser.add_argument('--num-topk', type=int, default=8, help='Number of top-k experts')
    parser.add_argument('--num-experts', type=int, default=256, help='Number of experts')
    args = parser.parse_args()
    
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['NCCL_NCHANNELS_PER_NET_PEER'] = '32'

    num_processes = args.num_processes
    torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)