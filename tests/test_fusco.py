import argparse
import os

import torch
from utils import bench, init_dist

from fusco import FUSCO, FuscoMoEDispatcher

global_fusco = None
intra_fusco = None
FUSCO_LIB_NAME = "libfusco.so"


def test_main(
    args: argparse.Namespace,
    local_rank: int,
    num_ranks: int,
    num_local_ranks: int,
    global_rank: int,
    world_group: torch.distributed.ProcessGroup,
):
    # Settings
    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts
    num_local_experts = num_experts // num_ranks
    local_expert_indices = list(
        range(global_rank * num_local_experts, (global_rank + 1) * num_local_experts)
    )

    assert num_experts % num_ranks == 0
    if local_rank == 0:
        print(f"[config] num_tokens={num_tokens}, hidden={hidden}, num_topk={num_topk}", flush=True)

    # Random data
    hidden_states = torch.randn(
        (num_tokens, hidden), dtype=torch.bfloat16, device=torch.device("cuda")
    )
    scores = (
        torch.randn(
            (num_tokens, num_experts), dtype=torch.bfloat16, device=torch.cuda.current_device()
        ).abs()
        + 1
    )
    probs, topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)
    topk_idx = topk_idx.to(torch.int64)

    num_nodes = num_ranks // num_local_ranks
    is_2dmode = num_topk > 1 and num_nodes > 1
    dispatcher = FuscoMoEDispatcher(
        num_local_experts,
        local_expert_indices,
        num_ranks,
        world_group,
        is_2dmode,
        global_fusco,
        intra_fusco,
        num_local_ranks,
    )

    dispatched_inputs, _ = dispatcher.dispatch(hidden_states, probs, topk_idx)

    _ = dispatcher.combine(dispatched_inputs)

    t_dispatch = bench(lambda: dispatcher.dispatch(hidden_states, probs, topk_idx))
    t_combine = bench(lambda: dispatcher.combine(hidden_states))

    if local_rank == 0:
        print(
            f"Dispatch performance: average={t_dispatch[0] * 1000:.3f} ms, min={t_dispatch[1] * 1000:.3f} ms, max={t_dispatch[2] * 1000:.3f} ms",
            flush=True,
        )
        print(
            f"Combine performance: average={t_combine[0] * 1000:.3f} ms, min={t_combine[1] * 1000:.3f} ms, max={t_combine[2] * 1000:.3f} ms",
            flush=True,
        )


def init_fusco(num_ranks, num_local_ranks, world_group, global_rank, args: argparse.Namespace):
    global global_fusco, intra_fusco

    library_path = os.path.realpath(os.path.join(args.library_path, FUSCO_LIB_NAME))
    if not os.path.exists(library_path):
        raise FileNotFoundError(f"Shared library {FUSCO_LIB_NAME} not found in {library_path}")

    global_fusco = FUSCO(
        nccl_ep_group=world_group,
        library_path=library_path,
    )

    num_nodes = num_ranks // num_local_ranks
    if num_nodes > 1:
        intra_group_ranks = [
            list(range(start, start + num_local_ranks))
            for start in range(0, num_ranks, num_local_ranks)
        ]
        intra_group = None
        nccl_options = torch.distributed.ProcessGroupNCCL.Options()
        nccl_options.config.cga_cluster_size = 8
        nccl_options.config.max_ctas = 32
        nccl_options.config.min_ctas = 32
        for ranks in intra_group_ranks:
            if global_rank in ranks:
                intra_group = torch.distributed.new_group(
                    ranks, backend="nccl", pg_options=nccl_options
                )
        intra_fusco = FUSCO(nccl_ep_group=intra_group, library_path=library_path)


def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    global_rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    assert num_ranks > 1, "at least 2 ranks are required"
    assert num_ranks % num_local_ranks == 0, "assume each node has the same number of GPU ranks"

    init_fusco(num_ranks, num_local_ranks, group, global_rank, args)

    torch.manual_seed(global_rank)

    test_main(args, local_rank, num_ranks, num_local_ranks, global_rank, group)

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available"
    available_gpus = torch.cuda.device_count()
    assert available_gpus > 0, "No CUDA-capable devices detected"

    parser = argparse.ArgumentParser(description="Test EP")
    parser.add_argument(
        "--num-processes", type=int, default=available_gpus, help="Number of processes to spawn"
    )
    parser.add_argument("--num-tokens", type=int, default=16384, help="Number of tokens")
    parser.add_argument("--hidden", type=int, default=7168, help="Hidden dimension size")
    parser.add_argument("--num-topk", type=int, default=8, help="Number of top-k experts")
    parser.add_argument("--num-experts", type=int, default=256, help="Number of experts")
    parser.add_argument(
        "--library-path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "lib"),
        help="Path to the shared library",
    )
    args = parser.parse_args()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["NCCL_NCHANNELS_PER_NET_PEER"] = "32"

    num_processes = args.num_processes
    torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)
