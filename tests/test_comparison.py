import argparse
import os

import torch
from deepep_dispatcher import DeepEPMoEDispatcher
from nccl_dispatcher import NCCLMoEDispatcher
from utils import bench, init_dist

from fusco import FUSCO, FuscoMoEDispatcher

global_fusco = None
intra_fusco = None
FUSCO_LIB_NAME = "libfusco.so"


def test_correctness(
    nccl_ds,
    deepep_ds,
    fused_ds,
    hidden_states: torch.Tensor,
    probs: torch.Tensor,
    topk_idx: torch.Tensor,
    local_rank,
):
    if local_rank == 0:
        print("Testing correctness...", flush=True)

    nccl_dispatched, _ = nccl_ds.dispatch(hidden_states, probs, topk_idx)
    nccl_combined = nccl_ds.combine(nccl_dispatched)

    fused_dispatched, _ = fused_ds.dispatch(hidden_states, probs, topk_idx)
    fused_combined = fused_ds.combine(fused_dispatched)

    deepep_dispatched, _ = deepep_ds.dispatch(hidden_states, probs, topk_idx)
    deepep_combined = deepep_ds.combine(deepep_dispatched)

    if local_rank == 0:
        print(
            f"NCCL == FUSCO: {torch.allclose(nccl_combined, fused_combined, rtol=2e-2, atol=2e-2)}",
            flush=True,
        )
        print(
            f"DeepEP == FUSCO: {torch.allclose(fused_combined, deepep_combined, rtol=2e-2, atol=2e-2)}",
            flush=True,
        )
        print(
            f"DeepEP == NCCL: {torch.allclose(deepep_combined, nccl_combined, rtol=2e-2, atol=2e-2)}",
            flush=True,
        )


def mlp_wrap(dispatcher, hidden_states: torch.Tensor, probs: torch.Tensor, topk_idx: torch.Tensor):
    dispatched, _ = dispatcher.dispatch(hidden_states, probs, topk_idx)
    combined = dispatcher.combine(dispatched)
    return combined


def bench_dispatcher(dispatcher, hidden_states, probs, topk_idx):
    hs = hidden_states.clone()
    p = probs.clone()
    idx = topk_idx.clone()
    return bench(lambda: mlp_wrap(dispatcher, hs, p, idx))


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

    deepep_dispatcher = DeepEPMoEDispatcher(
        num_local_experts, local_expert_indices, num_ranks, world_group, num_topk
    )

    nccl_dispatcher = NCCLMoEDispatcher(
        num_local_experts, local_expert_indices, num_ranks, world_group
    )

    num_nodes = num_ranks // num_local_ranks
    is_2dmode = num_topk > 1 and num_nodes > 1
    fusco_dispatcher = FuscoMoEDispatcher(
        num_local_experts,
        local_expert_indices,
        num_ranks,
        world_group,
        is_2dmode,
        global_fusco,
        intra_fusco,
        num_local_ranks,
    )

    t_deepep = bench_dispatcher(deepep_dispatcher, hidden_states, probs, topk_idx)
    t_nccl = bench_dispatcher(nccl_dispatcher, hidden_states, probs, topk_idx)
    t_fusco = bench_dispatcher(fusco_dispatcher, hidden_states, probs, topk_idx)

    if local_rank == 0:
        print(
            f"NCCL performance: average={t_nccl[0] * 1000:.3f} ms, min={t_nccl[1] * 1000:.3f} ms, max={t_nccl[2] * 1000:.3f} ms",
            flush=True,
        )
        print(
            f"DeepEP performance: average={t_deepep[0] * 1000:.3f} ms, min={t_deepep[1] * 1000:.3f} ms, max={t_deepep[2] * 1000:.3f} ms",
            flush=True,
        )
        print(
            f"FUSCO performance: average={t_fusco[0] * 1000:.3f} ms, min={t_fusco[1] * 1000:.3f} ms, max={t_fusco[2] * 1000:.3f} ms",
            flush=True,
        )

    test_correctness(
        nccl_dispatcher,
        deepep_dispatcher,
        fusco_dispatcher,
        hidden_states,
        probs,
        topk_idx,
        local_rank,
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
