from typing import NamedTuple

import torch

import idxtools

from .fusco import FUSCO


def gather_along_first_dim(input_, group):
    world_size = torch.distributed.get_world_size(group=group)

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    torch.distributed.all_gather_into_tensor(output, input_.contiguous(), group=group)

    return output


class FuscoInfo(NamedTuple):
    # cross-node
    global_fusco: FUSCO
    sendindices_s1: torch.Tensor
    recvindices_s1: torch.Tensor
    backindices_s1: torch.Tensor
    send_splits_s1: torch.Tensor
    recv_splits_s1: torch.Tensor
    send_tokens_s1: int
    recv_tokens_s1: int
    seq_len: int
    # intra-node
    intra_fusco: FUSCO = None
    sendindices_s2: torch.Tensor = None
    recvindices_s2: torch.Tensor = None
    backindices_s2: torch.Tensor = None
    send_splits_s2: torch.Tensor = None
    recv_splits_s2: torch.Tensor = None
    send_tokens_s2: int = 0
    recv_tokens_s2: int = 0
    intragroup_expanded_size: torch.Tensor = None
    intergroup_expanded_size: torch.Tensor = None
    probs: torch.Tensor = None


def dispatch_raw(
    hidden_states: torch.Tensor,
    info: FuscoInfo,
    apply_probs: bool,
):
    assert hidden_states.dim() == 2, "hidden_states must be a 2D tensor"
    hidden_dim = hidden_states.shape[1]
    curr_device = hidden_states.device
    curr_dtype = hidden_states.dtype
    curr_stream = torch.cuda.current_stream()

    is_2dmode = info.intra_fusco is not None

    s1_out = hidden_states.new_empty(
        (info.recv_tokens_s1, hidden_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    if not is_2dmode:
        if apply_probs:
            hidden_states = hidden_states * info.probs.unsqueeze(-1)
            sendindices_used = info.backindices_s1
        else:
            sendindices_used = info.sendindices_s1

        info.global_fusco.all_to_all(
            s1_out,
            hidden_states,
            recvindices=info.recvindices_s1,
            sendindices=sendindices_used,
            recv_splits=info.recv_splits_s1,
            send_splits=info.send_splits_s1,
            stream=curr_stream,
        )
        return s1_out
    else:
        s2_out = hidden_states.new_empty(
            (info.recv_tokens_s2, hidden_dim),
            dtype=curr_dtype,
            device=curr_device,
        )

        info.global_fusco.all_to_all(
            s1_out,
            hidden_states,
            recvindices=info.recvindices_s1,
            sendindices=info.sendindices_s1,
            recv_splits=info.recv_splits_s1,
            send_splits=info.send_splits_s1,
            stream=curr_stream,
        )

        if apply_probs:
            s1_out = s1_out.repeat_interleave(info.intragroup_expanded_size, dim=0)
            s1_out = s1_out * info.probs.unsqueeze(-1)
            sendindices_used = info.backindices_s2
        else:
            sendindices_used = info.sendindices_s2

        info.intra_fusco.all_to_all(
            s2_out,
            s1_out,
            recvindices=info.recvindices_s2,
            sendindices=sendindices_used,
            recv_splits=info.recv_splits_s2,
            send_splits=info.send_splits_s2,
            stream=curr_stream,
        )

        return s2_out


def combine_raw(hidden_states: torch.Tensor, info: FuscoInfo, apply_probs: bool):
    assert hidden_states.dim() == 2, "hidden_states must be a 2D tensor"
    hidden_dim = hidden_states.shape[1]
    curr_device = hidden_states.device
    curr_dtype = hidden_states.dtype
    curr_stream = torch.cuda.current_stream()

    is_2dmode = info.intra_fusco is not None

    s1_out = hidden_states.new_empty(
        (info.send_tokens_s1, hidden_dim),
        dtype=curr_dtype,
        device=curr_device,
    )

    if not is_2dmode:
        info.global_fusco.all_to_all(
            s1_out,
            hidden_states,
            recvindices=info.backindices_s1,
            sendindices=info.recvindices_s1,
            recv_splits=info.send_splits_s1,
            send_splits=info.recv_splits_s1,
            stream=curr_stream,
        )

        s1_out = s1_out.reshape((info.seq_len, -1, hidden_dim))
        if apply_probs:
            s1_out = s1_out * info.probs.unsqueeze(-1)
        s1_out = s1_out.sum(dim=1)

        return s1_out
    else:
        s2_out = hidden_states.new_empty(
            (info.send_tokens_s2, hidden_dim),
            dtype=curr_dtype,
            device=curr_device,
        )

        info.intra_fusco.all_to_all(
            s2_out,
            hidden_states,
            recvindices=info.backindices_s2,
            sendindices=info.recvindices_s2,
            recv_splits=info.send_splits_s2,
            send_splits=info.recv_splits_s2,
            stream=curr_stream,
        )

        if info.send_tokens_s2 > 0:
            if apply_probs:
                s2_out = s2_out * info.probs.unsqueeze(-1)
            s2_out = torch.segment_reduce(
                s2_out, lengths=info.intragroup_expanded_size, reduce="sum"
            )

        info.global_fusco.all_to_all(
            s1_out,
            s2_out,
            recvindices=info.backindices_s1,
            sendindices=info.recvindices_s1,
            recv_splits=info.send_splits_s1,
            send_splits=info.recv_splits_s1,
            stream=torch.cuda.current_stream(),
        )

        if info.send_tokens_s1 != info.intergroup_expanded_size.numel():
            s1_out = torch.segment_reduce(
                s1_out, lengths=info.intergroup_expanded_size, reduce="sum"
            )

        return s1_out


class FuscoDispatch(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        info: FuscoInfo,
        apply_probs: bool,
    ):
        ctx.info = info
        ctx.apply_probs = apply_probs
        return dispatch_raw(hidden_states, info, False)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return combine_raw(grad_output, ctx.info, False), None, None


class FuscoCombine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states: torch.Tensor, info: FuscoInfo, apply_probs):
        ctx.info = info
        ctx.apply_probs = apply_probs
        return combine_raw(hidden_states, info, apply_probs)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return dispatch_raw(grad_output, ctx.info, ctx.apply_probs), None, None


class FuscoMoEDispatcher:
    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: list[int],
        ep_size: int,
        ep_group: torch.distributed.ProcessGroup,
        is_2dmode: bool,
        global_fusco: FUSCO = None,
        intra_fusco: FUSCO = None,
        num_local_ranks: int = 8,
        apply_probs: bool = True,
    ):
        self.num_local_experts = num_local_experts
        self.num_experts = ep_size * num_local_experts
        assert self.num_local_experts > 0, "Expected at least one expert"
        self.local_expert_indices = local_expert_indices
        self.ep_size = ep_size
        self.ep_group = ep_group
        self.global_fusco = global_fusco
        self.intra_fusco = intra_fusco
        self.num_local_ranks = num_local_ranks
        self.apply_probs = apply_probs
        self.is_2dmode = is_2dmode

    def preprocess_1d(self, indices: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        seqlen, topk = indices.shape
        num_local_tokens_per_expert = torch.bincount(indices.view(-1), minlength=self.num_experts)

        num_local_tokens_per_rank = num_local_tokens_per_expert.view(
            self.ep_size, self.num_local_experts
        ).sum(dim=1)

        topk = indices.size(1)
        flatten_indices = indices.view(-1)

        sendindices_unique = torch.argsort(flatten_indices, stable=True).contiguous()
        sendindices_with_duplicates = (sendindices_unique // topk).contiguous()
        send_splits = num_local_tokens_per_rank.to(torch.device("cpu"))

        num_global_tokens_per_expert = gather_along_first_dim(
            num_local_tokens_per_expert, self.ep_group
        ).reshape(self.ep_size, self.num_experts)

        num_global_tokens_per_local_expert = num_global_tokens_per_expert[
            :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
        ].contiguous()

        num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(dim=0)

        num_tokens_per_ep = num_global_tokens_per_local_expert.sum(dim=1)

        num_ep_tokens = num_global_tokens_per_local_expert.sum()

        if num_ep_tokens > 0:
            recvindices = idxtools.indices_gen(
                num_global_tokens_per_local_expert,
                num_tokens_per_local_expert,
                num_tokens_per_ep,
                num_ep_tokens.item(),
            )
        else:
            recvindices = torch.empty(0, dtype=torch.int64, device=torch.cuda.current_device())
        recv_splits = num_tokens_per_ep.to(torch.device("cpu"))

        self.info = FuscoInfo(
            global_fusco=self.global_fusco,
            sendindices_s1=sendindices_with_duplicates,
            recvindices_s1=recvindices,
            backindices_s1=sendindices_unique,
            send_splits_s1=send_splits,
            recv_splits_s1=recv_splits,
            send_tokens_s1=indices.numel(),
            recv_tokens_s1=num_ep_tokens,
            seq_len=seqlen,
            probs=probs,
        )

        return num_tokens_per_local_expert

    def preprocess_2d(self, indices: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        seqlen, topk = indices.shape
        assert topk > 1, (
            "2D Dispatcher is efficient only when topk > 1, please use FuscoMoEDispatcher instead"
        )
        ranks_per_node = self.num_local_ranks
        my_rank = torch.distributed.get_rank(group=self.ep_group)
        local_rank = my_rank % ranks_per_node
        my_node = my_rank // ranks_per_node
        nnodes = self.ep_size // ranks_per_node
        num_experts_per_node = self.num_experts // nnodes
        node_expert_begin, node_expert_end = (
            num_experts_per_node * my_node,
            num_experts_per_node * (my_node + 1),
        )
        device = torch.cuda.current_device()

        # [ep_size * seqlen, topk]
        global_indices = gather_along_first_dim(indices, self.ep_group).reshape(
            self.ep_size, seqlen, topk
        )
        global_probs = gather_along_first_dim(probs, self.ep_group).reshape(
            self.ep_size, seqlen, topk
        )

        # ========================= stage 1 =========================
        intergroup = torch.arange(local_rank, self.ep_size, ranks_per_node, device=device)
        # [nNodes, seqlen, topk]
        intragroup_indices = global_indices[intergroup]
        intragroup_probs = global_probs[intergroup]
        intergroup_indices = intragroup_indices // num_experts_per_node * ranks_per_node + (
            my_rank % ranks_per_node
        )

        # [nNodes, seqlen, ep_size]
        intergroup_mapping = torch.zeros(
            size=[nnodes, seqlen, self.ep_size], dtype=torch.int64, device=device
        )
        intergroup_mapping.scatter_(2, intergroup_indices, 1)
        # [seqlen, ep_size]
        intergroup_send_mapping = intergroup_mapping[my_node]

        # [ep_size]
        send_splits_s1 = intergroup_send_mapping.sum(dim=0).to(torch.device("cpu"))
        # int
        send_tokens_s1 = send_splits_s1.sum()
        # [seqlen]
        intergroup_expanded_size = intergroup_send_mapping.sum(dim=1)
        # [nNodes]
        recv_tokens_per_rank_s1 = intergroup_mapping[:, :, my_rank].sum(dim=1)
        # int
        recv_tokens_s1 = recv_tokens_per_rank_s1.sum()

        recvindices_s1 = torch.arange(recv_tokens_s1, dtype=torch.int64, device=device)
        idx = torch.arange(self.ep_size, dtype=torch.int64, device=device)
        recv_splits_s1 = torch.where(
            idx % ranks_per_node == local_rank, recv_tokens_per_rank_s1[idx // ranks_per_node], 0
        ).to(torch.device("cpu"))

        ranks = torch.arange(self.ep_size, dtype=torch.int64, device=device)
        row_mask, col_mask = intergroup_send_mapping.nonzero(as_tuple=True)
        intergroup_send_mapping[row_mask, col_mask] = ranks[col_mask]
        intergroup_send_indices = intergroup_send_mapping[row_mask, col_mask]
        backindices_s1 = torch.argsort(intergroup_send_indices, stable=True)
        pos_mapping_s1 = torch.repeat_interleave(
            torch.arange(seqlen, device=device), intergroup_expanded_size
        )
        sendindices_s1 = pos_mapping_s1[backindices_s1]

        # ========================= stage 2 =========================
        # [ep_size, seqlen, num_experts]
        global_mapping = torch.zeros(
            size=[self.ep_size, seqlen, self.num_experts], dtype=torch.int64, device=device
        )
        global_mapping.scatter_(2, global_indices, 1)
        # [ep_size, num_experts]
        num_global_tokens_per_expert = global_mapping.sum(dim=1)

        # [nNodes * seqlen, topk]
        intragroup_indices = intragroup_indices.reshape(-1, topk)
        intragroup_probs = intragroup_probs.reshape(-1, topk)

        intragroup_mask = (intragroup_indices >= node_expert_begin) & (
            intragroup_indices < node_expert_end
        )
        intragroup_expanded_size = intragroup_mask.sum(dim=1)
        intragroup_expanded_size = intragroup_expanded_size[intragroup_expanded_size != 0]
        mask_indices = intragroup_indices[intragroup_mask]
        intragroup_probs = intragroup_probs[intragroup_mask]

        # [8, num_experts]
        group_send_tokens_per_expert_s2 = num_global_tokens_per_expert.reshape(
            nnodes, ranks_per_node, -1
        ).sum(dim=0)
        # [num_node_experts]
        send_tokens_per_expert_s2 = group_send_tokens_per_expert_s2[
            local_rank, node_expert_begin:node_expert_end
        ]
        # [8]
        send_splits_s2 = (
            send_tokens_per_expert_s2.view(ranks_per_node, self.num_local_experts)
            .sum(dim=1)
            .to(torch.device("cpu"))
        )
        # int
        send_tokens_s2 = send_splits_s2.sum()

        backindices_s2 = torch.argsort(mask_indices, stable=True)
        pos_mapping_s2 = torch.repeat_interleave(
            torch.arange(intragroup_expanded_size.numel(), device=device),
            intragroup_expanded_size,
        )
        sendindices_s2 = pos_mapping_s2[backindices_s2]

        # [8, num_local_experts]
        group_recv_tokens_per_expert_s2 = group_send_tokens_per_expert_s2[
            :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
        ].contiguous()
        # [num_local_experts]
        recv_tokens_per_expert_s2 = group_recv_tokens_per_expert_s2.sum(dim=0)
        # [8]
        recv_tokens_per_rank_s2 = group_recv_tokens_per_expert_s2.sum(dim=1)
        recv_splits_s2 = recv_tokens_per_rank_s2.to(torch.device("cpu"))
        # int
        recv_tokens_s2 = recv_tokens_per_rank_s2.sum()
        if recv_tokens_s2 > 0:
            recvindices_s2 = idxtools.indices_gen(
                group_recv_tokens_per_expert_s2,
                recv_tokens_per_expert_s2,
                recv_tokens_per_rank_s2,
                recv_tokens_s2.item(),
            )
        else:
            recvindices_s2 = torch.empty(0, dtype=torch.int64, device=device)

        assert send_tokens_s2 == intragroup_probs.numel(), (
            f"Mismatch: {send_tokens_s2} vs. {intragroup_probs.numel()}, this may be caused by duplicate experts selected in top-k (i.e., a token selecting the same expert multiple times)."
        )

        self.info = FuscoInfo(
            global_fusco=self.global_fusco,
            sendindices_s1=sendindices_s1,
            recvindices_s1=recvindices_s1,
            backindices_s1=backindices_s1,
            send_splits_s1=send_splits_s1,
            recv_splits_s1=recv_splits_s1,
            send_tokens_s1=send_tokens_s1,
            recv_tokens_s1=recv_tokens_s1,
            seq_len=seqlen,
            intra_fusco=self.intra_fusco,
            sendindices_s2=sendindices_s2,
            recvindices_s2=recvindices_s2,
            backindices_s2=backindices_s2,
            send_splits_s2=send_splits_s2,
            recv_splits_s2=recv_splits_s2,
            send_tokens_s2=send_tokens_s2,
            recv_tokens_s2=recv_tokens_s2,
            intragroup_expanded_size=intragroup_expanded_size,
            intergroup_expanded_size=intergroup_expanded_size,
            probs=intragroup_probs,
        )

        return recv_tokens_per_expert_s2

    def dispatch(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        self.hidden_shape = hidden_states.shape
        assert probs.dim() == 2, "Expected 2D tensor for probs"
        assert indices.dim() == 2, "Expected 2D tensor for token2expert mask"
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

        if self.is_2dmode:
            tokens_per_expert = self.preprocess_2d(indices, probs)
        else:
            tokens_per_expert = self.preprocess_1d(indices, probs)

        tokens_by_expert = FuscoDispatch.apply(hidden_states, self.info, self.apply_probs)

        return tokens_by_expert, tokens_per_expert

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = FuscoCombine.apply(hidden_states, self.info, self.apply_probs)
        return output.view(self.hidden_shape)
