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


class FuscoMoEDispatcher:
    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: list[int],
        ep_size: int,
        ep_group: torch.distributed.ProcessGroup,
        fusco: FUSCO,
    ):
        self.num_local_experts = num_local_experts
        self.num_experts = ep_size * num_local_experts
        assert self.num_local_experts > 0, "Expected at least one expert"
        self.local_expert_indices = local_expert_indices
        self.ep_size = ep_size
        self.ep_group = ep_group
        self.fusco = fusco
        self.probs = None

    def preprocess(self, indices: torch.Tensor) -> torch.Tensor:
        num_local_tokens_per_expert = torch.bincount(indices.view(-1), minlength=self.num_experts)

        num_local_tokens_per_rank = num_local_tokens_per_expert.view(
            self.ep_size, self.num_local_experts
        ).sum(dim=1)

        topk = indices.size(1)
        flatten_indices = indices.view(-1)

        self.sendindices_unique = torch.argsort(flatten_indices, stable=True).contiguous()
        self.sendindices_with_duplicates = (self.sendindices_unique // topk).contiguous()
        self.send_splits = num_local_tokens_per_rank.to(torch.device("cpu"))

        num_global_tokens_per_expert = gather_along_first_dim(
            num_local_tokens_per_expert, self.ep_group
        ).reshape(self.ep_size, self.num_experts)

        num_global_tokens_per_local_expert = num_global_tokens_per_expert[
            :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
        ].contiguous()

        num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(dim=0)

        num_tokens_per_ep = num_global_tokens_per_local_expert.sum(dim=1)

        self.num_ep_tokens = num_global_tokens_per_local_expert.sum()

        if self.num_ep_tokens > 0:
            self.recvindices = idxtools.indices_gen(
                num_global_tokens_per_local_expert,
                num_tokens_per_local_expert,
                num_tokens_per_ep,
                self.num_ep_tokens.item(),
            )
        else:
            self.recvindices = torch.empty(0, dtype=torch.int64, device=torch.cuda.current_device())
        self.recv_splits = num_tokens_per_ep.to(torch.device("cpu"))

        return num_tokens_per_local_expert

    def dispatch(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        hidden_dim = hidden_states.shape[-1]
        self.probs = probs
        self.indices_shape = indices.shape

        tokens_per_expert = self.preprocess(indices)

        tokens_by_expert = hidden_states.new_empty(
            (self.num_ep_tokens, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        self.fusco.all_to_all(
            tokens_by_expert,
            hidden_states,
            recvindices=self.recvindices,
            sendindices=self.sendindices_with_duplicates,
            recv_splits=self.recv_splits,
            send_splits=self.send_splits,
            stream=torch.cuda.current_stream(),
        )

        return tokens_by_expert, tokens_per_expert

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_dim = hidden_states.shape[-1]
        outputs_unpermuted = hidden_states.new_empty(
            (self.indices_shape[0], self.indices_shape[1], hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        self.fusco.all_to_all(
            outputs_unpermuted,
            hidden_states,
            recvindices=self.sendindices_unique,
            sendindices=self.recvindices,
            recv_splits=self.send_splits,
            send_splits=self.recv_splits,
            stream=torch.cuda.current_stream(),
        )

        outputs_unpermuted = outputs_unpermuted * self.probs.unsqueeze(-1)
        outputs_unpermuted = outputs_unpermuted.sum(dim=1)

        return outputs_unpermuted


class Fusco2DMoEDispatcher:
    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: list[int],
        ep_size: int,
        ep_group: torch.distributed.ProcessGroup,
        global_fusco: FUSCO,
        intra_fusco: FUSCO,
        num_local_ranks: int = 8,
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

    def preprocess(self, indices: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
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
        self.send_splits_s1 = intergroup_send_mapping.sum(dim=0).to(torch.device("cpu"))
        # int
        self.send_tokens_s1 = self.send_splits_s1.sum()
        # [seqlen]
        self.intergroup_expanded_size = intergroup_send_mapping.sum(dim=1)
        # [nNodes]
        recv_tokens_per_rank_s1 = intergroup_mapping[:, :, my_rank].sum(dim=1)
        # int
        self.recv_tokens_s1 = recv_tokens_per_rank_s1.sum()

        self.recvindices_s1 = torch.arange(self.recv_tokens_s1, dtype=torch.int64, device=device)
        idx = torch.arange(self.ep_size, dtype=torch.int64, device=device)
        self.recv_splits_s1 = torch.where(
            idx % ranks_per_node == local_rank, recv_tokens_per_rank_s1[idx // ranks_per_node], 0
        ).to(torch.device("cpu"))

        ranks = torch.arange(self.ep_size, dtype=torch.int64, device=device)
        row_mask, col_mask = intergroup_send_mapping.nonzero(as_tuple=True)
        intergroup_send_mapping[row_mask, col_mask] = ranks[col_mask]
        intergroup_send_indices = intergroup_send_mapping[row_mask, col_mask]
        self.backindices_s1 = torch.argsort(intergroup_send_indices, stable=True)
        pos_mapping_s1 = torch.repeat_interleave(
            torch.arange(seqlen, device=device), self.intergroup_expanded_size
        )
        self.sendindices_s1 = pos_mapping_s1[self.backindices_s1]

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
        self.intragroup_expanded_size = intragroup_expanded_size[intragroup_expanded_size != 0]
        mask_indices = intragroup_indices[intragroup_mask]
        self.intragroup_probs = intragroup_probs[intragroup_mask].unsqueeze(-1)

        # [8, num_experts]
        group_send_tokens_per_expert_s2 = num_global_tokens_per_expert.reshape(
            nnodes, ranks_per_node, -1
        ).sum(dim=0)
        # [num_node_experts]
        send_tokens_per_expert_s2 = group_send_tokens_per_expert_s2[
            local_rank, node_expert_begin:node_expert_end
        ]
        # [8]
        self.send_splits_s2 = (
            send_tokens_per_expert_s2.view(ranks_per_node, self.num_local_experts)
            .sum(dim=1)
            .to(torch.device("cpu"))
        )
        # int
        self.send_tokens_s2 = self.send_splits_s2.sum()

        self.backindices_s2 = torch.argsort(mask_indices, stable=True)
        pos_mapping_s2 = torch.repeat_interleave(
            torch.arange(self.intragroup_expanded_size.numel(), device=device),
            self.intragroup_expanded_size,
        )
        self.sendindices_s2 = pos_mapping_s2[self.backindices_s2]

        # [8, num_local_experts]
        group_recv_tokens_per_expert_s2 = group_send_tokens_per_expert_s2[
            :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
        ].contiguous()
        # [num_local_experts]
        recv_tokens_per_expert_s2 = group_recv_tokens_per_expert_s2.sum(dim=0)
        # [8]
        recv_tokens_per_rank_s2 = group_recv_tokens_per_expert_s2.sum(dim=1)
        self.recv_splits_s2 = recv_tokens_per_rank_s2.to(torch.device("cpu"))
        # int
        self.recv_tokens_s2 = recv_tokens_per_rank_s2.sum()
        if self.recv_tokens_s2 > 0:
            self.recvindices_s2 = idxtools.indices_gen(
                group_recv_tokens_per_expert_s2,
                recv_tokens_per_expert_s2,
                recv_tokens_per_rank_s2,
                self.recv_tokens_s2.item(),
            )
        else:
            self.recvindices_s2 = torch.empty(0, dtype=torch.int64, device=device)

        assert self.send_tokens_s2 == self.intragroup_probs.numel(), (
            f"Mismatch: {self.send_tokens_s2} vs. {self.intragroup_probs.numel()}, this may be caused by duplicate experts selected in top-k (i.e., a token selecting the same expert multiple times)."
        )

        return recv_tokens_per_expert_s2

    def dispatch(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        if probs.dtype != hidden_states.dtype:
            probs = probs.to(hidden_states.dtype)
        tokens_per_expert = self.preprocess(indices, probs)

        hidden_dim = hidden_states.shape[1]
        buffer = hidden_states.new_empty(
            (self.recv_tokens_s1, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        tokens_by_expert = hidden_states.new_empty(
            (self.recv_tokens_s2, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # cross-nodes
        self.global_fusco.all_to_all(
            buffer,
            hidden_states,
            recvindices=self.recvindices_s1,
            sendindices=self.sendindices_s1,
            recv_splits=self.recv_splits_s1,
            send_splits=self.send_splits_s1,
            stream=torch.cuda.current_stream(),
        )

        # intra-nodes
        self.intra_fusco.all_to_all(
            tokens_by_expert,
            buffer,
            recvindices=self.recvindices_s2,
            sendindices=self.sendindices_s2,
            recv_splits=self.recv_splits_s2,
            send_splits=self.send_splits_s2,
            stream=torch.cuda.current_stream(),
        )

        return tokens_by_expert, tokens_per_expert

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_dim = hidden_states.shape[-1]
        outputs_unpermuted = hidden_states.new_empty(
            size=(self.send_tokens_s1, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        outputs_buffer = hidden_states.new_empty(
            size=(self.send_tokens_s2, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # intra-nodes
        self.intra_fusco.all_to_all(
            outputs_buffer,
            hidden_states,
            recvindices=self.backindices_s2,
            sendindices=self.recvindices_s2,
            recv_splits=self.send_splits_s2,
            send_splits=self.recv_splits_s2,
            stream=torch.cuda.current_stream(),
        )

        if self.send_tokens_s2 > 0:
            outputs_buffer = outputs_buffer * self.intragroup_probs
            outputs_buffer = torch.segment_reduce(
                outputs_buffer, lengths=self.intragroup_expanded_size, reduce="sum"
            )

        # cross-nodes
        self.global_fusco.all_to_all(
            outputs_unpermuted,
            outputs_buffer,
            recvindices=self.backindices_s1,
            sendindices=self.recvindices_s1,
            recv_splits=self.send_splits_s1,
            send_splits=self.recv_splits_s1,
            stream=torch.cuda.current_stream(),
        )

        if self.send_tokens_s1 != self.intergroup_expanded_size.numel():
            outputs_unpermuted = torch.segment_reduce(
                outputs_unpermuted, lengths=self.intergroup_expanded_size, reduce="sum"
            )
        return outputs_unpermuted
