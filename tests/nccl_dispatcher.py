# The implementation of NCCLMoEDispatcher is adapted from Megatron-LM v0.12.0.

import torch


def gather_along_first_dim(input_, group):
    world_size = torch.distributed.get_world_size(group=group)

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    torch.distributed.all_gather_into_tensor(output, input_.contiguous(), group=group)

    return output


def sort_chunks_by_idxs(
    input: torch.Tensor, split_sizes: torch.Tensor, sorted_idxs: torch.Tensor, fused: bool = False
):
    input = torch.split(input, split_sizes.tolist(), dim=0)
    output = torch.cat([input[i] for i in sorted_idxs.tolist()], dim=0)
    return output


def all_to_all(group, input, output_split_sizes, input_split_sizes):
    world_size = torch.distributed.get_world_size(group=group)
    if world_size == 1:
        return input

    input = input.contiguous()
    output = input.new_empty(
        size=[sum(output_split_sizes)] + list(input.size()[1:]),
        dtype=input.dtype,
        device=torch.cuda.current_device(),
    )
    torch.distributed.all_to_all_single(
        output,
        input,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
    )
    return output


def permute(tokens, indices):
    if indices.dim() == 1:
        topk = 1
    else:
        topk = indices.size(1)
    flatten_indices = indices.view(-1)
    sorted_indices = torch.argsort(flatten_indices, stable=True)
    permuted_tokens = tokens.index_select(0, sorted_indices // topk)
    return permuted_tokens, sorted_indices


def unpermute(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    probs: torch.Tensor = None,
):
    assert sorted_indices.numel() == permuted_tokens.size(0)
    if probs is not None:
        num_unpermuted_tokens = probs.numel()
        topk = probs.size(1)
    else:
        num_unpermuted_tokens = permuted_tokens.size(0)
        topk = 1

    unpermuted_tokens = torch.zeros(
        [num_unpermuted_tokens, permuted_tokens.shape[-1]],
        dtype=permuted_tokens.dtype,
        device=permuted_tokens.device,
    )
    unpermuted_tokens.index_copy_(0, sorted_indices, permuted_tokens)
    if probs is not None:
        unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.size(-1))
        unpermuted_tokens = unpermuted_tokens * probs.unsqueeze(-1)
        unpermuted_tokens = unpermuted_tokens.sum(dim=1)
    return unpermuted_tokens


class NCCLMoEDispatcher:
    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: list[int],
        ep_size: int,
        ep_group: torch.distributed.ProcessGroup,
    ) -> None:
        self.num_local_experts = num_local_experts
        self.num_experts = ep_size * num_local_experts
        self.local_expert_indices = local_expert_indices
        self.ep_size = ep_size
        self.ep_group = ep_group

        input_chunk_idxs = torch.arange(self.num_experts, device=torch.cuda.current_device())
        self.sort_input_by_local_experts = input_chunk_idxs.reshape(
            -1, self.num_local_experts
        ).T.ravel()
        self.restore_output_by_local_experts = input_chunk_idxs.reshape(
            self.num_local_experts, -1
        ).T.ravel()

    def preprocess(self, indices: torch.Tensor) -> torch.Tensor:
        num_local_tokens_per_expert = torch.bincount(indices.view(-1), minlength=self.num_experts)

        self.input_splits = (
            num_local_tokens_per_expert.reshape(self.ep_size, self.num_local_experts)
            .sum(axis=1)
            .to(torch.device("cpu"))
            .numpy()
        )

        num_global_tokens_per_expert = gather_along_first_dim(
            num_local_tokens_per_expert, self.ep_group
        ).reshape(self.ep_size, self.num_experts)

        num_global_tokens_per_local_expert = num_global_tokens_per_expert[
            :, self.local_expert_indices[0] : self.local_expert_indices[-1] + 1
        ].contiguous()

        num_global_tokens_per_rank = num_global_tokens_per_local_expert.sum(axis=1)

        self.output_splits = num_global_tokens_per_rank.to(torch.device("cpu")).numpy()

        num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(dim=0)

        num_tokens_per_local_expert = num_tokens_per_local_expert.to(torch.device("cpu"))

        if self.num_local_experts > 1:
            self.num_global_tokens_per_local_expert = num_global_tokens_per_local_expert.view(
                -1, self.num_local_experts
            )

        return num_tokens_per_local_expert

    def dispatch(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.hidden_shape = hidden_states.shape
        self.probs = probs
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
        tokens_per_expert = self.preprocess(indices)

        self.hidden_shape_before_permute = hidden_states.shape
        permutated_local_input_tokens, self.reversed_local_input_permutation_mapping = permute(
            hidden_states,
            indices,
        )

        global_input_tokens = all_to_all(
            self.ep_group, permutated_local_input_tokens, self.output_splits, self.input_splits
        )

        if self.num_local_experts > 1:
            global_input_tokens = sort_chunks_by_idxs(
                global_input_tokens,
                self.num_global_tokens_per_local_expert.ravel(),
                self.sort_input_by_local_experts,
                fused=False,
            )
        return global_input_tokens, tokens_per_expert

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.num_local_experts > 1:
            hidden_states = sort_chunks_by_idxs(
                hidden_states,
                self.num_global_tokens_per_local_expert.T.ravel(),
                self.restore_output_by_local_experts,
                fused=False,
            )

        permutated_local_input_tokens = all_to_all(
            self.ep_group, hidden_states, self.input_splits, self.output_splits
        )

        output = unpermute(
            permutated_local_input_tokens,
            self.reversed_local_input_permutation_mapping,
            probs=self.probs,
        )

        output = output.view(self.hidden_shape)

        return output
