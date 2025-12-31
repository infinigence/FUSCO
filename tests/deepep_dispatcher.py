# The implementation of DeepepMoEDispatcher is adapted from Megatron-LM v0.12.0.

from abc import ABC, abstractmethod
from typing import Optional

import torch

try:
    from deep_ep import Buffer

    Buffer.set_num_sms(24)
    HAVE_DEEP_EP = True
except ImportError:
    HAVE_DEEP_EP = False

_buffer = None


def get_hidden_bytes(x: torch.Tensor) -> int:
    return x.size(1) * max(x.element_size(), 2)


def get_buffer(group: torch.distributed.ProcessGroup, hidden_bytes: int):
    global _buffer
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (
        Buffer.get_dispatch_config(group.size()),
        Buffer.get_combine_config(group.size()),
    ):
        # Split long line for PEP8 compliance
        num_nvl_bytes = max(
            config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes
        )
        num_rdma_bytes = max(
            config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes
        )

    # Allocate buffer if not existed or not enough buffer
    # NOTES: the adaptive routing configuration of the network **must be off**
    if (
        _buffer is None
        or _buffer.group != group
        or _buffer.num_nvl_bytes < num_nvl_bytes
        or _buffer.num_rdma_bytes < num_rdma_bytes
    ):
        _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes, low_latency_mode=True)
    return _buffer


class FusedDispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, token_indices, token_probs, num_experts, group, previous_event=None):
        buffer = get_buffer(group, get_hidden_bytes(x))
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            previous_event,
        ) = buffer.get_dispatch_layout(
            token_indices,
            num_experts,
            previous_event=None,
            async_finish=False,
            allocate_on_comm_stream=False,
        )
        (
            recv_x,
            recv_token_indices,
            recv_token_probs,
            num_recv_tokens_per_expert_list,
            handle,
            event,
        ) = buffer.dispatch(
            x,
            topk_idx=token_indices,
            topk_weights=token_probs,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=None,
            async_finish=False,
            allocate_on_comm_stream=False,
        )

        ctx.group = group
        ctx.handle = handle
        ctx.event = event
        tokens_per_expert = torch.tensor(num_recv_tokens_per_expert_list)

        return (recv_x, recv_token_indices, recv_token_probs, tokens_per_expert, handle)

    @staticmethod
    def backward(
        ctx, grad_output, grad_token_indices, grad_token_probs, grad_tokens_per_expert, grad_handle
    ):
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))
        handle = ctx.handle

        grad_x, grad_token_probs, event = buffer.combine(
            grad_output.contiguous(),
            handle,
            topk_weights=grad_token_probs.float(),
            previous_event=None,
            async_finish=False,
            allocate_on_comm_stream=False,
        )
        return grad_x, None, grad_token_probs, None, None, None


class FusedCombine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group, handle, previous_event=None):
        buffer = get_buffer(group, get_hidden_bytes(x))
        combined_x, _, event = buffer.combine(
            x,
            handle=handle,
            async_finish=False,
            previous_event=None,
            allocate_on_comm_stream=False,
        )
        ctx.handle = handle
        ctx.group = group

        return combined_x, event

    @staticmethod
    def backward(ctx, grad_output, previous_event=None):
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))
        grad_x, _, _, _, _, event = buffer.dispatch(
            grad_output.contiguous(),
            handle=ctx.handle,
            previous_event=previous_event,
            async_finish=False,
            allocate_on_comm_stream=False,
        )
        return grad_x, None, None, None


if HAVE_DEEP_EP:

    def fused_dispatch(x, token_indices, token_probs, num_experts, group, previous_event=None):
        return FusedDispatch.apply(
            x.contiguous(), token_indices, token_probs, num_experts, group, previous_event
        )

    def fused_combine(x, group, handle, previous_event=None):
        return FusedCombine.apply(x, group, handle, previous_event)

else:
    fused_dispatch = None
    fused_combine = None


def permute(
    tokens,
    routing_map,
    num_out_tokens: Optional[int] = None,
    fused: bool = False,
    drop_and_pad: bool = False,
):
    num_tokens, hidden = tokens.shape
    num_experts = routing_map.shape[1]
    if drop_and_pad and num_out_tokens is not None:
        capacity = num_out_tokens // num_experts
        assert not routing_map.requires_grad
        routing_map = routing_map.to(dtype=torch.int8).T.contiguous()
        sorted_indices = routing_map.argsort(dim=-1, descending=True, stable=True)[
            :, :capacity
        ].contiguous()
        sorted_indices = sorted_indices.view(-1)
    else:
        routing_map = routing_map.bool().T.contiguous()

        token_indices = (
            torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_experts, -1)
        )
        sorted_indices = token_indices.masked_select(routing_map)

    permuted_input = tokens.index_select(0, sorted_indices)

    return permuted_input, sorted_indices


def unpermute(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    restore_shape: torch.Size,
    probs: torch.Tensor = None,
    routing_map: torch.Tensor = None,
    fused: bool = False,
    drop_and_pad: bool = False,
):
    _, hidden = restore_shape
    input_dtype = permuted_tokens.dtype

    if probs is not None:
        assert routing_map is not None, "Mask must be provided to permute the probs."
        if drop_and_pad:
            num_experts = routing_map.size(1)
            num_permuted_tokens = sorted_indices.size(0)
            capacity = num_permuted_tokens // num_experts
            num_unpermuted_tokens = probs.size(0)

            probs_T_1D = probs.T.contiguous().view(-1)

            indices_dim0 = torch.arange(num_experts, device=routing_map.device).unsqueeze(-1)
            indices_dim1 = sorted_indices.view(num_experts, capacity)
            indices_1D = (indices_dim0 * num_unpermuted_tokens + indices_dim1).view(-1)

            permuted_probs = probs_T_1D.index_select(0, indices_1D)
        else:
            permuted_probs = probs.T.contiguous().masked_select(routing_map.T.contiguous())
        permuted_tokens = permuted_tokens * permuted_probs.unsqueeze(-1)

    output_tokens = torch.zeros(
        restore_shape, dtype=permuted_tokens.dtype, device=permuted_tokens.device
    )
    output_tokens.scatter_add_(0, sorted_indices.unsqueeze(1).expand(-1, hidden), permuted_tokens)
    return output_tokens.to(dtype=input_dtype)


class _DispatchManager(ABC):
    @abstractmethod
    def setup_metadata(self, routing_map: torch.Tensor, probs: torch.Tensor):
        pass

    @abstractmethod
    def dispatch(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_dispached_metadata(self) -> torch.Tensor:
        pass

    @abstractmethod
    def get_permuted_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_restored_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pass


class _DeepepManager(_DispatchManager):
    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool = False,
        capacity_factor: Optional[float] = None,
        num_experts: Optional[int] = None,
        num_local_experts: Optional[int] = None,
        router_dtype: Optional[str] = None,
    ):
        self.group = group
        self.router_topk = router_topk
        self.capacity_factor = capacity_factor
        self.permute_fusion = permute_fusion
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.router_dtype = router_dtype

        # Metadata
        self.token_indices: Optional[torch.Tensor] = None
        self.token_probs: Optional[torch.Tensor] = None
        # Handle used for combine operation
        self.handle = None

        if fused_dispatch is None:
            raise ImportError(
                "DeepEP is not installed. Please install DeepEP package from "
                "https://github.com/deepseek-ai/deepep."
            )

    def setup_metadata(self, routing_map: torch.Tensor, probs: torch.Tensor):
        self.token_probs = probs
        self.token_indices = routing_map

    def dispatch(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.token_probs.dtype != torch.float32:
            self.token_probs = self.token_probs.float()
        hidden_states, dispatched_indices, dispatched_probs, num_tokens_per_expert, handle = (
            fused_dispatch(
                hidden_states, self.token_indices, self.token_probs, self.num_experts, self.group
            )
        )
        self.handle = handle
        self.tokens_per_expert = num_tokens_per_expert
        self.dispatched_indices = dispatched_indices
        self.dispatched_probs = dispatched_probs

        return hidden_states

    def _indices_to_multihot(self, indices, probs):
        batch_size = indices.shape[0]
        multihot_routing_map = torch.zeros(
            (batch_size, self.num_local_experts), dtype=torch.long, device=indices.device
        )

        multihot_probs = torch.zeros(
            (batch_size, self.num_local_experts), dtype=torch.float, device=indices.device
        )
        mask = indices != -1
        valid_indices = indices[mask]
        row_indices = torch.arange(batch_size, device=indices.device).repeat_interleave(
            mask.sum(dim=1)
        )
        multihot_routing_map[row_indices, valid_indices] = 1
        multihot_probs[row_indices, valid_indices] = probs[mask]
        return multihot_routing_map.bool(), multihot_probs

    def get_dispached_metadata(self) -> torch.Tensor:
        return self.dispatched_indices, self.dispatched_probs

    def get_number_of_tokens_per_expert(self) -> torch.Tensor:
        return self.tokens_per_expert

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, event = fused_combine(hidden_states, self.group, self.handle)
        self.handle = None
        return hidden_states

    def get_permuted_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.dispatched_routing_map, self.dispatched_probs = self._indices_to_multihot(
            self.dispatched_indices, self.dispatched_probs
        )
        self.hidden_shape_before_permute = hidden_states.shape
        hidden_states, self.reversed_mapping_for_combine = permute(
            hidden_states,
            self.dispatched_routing_map,
            num_out_tokens=sum(self.tokens_per_expert),
            fused=self.permute_fusion,
        )
        return hidden_states

    def get_restored_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert self.dispatched_probs.dtype == torch.float32, "DeepEP only supports float32 probs"
        if self.router_dtype == "fp64":
            self.dispatched_probs = self.dispatched_probs.to(torch.float64)
        hidden_states = unpermute(
            hidden_states,
            self.reversed_mapping_for_combine,
            restore_shape=self.hidden_shape_before_permute,
            routing_map=self.dispatched_routing_map,
            probs=self.dispatched_probs,
            fused=self.permute_fusion,
        )
        return hidden_states


class DeepEPMoEDispatcher:
    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: list[int],
        ep_size: int,
        ep_group: torch.distributed.ProcessGroup,
        router_topk: int,
    ):
        self.num_local_experts = num_local_experts
        self.num_experts = ep_size * num_local_experts
        self.local_expert_indices = local_expert_indices
        self.ep_size = ep_size
        self.ep_group = ep_group
        self._comm_manager = _DeepepManager(
            group=self.ep_group,
            router_topk=router_topk,
            permute_fusion=False,
            capacity_factor=None,
            num_experts=self.num_experts,
            num_local_experts=self.num_local_experts,
            router_dtype="fp32",
        )

    def set_shared_experts(self, shared_experts):
        raise NotImplementedError(
            "Shared expert overlap is not supported in Flex Token Dispatcher."
        )

    def _initialize_metadata(self, routing_map: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        num_local_tokens = routing_map.shape[0]
        world_size = self.ep_size
        routing_map = (
            routing_map.reshape(num_local_tokens, self.ep_size, 1, self.num_local_experts)
            .expand(-1, -1, 1, -1)
            .reshape(num_local_tokens, world_size, self.num_local_experts)
        ).contiguous()
        probs = (
            probs.reshape(num_local_tokens, self.ep_size, 1, self.num_local_experts)
            .expand(-1, -1, 1, -1)
            .reshape(num_local_tokens, world_size, self.num_local_experts)
        ).contiguous()
        return routing_map, probs

    def dispatch(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor
    ) -> torch.Tensor:
        self.hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

        self._comm_manager.setup_metadata(routing_map, probs)
        hidden_states = self._comm_manager.dispatch(hidden_states)
        global_input_tokens = self._comm_manager.get_permuted_hidden_states_by_experts(
            hidden_states
        )
        tokens_per_expert = self._comm_manager.get_number_of_tokens_per_expert()

        return global_input_tokens, tokens_per_expert

    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        buffer = self._comm_manager.get_restored_hidden_states_by_experts(hidden_states)
        output = self._comm_manager.combine(buffer)
        return output.view(self.hidden_shape)
