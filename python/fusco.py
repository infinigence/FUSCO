import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp
from typing import Dict, Optional, List

from shared_lib import (
    SharedLib,
    buffer_type,
    cudaStream_t,
    ncclComm_t,
    ncclDataTypeEnum,
    ncclRedOpTypeEnum,
    ncclUniqueId,
)

class FUSCO:
    def __init__(
        self,
        group_ranks: List[List[int]],
        library_path: str,
    ):
        assert dist.is_initialized()
        global_rank = dist.get_rank()
        device = torch.device('cuda', torch.cuda.current_device())

        for ranks in group_ranks:
            cpu_group = torch.distributed.new_group(ranks, backend="gloo")
            if global_rank in ranks:
                self.cpu_group = cpu_group
        self.local_rank = dist.get_rank(self.cpu_group)
        self.local_size = dist.get_world_size(self.cpu_group)

        try:
            self.shared_lib = SharedLib(library_path)
        except Exception:
            assert False, "Failed to load Shared library."

        if self.local_rank == 0:
            self.unique_id = self.shared_lib.ncclGetUniqueId()
        else:
            self.unique_id = ncclUniqueId()

        tensor = torch.ByteTensor(list(self.unique_id.internal))
        ranks = dist.get_process_group_ranks(self.cpu_group)

        dist.broadcast(tensor, src=ranks[0], group=self.cpu_group)
        byte_list = tensor.tolist()
        for i, byte in enumerate(byte_list):
            self.unique_id.internal[i] = byte
        self.device = device

        with torch.cuda.device(device):
            self.comm: ncclComm_t = self.shared_lib.ncclCommInitRank(
                self.local_size, self.unique_id, self.local_rank
            )
            self.stream = torch.cuda.Stream()

    def all_to_all(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        recvindices: torch.Tensor,
        sendindices: torch.Tensor,
        recv_splits: torch.Tensor,
        send_splits: torch.Tensor,
        stream=None,
    ):
        assert output.is_contiguous(), "output must be contiguous"
        assert input.is_contiguous(), "input must be contiguous"
        assert recvindices.is_contiguous(), "recvindices must be contiguous"
        assert sendindices.is_contiguous(), "sendindices must be contiguous"
        assert input.dim() == 2, "input must be a 2D tensor"
        assert recv_splits.dim() == 1, "recv_splits must be a 1D tensor"
        assert send_splits.dim() == 1, "send_splits must be a 1D tensor"
        assert recv_splits.numel() == self.local_size, "recv_splits must have the same number of elements as ranks in the group"
        assert send_splits.numel() == self.local_size, "send_splits must have the same number of elements as ranks in the group"
        assert sendindices.element_size() == 8, "sendindices must be a 64-bit integer"
        assert recvindices.element_size() == 8, "recvindices must be a 64-bit integer"
        assert recv_splits.device.type == "cpu", "recv_splits must be on CPU"
        assert send_splits.device.type == "cpu", "send_splits must be on CPU"
        if input.numel() == 0 and output.numel() == 0:
            return

        data = input if input.numel() != 0 else output
        hidden_size = data.shape[-1]
        token_size = hidden_size * data.element_size()
        num_ranks = recv_splits.numel()
        sendoffset, recvoffset = 0, 0
        use_stream = self.stream if stream is None else stream

        self.shared_lib.ncclGroupStart()
        for i in range(num_ranks):
            sendcounts, recvcounts = send_splits[i].item(), recv_splits[i].item()
            if sendcounts > 0:
                assert sendoffset + sendcounts <= sendindices.numel(), f"sendindices overflow: {sendoffset + sendcounts} > {sendindices.numel()}"
                ptr = sendindices.data_ptr() + sendoffset * 8
                self.shared_lib.gatherSend(
                    buffer_type(input.data_ptr()),
                    sendcounts * hidden_size,
                    ncclDataTypeEnum.from_torch(input.dtype),
                    i,
                    self.comm,
                    cudaStream_t(use_stream.cuda_stream),
                    buffer_type(ptr),
                    token_size,
                )
                sendoffset += sendcounts
            if recvcounts > 0:
                assert recvoffset + recvcounts <= recvindices.numel(), f"recvindices overflow: {recvoffset + recvcounts} > {recvindices.numel()}"
                ptr = recvindices.data_ptr() + recvoffset * 8
                self.shared_lib.scatterRecv(
                    buffer_type(output.data_ptr()),
                    recvcounts * hidden_size,
                    ncclDataTypeEnum.from_torch(output.dtype),
                    i,
                    self.comm,
                    cudaStream_t(use_stream.cuda_stream),
                    buffer_type(ptr),
                    token_size,
                )
                recvoffset += recvcounts
        self.shared_lib.ncclGroupEnd()
        
