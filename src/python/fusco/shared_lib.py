import ctypes
import logging
import platform
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch.distributed import ReduceOp

logger = logging.getLogger(__name__)

ncclResult_t = ctypes.c_int
ncclComm_t = ctypes.c_void_p
ncclWindow_t = ctypes.c_void_p


class ncclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


cudaStream_t = ctypes.c_void_p
buffer_type = ctypes.c_void_p

ncclDataType_t = ctypes.c_int


class ncclDataTypeEnum:
    ncclInt8 = 0
    ncclChar = 0
    ncclUint8 = 1
    ncclInt32 = 2
    ncclInt = 2
    ncclUint32 = 3
    ncclInt64 = 4
    ncclUint64 = 5
    ncclFloat16 = 6
    ncclHalf = 6
    ncclFloat32 = 7
    ncclFloat = 7
    ncclFloat64 = 8
    ncclDouble = 8
    ncclBfloat16 = 9
    ncclNumTypes = 10

    @classmethod
    def from_torch(cls, dtype: torch.dtype) -> int:
        if dtype == torch.int8:
            return cls.ncclInt8
        if dtype == torch.uint8:
            return cls.ncclUint8
        if dtype == torch.int32:
            return cls.ncclInt32
        if dtype == torch.int64:
            return cls.ncclInt64
        if dtype == torch.float16:
            return cls.ncclFloat16
        if dtype == torch.float32:
            return cls.ncclFloat32
        if dtype == torch.float64:
            return cls.ncclFloat64
        if dtype == torch.bfloat16:
            return cls.ncclBfloat16
        raise ValueError(f"Unsupported dtype: {dtype}")


ncclRedOp_t = ctypes.c_int


class ncclRedOpTypeEnum:
    ncclSum = 0
    ncclProd = 1
    ncclMax = 2
    ncclMin = 3
    ncclAvg = 4
    ncclNumOps = 5

    @classmethod
    def from_torch(cls, op: ReduceOp) -> int:
        if op == ReduceOp.SUM:
            return cls.ncclSum
        if op == ReduceOp.PRODUCT:
            return cls.ncclProd
        if op == ReduceOp.MAX:
            return cls.ncclMax
        if op == ReduceOp.MIN:
            return cls.ncclMin
        if op == ReduceOp.AVG:
            return cls.ncclAvg
        raise ValueError(f"Unsupported op: {op}")


@dataclass
class Function:
    name: str
    restype: Any
    argtypes: list[Any]


class SharedLib:
    exported_functions = [
        Function(
            "gatherSend",
            ncclResult_t,
            [
                buffer_type,
                ctypes.c_size_t,
                ncclDataType_t,
                ctypes.c_int,
                ncclComm_t,
                cudaStream_t,
                buffer_type,
                ctypes.c_int,
            ],
        ),
        Function(
            "scatterRecv",
            ncclResult_t,
            [
                buffer_type,
                ctypes.c_size_t,
                ncclDataType_t,
                ctypes.c_int,
                ncclComm_t,
                cudaStream_t,
                buffer_type,
                ctypes.c_int,
            ],
        ),
        Function("ncclGroupStart", ncclResult_t, []),
        Function("ncclGroupEnd", ncclResult_t, []),
        Function("ncclGetErrorString", ctypes.c_char_p, [ncclResult_t]),
        Function("ncclGetUniqueId", ncclResult_t, [ctypes.POINTER(ncclUniqueId)]),
        Function(
            "ncclCommInitRank",
            ncclResult_t,
            [ctypes.POINTER(ncclComm_t), ctypes.c_int, ncclUniqueId, ctypes.c_int],
        ),
    ]

    path_to_library_cache: dict[str, Any] = {}

    # class attribute to store the mapping from library path
    #  to the corresponding dictionary
    path_to_dict_mapping: dict[str, dict[str, Any]] = {}

    def __init__(self, so_file: Optional[str] = None):
        try:
            if so_file not in SharedLib.path_to_dict_mapping:
                lib = ctypes.CDLL(so_file)
                SharedLib.path_to_library_cache[so_file] = lib
            self.lib = SharedLib.path_to_library_cache[so_file]
        except Exception as e:
            logger.error(
                "Failed to load shared library from %s .",
                so_file,
                platform.platform(),
            )
            raise e

        if so_file not in SharedLib.path_to_dict_mapping:
            _funcs: dict[str, Any] = {}
            exported_functions = SharedLib.exported_functions
            for func in exported_functions:
                f = getattr(self.lib, func.name)
                f.restype = func.restype
                f.argtypes = func.argtypes
                _funcs[func.name] = f
            SharedLib.path_to_dict_mapping[so_file] = _funcs
        self._funcs = SharedLib.path_to_dict_mapping[so_file]

    def ncclGetErrorString(self, result: ncclResult_t) -> str:
        return self._funcs["ncclGetErrorString"](result).decode("utf-8")

    def NCCL_CHECK(self, result: ncclResult_t) -> None:
        if result != 0:
            error_str = self.ncclGetErrorString(result)
            raise RuntimeError(f"NCCL error: {error_str}")

    def ncclGetUniqueId(self) -> ncclUniqueId:
        unique_id = ncclUniqueId()
        self.NCCL_CHECK(self._funcs["ncclGetUniqueId"](ctypes.byref(unique_id)))
        return unique_id

    def ncclCommInitRank(self, world_size: int, unique_id: ncclUniqueId, rank: int) -> ncclComm_t:
        comm = ncclComm_t()
        self.NCCL_CHECK(
            self._funcs["ncclCommInitRank"](ctypes.byref(comm), world_size, unique_id, rank)
        )
        return comm

    def gatherSend(
        self,
        sendbuff: buffer_type,
        count: int,
        datatype: int,
        dest: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
        segmentIndices: buffer_type,
        bytesPerSegment: int,
    ) -> None:
        self.NCCL_CHECK(
            self._funcs["gatherSend"](
                sendbuff, count, datatype, dest, comm, stream, segmentIndices, bytesPerSegment
            )
        )

    def scatterRecv(
        self,
        recvbuff: buffer_type,
        count: int,
        datatype: int,
        src: int,
        comm: ncclComm_t,
        stream: cudaStream_t,
        segmentIndices: buffer_type,
        bytesPerSegment: int,
    ) -> None:
        self.NCCL_CHECK(
            self._funcs["scatterRecv"](
                recvbuff, count, datatype, src, comm, stream, segmentIndices, bytesPerSegment
            )
        )

    def ncclGroupStart(self) -> None:
        self.NCCL_CHECK(self._funcs["ncclGroupStart"]())

    def ncclGroupEnd(self) -> None:
        self.NCCL_CHECK(self._funcs["ncclGroupEnd"]())


__all__ = [
    "SharedLib",
    "ncclDataTypeEnum",
    "ncclRedOpTypeEnum",
    "ncclUniqueId",
    "ncclComm_t",
    "cudaStream_t",
    "buffer_type",
]
