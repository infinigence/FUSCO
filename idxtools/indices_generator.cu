#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cassert>
#include <vector>

/*
    rank_expert_tokens_matrix: [ep_size, num_experts], 
                               the numebr of tokens rank A -> expert B
*/ 
__global__ void generator(
    const int64_t *rank_exp_tokens_matrix,
    const int64_t *exp_tokens_array,
    const int64_t *rank_tokens_array,
    int64_t *dst, 
    const int64_t nRanks, 
    const int64_t nExperts, 
    const int64_t len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        int rank = 0, exp = 0;
        int64_t left = 0, right = 0;
        int64_t offset_inter_expert = 0, offset_inter_rank = 0, offset_intra_expert = 0;
        for (; rank < nRanks; ++rank) {
            right = left + rank_tokens_array[rank];
            if (left <= idx && idx < right)
                break;
            left = right;
        }
        for (exp = 0; exp < nExperts; ++exp) {
            right = left + rank_exp_tokens_matrix[rank * nExperts + exp];
            if (left <= idx && idx < right)
                break;
            left = right;
        }
        
        for (int i = 0; i < exp; ++i)
            offset_inter_expert += exp_tokens_array[i];
        for (int i = 0; i < rank; ++i)
            offset_inter_rank += rank_exp_tokens_matrix[i * nExperts + exp];
        offset_intra_expert = idx - left;

        dst[idx] = offset_inter_expert + offset_inter_rank + offset_intra_expert;
    }
}

torch::Tensor indices_gen(torch::Tensor num_ep_tokens_per_local_expert, torch::Tensor num_tokens_per_local_expert, torch::Tensor num_tokens_per_rank, int64_t len) {
    int64_t nRanks = num_ep_tokens_per_local_expert.size(0);
    int64_t nExperts = num_ep_tokens_per_local_expert.size(1);
    torch::Tensor ret = torch::empty({len}, torch::dtype(torch::kInt64).device(num_ep_tokens_per_local_expert.device()));
    int threads = 256;
    int blocks = (len + threads - 1) / threads;
    if (len != 0) {
        generator<<<blocks, threads>>>(
            num_ep_tokens_per_local_expert.data_ptr<int64_t>(),
            num_tokens_per_local_expert.data_ptr<int64_t>(),
            num_tokens_per_rank.data_ptr<int64_t>(),
            ret.data_ptr<int64_t>(),
            nRanks,
            nExperts,
            len
        );
    }
    return ret;
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("indices_gen", &indices_gen, "indices generator");
}