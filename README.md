# FUSCO

High-performance distributed data shuffling (all-to-all) library for MoE training and inference

## Quick Start

### Hardware Requirements

FUSCO builds on NCCL, leveraging its network abstraction layer to support diverse cluster networks.

- NVIDIA GPUs with CUDA 11.8+ (e.g., A100, H100)
- High-speed interconnects (e.g., NVLink, PCIe, InfiniBand/RoCE, TCP/IP)

### Software Requirements

- Python 3.10+
- CUDA 11.8+
- Ubuntu 20.04+ or compatible Linux distribution (GLIBC 2.27+)

### Installation

FUSCO consists of a Python package and a precompiled shared library `libfusco.so`.

You can install the package into your current Python environment (a small index generation kernel will also be compiled during installation):

```bash
python setup.py install
```

Alternatively, for a clean and isolated environment, you can install the package using `uv`: 

```bash
pip install uv
uv venv
source .venv/bin/activate
uv pip install . -v
```

Current repository only contains the Python package and a small kernel for index generation, and does not include the shared library's source as we are still working on organizing it.

You can download the precompiled `libfusco.so` from the releases page: https://github.com/infinigence/FUSCO/releases.
After downloading, place `libfusco.so` in the `lib` directory at the root of this repository.

```bash
mkdir -p lib
curl -L -o lib/libfusco.so https://ghfast.top/https://github.com/infinigence/FUSCO/releases/download/v0.1/libfusco.so
```

### Test Example

```bash
python tests/test_fusco.py

# This script runs DeepEP/NCCL. Please ensure they are installed and properly set up in your environment.
python tests/test_comparison.py
```

## Performance

We benchmark FUSCO on 64 H100 GPUs (~480 GB/s NVLink) across 8 servers, each with 10×400 Gb/s RoCE NICs (~50 GB/s each), using PyTorch 2.7.0. Dispatch and combine are tested on the DeepSeek-V3 MoE setup (7168 hidden, top-8 experts) under different routing scenarios.

**The latency is measured by the total time of pre-MoE and post-MoE data permutation and the dispatch/combine communication.**

Using routing results of inference of DeepSeek-V3:

| Length (tokens per GPU) |   NCCL   |  DeepEP  |  FUSCO   |
| :---------------------: | :------: | :------: | :------: |
|          4096           | 44.4 ms  | 30.7 ms  | 27.2 ms  |
|          8192           | 80.2 ms  | 58.4 ms  | 50.1 ms  |
|          16384          | 143.6 ms | 116.1 ms | 86.8 ms  |
|          32768          | 266.4 ms | 219.6 ms | 164.4 ms |

Per-token routing confined to a single node:

| Length (tokens per GPU) |   NCCL   |  DeepEP  |  FUSCO  |
| :---------------------: | :------: | :------: | :-----: |
|          4096           | 44.0 ms  | 24.7 ms  | 12.6 ms |
|          8192           | 81.4 ms  | 43.8 ms  | 22.3 ms |
|          16384          | 150.1 ms | 81.8 ms  | 41.0 ms |
|          32768          | 274.8 ms | 144.2 ms | 71.5 ms |

Load imbalance:

| Length (tokens per GPU) |   NCCL   |  DeepEP  |  FUSCO   |
| :---------------------: | :------: | :------: | :------: |
|          4096           | 86.2 ms  | 56.0 ms  | 43.3 ms  |
|          8192           | 167.5 ms | 108.5 ms | 78.9 ms  |
|          16384          | 339.3 ms | 216.2 ms | 151.3 ms |
|          32768          | 656.2 ms | 420.9 ms | 305.0 ms |



## Citation

If you find FUSCO helpful, please cite the paper:

```bibtex
@article{zhu2025fusco,
	title={FUSCO: High-Performance Distributed Data Shuffling via Transformation-Communication Fusion}, 
	author={Zhu, Zhuoran and Zhu, Chunyang and Lin, Hao and Fu, Xu and Zhou, Yiming and Zhang, Quanlu and Li, Zhenhua and Qian, Feng and Yu, Chao and Li, Boxun and Dai, Guohao and Wang, Yu},
	journal={arXiv preprint arXiv:2512.22036},
	year={2025},
}
```
