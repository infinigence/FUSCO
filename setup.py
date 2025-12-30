import os
import re

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_nvcc_flags() -> list[str]:
    """Compute nvcc flags based on the installed CUDA version.

    This mirrors the logic in idxtools/setup.py so that the extension
    builds identically when installed via the root project.
    """

    cuda_home = os.getenv("CUDA_HOME") or os.getenv("CUDA_PATH") or "/usr/local/cuda"
    if not os.path.exists(cuda_home):
        raise EnvironmentError("CUDA_HOME is not set correctly or CUDA is not installed.")

    nvcc_path = os.path.join(cuda_home, "bin", "nvcc")
    if not os.path.exists(nvcc_path):
        raise EnvironmentError(f"nvcc compiler not found in {nvcc_path}.")

    cuda_version_str = os.popen(f"{nvcc_path} --version").read()
    match = re.search(r"release (\d+)\.(\d+)", cuda_version_str)
    if not match:
        raise EnvironmentError("Could not determine CUDA version from nvcc output.")

    major, minor = int(match.group(1)), int(match.group(2))

    flags: list[str] = [
        "-O3",
        "-Xptxas=-O3",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF_OPERATORS__",
    ]

    cuda8_gencodes = [
        "-gencode=arch=compute_50,code=sm_50",
        "-gencode=arch=compute_60,code=sm_60",
        "-gencode=arch=compute_61,code=sm_61",
    ]
    cuda9_gencodes = ["-gencode=arch=compute_70,code=sm_70"]
    cuda10_gencodes = ["-gencode=arch=compute_75,code=sm_75"]
    cuda11_gencodes = ["-gencode=arch=compute_80,code=sm_80"]
    cuda12_gencodes = ["-gencode=arch=compute_90,code=sm_90"]
    cuda12_8_gencodes = [
        "-gencode=arch=compute_100,code=sm_100",
        "-gencode=arch=compute_120,code=sm_120",
    ]
    cuda13_gencodes = ["-gencode=arch=compute_110,code=sm_110"]

    cuda8_ptx = ["-gencode=arch=compute_61,code=compute_61"]
    cuda9_ptx = ["-gencode=arch=compute_70,code=compute_70"]
    cuda11_ptx = ["-gencode=arch=compute_80,code=compute_80"]
    cuda12_ptx = ["-gencode=arch=compute_90,code=compute_90"]
    cuda13_ptx = ["-gencode=arch=compute_120,code=compute_120"]

    gencode_flags: list[str] = []
    if major >= 13:
        gencode_flags += (
            cuda10_gencodes
            + cuda11_gencodes
            + cuda12_gencodes
            + cuda12_8_gencodes
            + cuda13_gencodes
            + cuda13_ptx
        )
    elif major == 12 and minor >= 8:
        gencode_flags += (
            cuda8_gencodes
            + cuda9_gencodes
            + cuda11_gencodes
            + cuda12_gencodes
            + cuda12_8_gencodes
            + cuda13_ptx
        )
    elif (major == 11 and minor >= 8) or (major == 12 and minor < 8):
        gencode_flags += (
            cuda8_gencodes
            + cuda9_gencodes
            + cuda11_gencodes
            + cuda12_gencodes
            + cuda12_ptx
        )
    elif major >= 11:
        gencode_flags += cuda8_gencodes + cuda9_gencodes + cuda11_gencodes + cuda11_ptx
    elif major >= 9:
        gencode_flags += cuda8_gencodes + cuda9_gencodes + cuda9_ptx
    else:
        gencode_flags += cuda8_gencodes + cuda8_ptx

    # Allow override via NVCC_FLAGS environment variable
    set_gencode_flags = os.getenv("NVCC_FLAGS")
    if set_gencode_flags:
        # If user sets NVCC_FLAGS, respect it entirely
        gencode_flags = set_gencode_flags.split()

    flags += gencode_flags

    print(f"Detected CUDA version: {major}.{minor}")
    print(f"Using nvcc flags: {' '.join(flags)}")

    return flags


setup(
    # Install the Python package `fusco` from the project root
    packages=find_packages(where="."),
    package_dir={"": "."},

    ext_modules=[
        CUDAExtension(
            name="idxtools",  # importable as `import idxtools`
            sources=["idxtools/indices_generator.cu"],
            extra_compile_args={
                "nvcc": get_nvcc_flags(),
                "cxx": ["-O2"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)