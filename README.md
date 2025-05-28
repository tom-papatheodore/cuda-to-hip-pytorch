# 🧠 CUDA-to-HIP PyTorch Training & Competition

Welcome to the **CUDA-to-HIP PyTorch Developer Challenge** — a hands-on learning and performance competition designed to teach you how to write custom GPU kernels in CUDA and HIP and use them in PyTorch applications. You’ll start from a fully working matrix multiply package and optimize it for **AMD GPUs using ROCm**.

---

## 📦 Overview

In this training, you’ll work from a minimal working package that integrates a **custom CUDA matrix multiplication kernel** into PyTorch via `pybind11`. The training emphasizes:

- Using **ROCm-enabled PyTorch** to automatically **hipify** your CUDA kernels
- Understanding the PyTorch C++ extension build system
- Writing performance-critical GPU code
- Eventually maintaining **independent CUDA and HIP kernels** for advanced optimization

---

## 🗂 Repository Layout

Only the following directory will be used for this challenge:

```
cuda-to-hip-pytorch
├── gpu-mm
│   ├── install.sh
│   ├── requirements_pypi.txt
│   ├── requirements_pytorch_rocm.txt
│   ├── requirements_pytorch.txt
│   ├── setup.py
│   ├── src
│   │   └── gpu_mm
│   │       ├── bindings.cpp
│   │       ├── __init__.py
│   │       └── matmul.cu
│   └── tests
│       └── test_mm.py
```


---

## 📅 Schedule

### Week 1: **Understanding the Codebase**

- Study the structure of `gpu-mm`
- Learn how `setup.py` builds the package using PyTorch’s C++ extension utilities
- Understand how `matmul.cu` is automatically converted to HIP if ROCm is detected
- Run and examine `test_mm.py` for validation

You’ll learn:
- How `bindings.cpp` exposes the kernel to Python
- The role of `pybind11` and `ATen`
- The layout of a minimal Python extension package

> 🛠 Try building the package on AMD GPUs using ROCm-enabled PyTorch.

```bash
# Clone this repo and cd into cuda-to-hip-pytorch/gpu-mm
$ git clone https://github.com/tom-papatheodore/cuda-to-hip-pytorch.git
$ cd cuda-to-hip-pytorch/gpu-mm

# [OPTIONAL] Create a new Python virtual environment to build, install, and test. Then activate it.
$ python3 -m venv --upgrade-deps venv
$ source venv/bin/activate

# Install dependencies, including ROCm-enabled PyTorch
# (custom URLs require separate requirements files for both CUDA and HIP).
(venv)$ pip install --no-build-isolation -r requirements_pypi.txt
(venv)$ pip install --no-build-isolation -r requirements_pytorch_rocm.txt

# Install gpu_mm package. Here, the ROCm-enabled PyTorch hipifies the matmul.cu file during the build, 
# which creates a matmul.hip file that sits next to the original matmul.cu.
(venv)$ pip install -v --no-build-isolation -e .
... 
Successfully preprocessed all matching files.
Total number of unsupported CUDA function calls: 0
Total number of replaced kernel launches: 1
...
Successfully built gpu_mm
Installing collected packages: gpu_mm
Successfully installed gpu_mm-0.0.0
```

> 🚀 Now try running the test.

```# Run test.
(venv)$ python3 tests/test_mm.py
Custom kernel avg. time over 50 runs:  0.294511 s
PyTorch matmul avg. time over 50 runs: 0.009403 s
Max error: 3.113e-03
```

---

### Week 2: **CUDA Kernel Optimization (Auto-Hipify)**

Now the competition begins! Your task is to optimize the `matmul.cu` kernel to achieve the **fastest runtime** for a fixed-size matrix multiplication problem.

**Rules**:
- Work only in the `gpu-mm` package.
- You are free to modify `matmul.cu`, but you must keep the file in CUDA.
- You may use any optimization strategy that will still auto-hipify successfully.
- Do not rewrite the kernel in `.hip` or introduce separate HIP code — that comes later.

> 🔬 You’ll submit your optimized version and benchmark results using a provided harness.

---

### Week 3–4: **Maintaining Independent CUDA & HIP Kernels**

Once you've reached the limits of hipify, you may find the need to write **native HIP kernels** to take advantage of ROCm-specific optimizations.

At this point, we will introduce new instructions for:
- Adding a `.hip` file with a manually written HIP kernel
- Updating `bindings.cpp` to choose between CUDA and HIP kernels
- Extending `setup.py` to support both code paths

Your final task:
- Maintain separate CUDA and HIP kernels.
- Optimize the HIP kernel for ROCm devices.

You’ll submit:
- Optimized CUDA kernel
- Optimized HIP kernel
- Final benchmark results
- Brief write-up of your approach and findings

---

## 🧪 Benchmarking & Evaluation

You will be evaluated on:
- ✅ Correctness (must pass validation tests)
- ⚡ Performance (fastest wall-time wins)
- 🧠 Insight (your explanation of what you tried and what worked)

> Matrix sizes, precision format, and test harness will be provided separately.

---

## 👥 Teams or Solo

You may work independently or in teams of up to 3 people.

---

## 📎 Getting Started

1. Clone this repository
2. Set up your environment:
   ```bash
   cd cuda-to-hip-pytorch/gpu-mm
   ./install.sh
