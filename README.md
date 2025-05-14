# CUDA to HIP in PyTorch

This repository is meant to show how to port a simple PyTorch program that uses a custom CUDA kernel to HIP.

## Build Instuctions

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

## Testing

```bash
# Run test.
(venv)$ python3 tests/test_mm.py
Custom kernel avg. time over 50 runs:  0.294511 s
PyTorch matmul avg. time over 50 runs: 0.009403 s
Max error: 3.113e-03
```
