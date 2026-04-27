import os, sys
try:
    for p in sys.path:
        nvrtc_path = os.path.join(p, 'nvidia', 'cuda_nvrtc', 'bin')
        if os.path.exists(nvrtc_path):
            os.add_dll_directory(nvrtc_path)
            os.environ['CUDA_PATH'] = os.path.dirname(nvrtc_path)
except Exception: pass
import cupy as cp
kernel_code = 'extern "C" { __global__ void test() {} }'
cp.RawKernel(kernel_code, 'test')
print('Success')
