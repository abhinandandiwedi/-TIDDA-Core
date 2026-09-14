import torch

print("torch:", torch.__version__)
print("HIP:", torch.version.hip)
print("CUDA API available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("\nRunning REAL GPU computation test...")
    # create a reasonably sized torch tensor on the GPU
    A = torch.randn(4096, 4096, device="cuda")
    B = torch.randn(4096, 4096, device="cuda")
    # perform matrix multiplication
    C = torch.matmul(A, B)
    # synchronize
    torch.cuda.synchronize()
    print("SUCCESS: matrix multiplication (4096x4096) completed on", torch.cuda.get_device_name(0))
else:
    print("FAILED: No GPU available for computation test.")
