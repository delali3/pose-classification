# # GPU Detection and Setup Script
# import torch
# import subprocess
# import platform

# print("=" * 50)
# print("GPU DETECTION AND SETUP")
# print("=" * 50)

# # System info
# print(f"Python version: {platform.python_version()}")
# print(f"PyTorch version: {torch.__version__}")
# print(f"GPU name: {torch.cuda.get_device_name(0)}")

# # Check CUDA availability
# cuda_available = torch.cuda.is_available()
# print(f"CUDA available: {cuda_available}")

# if cuda_available:
#     print(f"GPU name: {torch.cuda.get_device_name(0)}")
#     print(f"CUDA version: {torch.version.cuda}")
#     print(f"Number of GPUs: {torch.cuda.device_count()}")
#     device = torch.device("cuda")
#     print(f"Using device: {device}")
    
#     # Test GPU memory
#     print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
# else:
#     print("CUDA not available. Using CPU.")
#     device = torch.device("cpu")
#     print(f"Using device: {device}")
    
#     print("\n" + "=" * 50)
#     print("TO ENABLE GPU ACCELERATION:")
#     print("=" * 50)
#     print("1. Check if you have an NVIDIA GPU:")
#     print("   - Run 'nvidia-smi' in command prompt")
#     print("   - Or check Device Manager > Display adapters")
#     print("\n2. If you have NVIDIA GPU, install CUDA-enabled PyTorch:")
#     print("   pip uninstall torch torchvision torchaudio")
#     print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
#     print("\n3. If no NVIDIA GPU, CPU-only is fine for most tasks")

# # Test a simple tensor operation
# print("\n" + "=" * 30)
# print("TESTING TENSOR OPERATIONS")
# print("=" * 30)

# try:
#     # Create a test tensor
#     x = torch.randn(2, 3).to(device)
#     y = torch.randn(2, 3).to(device)
#     z = x + y
#     print(f"✅ Tensor operations working on {device}")
#     print(f"Sample tensor shape: {z.shape}")
#     print(f"Sample tensor device: {z.device}")
# except Exception as e:
#     print(f"❌ Error with tensor operations: {e}")

# print("\n" + "=" * 50)


import tensorflow as tf
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("GPUs:", tf.config.list_physical_devices('GPU'))
