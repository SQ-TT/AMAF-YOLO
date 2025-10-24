import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch

num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
print(f"CUDA is available: {torch.cuda.is_available()}")
current_gpu = torch.cuda.current_device()
print(f"Current GPU device: {current_gpu}")
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())