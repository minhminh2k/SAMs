import torch

# Tạo một chuỗi ngẫu nhiên từ 0 đến 9
n = 10
random_indices = torch.randperm(7)[0:5]
print(random_indices)
