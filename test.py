import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

# Tạo tensor mẫu (1054, 1000)
input_tensor = torch.rand(3, 123, 434)  # Thêm batch và channel dimension nếu cần
i1 = torch.rand(4, 123, 434)
i2 = torch.rand(2, 123, 434)

tensor_list = [input_tensor, i1, i2]
concatenated_tensor = torch.cat(tensor_list, dim=0)

print(concatenated_tensor.shape)