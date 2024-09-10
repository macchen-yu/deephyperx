import torch
import torch.nn as nn

# 定义一个简单的3D卷积层
dilation = 1  # 可以更改此值来观察不同的扩展效果
conv3d_layer = nn.Conv3d(1, 20, (3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=0)

# 创建一个示例输入张量 (1, 1, 10, 10, 10)
# 这里 (1, 1, 10, 10, 10) 分别是 (批次大小, 输入通道数, 深度, 高度, 宽度)
input_tensor = torch.randn(1, 1, 10, 10, 10)

# 通过3D卷积层得到输出
output_tensor = conv3d_layer(input_tensor)

print(f"Input shape: {input_tensor.shape}")
print(f"Output shape: {output_tensor.shape}")
