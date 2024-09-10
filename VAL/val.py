import torch
import torch.nn as nn


# 定义一个简单的神经网络模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(10, 50)
        self.layer2 = nn.Linear(50, 10)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x


# 加载模型和权重
def load_model(weights_path):
    model = SimpleModel()
    model.load_state_dict(torch.load(weights_path))
    model.eval()  # 设置为评估模式
    return model


# 验证模型
def validate_model(model):
    # 创建一个随机输入张量
    test_input = torch.randn(1, 10)
    with torch.no_grad():
        output = model(test_input)
    print("Model output:", output)


if __name__ == "__main__":
    weights_path = "2024_07_23_18_16_37_epoch15_0.99.pth"
    model = load_model(weights_path)
    validate_model(model)
