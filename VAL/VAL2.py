import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

class HamidaEtAl(nn.Module):
    """
    3-D Deep Learning Approach for Remote Sensing Image Classification
    Amina Ben Hamida, Alexandre Benoit, Patrick Lambert, Chokri Ben Amar
    IEEE TGRS, 2018
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8344565
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=5, dilation=1):
        super(HamidaEtAl, self).__init__()
        self.patch_size = patch_size
        self.input_channels = input_channels
        dilation = (dilation, 1, 1)

        if patch_size == 3:
            self.conv1 = nn.Conv3d(
                1, 20, (3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=1
            )
        else:
            self.conv1 = nn.Conv3d(
                1, 20, (3, 3, 3), stride=(1, 1, 1), dilation=dilation, padding=0
            )

        self.pool1 = nn.Conv3d(
            20, 20, (3, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0)
        )
        self.conv2 = nn.Conv3d(
            20, 35, (3, 3, 3), dilation=dilation, stride=(1, 1, 1), padding=(1, 0, 0)
        )
        self.pool2 = nn.Conv3d(
            35, 35, (3, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0)
        )
        self.conv3 = nn.Conv3d(
            35, 35, (3, 1, 1), dilation=dilation, stride=(1, 1, 1), padding=(1, 0, 0)
        )
        self.conv4 = nn.Conv3d(
            35, 35, (2, 1, 1), dilation=dilation, stride=(2, 1, 1), padding=(1, 0, 0)
        )

        self.features_size = self._get_final_flattened_size()
        self.fc = nn.Linear(self.features_size, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, 1, self.input_channels, self.patch_size, self.patch_size)
            )
            x = self.pool1(self.conv1(x))
            x = self.pool2(self.conv2(x))
            x = self.conv3(x)
            x = self.conv4(x)
            _, t, c, w, h = x.size()
        return t * c * w * h

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, self.features_size)
        x = self.fc(x)
        return x

def load_model(weights_path, input_channels, n_classes, patch_size=5):
    model = HamidaEtAl(input_channels, n_classes, patch_size)
    print(f"Model fc layer weight size: {model.fc.weight.size()}")
    model.load_state_dict(torch.load(weights_path))
    print(f"Model fc layer weight size after loading: {model.fc.weight.size()}")
    model.eval()
    return model

def validate_model(model):
    # 随机生成(100, 100, 244)的numpy数组
    random_numpy_input = np.random.rand(3, 3, 244).astype(np.float32)
    # 转换为PyTorch张量
    test_input = torch.from_numpy(random_numpy_input).unsqueeze(0).unsqueeze(0)  # 添加批次和通道维度

    with torch.no_grad():
        output = model(test_input)
    print("Model output:", output)

if __name__ == "__main__":
    weights_path =  "2024_07_23_18_16_37_epoch15_0.99.pth"
    input_channels = 244  # 确保与保存权重时的设置一致
    n_classes = 8  # 确保与保存权重时的设置一致
    patch_size = 3  # 确保与保存权重时的设置一致

    model = load_model(weights_path, input_channels, n_classes, patch_size)
    validate_model(model)
