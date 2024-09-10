import os
import subprocess

# 定义基础命令及其通用参数
base_command = "python inference.py --model hamida --cuda 0"
checkpoint = "--checkpoint checkpoints/hamida_et_al/fx17_Fianl/2024_07_24_02_08_30_epoch20_1.00.pth"
patch_size = "--patch_size 7"
n_classes = "--n_classes 8"

# 获取./patch_fx17目录下的所有.npy文件
folder_path = "./patch_fx17"
images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npy')]

# 遍历每个图像文件，生成并执行命令
for image in images:
    # 构建完整的命令
    command = f"{base_command} --image {image} {checkpoint} {patch_size} {n_classes}"

    # 打印命令以供调试（可选）
    print(f"Executing: {command}")

    # 运行命令
    subprocess.run(command, shell=True)
