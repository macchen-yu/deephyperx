import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

# 要推論的資料夾
numberfolders = 1

# 定义基础命令及其通用参数
base_command = "python inference10_10.py --model hamida --cuda 0"
checkpoint = "--checkpoint checkpoints/hamida_et_al/fx17_Fianl/2024_07_24_02_08_30_epoch20_1.00.pth"
patch_size = "--patch_size 7"
n_classes = "--n_classes 8"
folder_number = f"--folder_number {numberfolders}"

base_folder_path = "./VAL_patch10_10/"
folder_path = base_folder_path + f"{numberfolders}/"

# 检查目录是否存在
if not os.path.exists(folder_path):
    print(f"Error: The folder '{folder_path}' does not exist.")
else:
    # 获取目录下的所有.npy文件
    images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npy')]

    # 定义执行命令的函数
    def run_inference(image):
        command = f"{base_command} {folder_number} --image {image} {checkpoint} {patch_size} {n_classes}"
        print(f"Executing: {command}")
        subprocess.run(command, shell=True)

    # 使用 ThreadPoolExecutor 并行执行任务
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(run_inference, images)
