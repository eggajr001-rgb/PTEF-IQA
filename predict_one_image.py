import argparse
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from models.maniqa import MANIQA

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, required=True, help='测试图片的绝对或相对路径')
    parser.add_argument('--ckpt_path', type=str, required=True, help='比如 ptef_seed43_best.pt')
    return parser.parse_args()


class Config:

    def __init__(self):
        self.embed_dim = 768
        self.num_outputs = 1
        self.dim_mlp = 768
        self.patch_size = 8
        self.img_size = 224
        self.window_size = 4
        self.depths = [2, 2]
        self.num_heads = [4, 4]
        self.num_tab = 2
        self.scale = 0.8


        self.num_avg_val = 5
        self.sigma = 0.1


def main():
    args = parse_args()
    config = Config()

    print(">>> 正在初始化 PTEF (HPSM+SCSA+SRA) 模型...")
    net = MANIQA(
        embed_dim=config.embed_dim,
        num_outputs=config.num_outputs,
        dim_mlp=config.dim_mlp,
        patch_size=config.patch_size,
        img_size=config.img_size,
        window_size=config.window_size,
        depths=config.depths,
        num_heads=config.num_heads,
        num_tab=config.num_tab,
        scale=config.scale
    )

    net = torch.nn.DataParallel(net).cuda()

    print(f">>> 准备加载权重: {args.ckpt_path}")
    if not os.path.exists(args.ckpt_path):
        print("找不到权重文件，检查路径。")
        return


    net.module.load_state_dict(torch.load(args.ckpt_path), strict=False)
    net.eval()

    print(f">>> 读取并处理图片: {args.img_path}")
    img = Image.open(args.img_path).convert('RGB')


    transform = transforms.Compose([
        transforms.FiveCrop(config.img_size),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    img_tensor = transform(img)

    with torch.no_grad():
        final_list = []
        base_list = []
        delta_list = []

        for i in range(config.num_avg_val):
            x_crop = img_tensor[i].unsqueeze(0).cuda()

            # 开启 return_delta_score=True 钩子，单独把 SRA 的残差值拿出来
            final_s, _, delta_s = net(x_crop, return_delta_score=True)

            final_val = final_s.item()
            delta_val = delta_s.item()


            base_val = final_val - (config.sigma * delta_val)

            final_list.append(final_val)
            base_list.append(base_val)
            delta_list.append(delta_val)

        # 把5个crop的结果求个均值
        avg_base = np.mean(base_list)
        avg_delta = np.mean(delta_list)
        avg_final = np.mean(final_list)

    # 终端打印结果，直观展示
    print("\n---------------------------------------------------------")
    print(" PTEF 预测结果 (基于 5-Crop 测试)")
    print("---------------------------------------------------------")
    print(f" Phase I  基础感知分数 (S_base) : {avg_base:.4f}")
    print(f" Phase II SRA补偿幅度 (Δs)      : {avg_delta:.4f}")
    print(f"          实际校准值 (σ*Δs)     : {avg_delta * config.sigma:+.4f} (σ={config.sigma})")
    print(" - - - - - - - - - - - - - - - - - - - - - - - - - - - - ")
    print(f" 模型最终输出质量分 (S_final)   : {avg_final:.4f}")
    print("---------------------------------------------------------\n")


if __name__ == '__main__':
    main()