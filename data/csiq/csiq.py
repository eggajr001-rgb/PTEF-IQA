import os
import torch
import numpy as np
import cv2
import pandas as pd


class Csiq(torch.utils.data.Dataset):
    def __init__(self, dis_path, txt_file_name, list_name, transform, keep_ratio):
        super(Csiq, self).__init__()

        self.dis_path = dis_path
        self.txt_file_name = txt_file_name
        self.transform = transform

        self.list_name = list_name

        # 读取创建的 csiq_label.txt 文件
        data = pd.read_csv(self.txt_file_name)

        self.data_dict = []
        for i, row in data.iterrows():
            # 从 'CSIQ/src_imgs/1600.png' 中提取出 '1600.png'
            ref_name = os.path.basename(row['ref_img_path'])

            # 检查这张图片的参考图是否在我们要加载的列表 (list_name) 中
            if ref_name in self.list_name:


                relative_path_parts = row['dis_img_path'].split('dst_imgs/')
                if len(relative_path_parts) > 1:
                    relative_path = relative_path_parts[1]
                else:
                    relative_path = row['dis_img_path']
                # 拼接成最终的完整 Windows 路径
                full_dis_path = os.path.join(self.dis_path, relative_path)

                self.data_dict.append({
                    'd_img_path': full_dis_path,
                    'score': float(row['score'])
                })

        if keep_ratio < 1.0:
            n_kept = int(len(self.data_dict) * keep_ratio)
            self.data_dict = [self.data_dict[i] for i in np.random.permutation(len(self.data_dict))[:n_kept]]

    def normalization(self, data):
        # 归一化分数到 0-1 范围
        range_val = np.max(data) - np.min(data)
        if range_val == 0:
            return data
        return (data - np.min(data)) / range_val

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        d_img_path = self.data_dict[idx]['d_img_path']
        try:
            d_img = cv2.imread(d_img_path, cv2.IMREAD_COLOR)
            d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"!!!!!!!!!!!!\n图像加载失败: {d_img_path}\n错误: {e}\n!!!!!!!!!!!!")

            d_img = np.zeros((224, 224, 3), dtype=np.uint8)

        d_img = np.array(d_img).astype('float32') / 255
        d_img = np.transpose(d_img, (2, 0, 1))

        score = self.data_dict[idx]['score']
        score = np.array(score).astype('float32')

        sample = {'d_img_org': d_img, 'score': score}
        if self.transform:
            sample = self.transform(sample)

        return sample