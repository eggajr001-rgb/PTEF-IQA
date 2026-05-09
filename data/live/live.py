import os
import torch
import numpy as np
import cv2


class Live(torch.utils.data.Dataset):
    def __init__(self, dis_path, txt_file_name, list_name, transform, keep_ratio):
        super(Live, self).__init__()
        self.dis_path = dis_path
        self.txt_file_name = txt_file_name
        self.transform = transform

        dis_files_data, score_data = [], []
        with open(self.txt_file_name, 'r') as listFile:

            next(listFile)
            for line in listFile:

                try:
                    dis_path_full, dis_type, ref_path, score = line.strip().split(',')
                except ValueError:
                    continue


                ref_name = os.path.basename(ref_path)

                if ref_name in list_name:
                    score = float(score)

                    dis_type_folder = dis_path_full.split('/')[1]  # 'jp2k'
                    dis_img_name = os.path.basename(dis_path_full)  # 'img2.bmp'

                    dis_files_data.append((dis_type_folder, dis_img_name))
                    score_data.append(score)


        score_data = np.array(score_data)
        score_data = self.normalization(score_data)
        score_data = score_data.astype('float').reshape(-1, 1)
        self.data_dict = {'d_img_list': dis_files_data, 'score_list': score_data}

    def normalization(self, data):
        range = np.max(data) - np.min(data)
        return (data - np.min(data)) / range

    def __len__(self):
        return len(self.data_dict['d_img_list'])

    def __getitem__(self, idx):
        dis_type_folder, d_img_name = self.data_dict['d_img_list'][idx]


        img_path = os.path.join(self.dis_path, dis_type_folder, d_img_name)

        d_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype('float32') / 255
        d_img = np.transpose(d_img, (2, 0, 1))

        score = self.data_dict['score_list'][idx]
        sample = {
            'd_img_org': d_img,
            'score': score
        }
        if self.transform:
            sample = self.transform(sample)
        return sample