import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from timm.models.vision_transformer import Block
from models.swin import SwinTransformer
from torch import nn
from einops import rearrange

# 导入 HPSM 的工具函数
from utils.hpsm_utils import get_masking_factor_M, get_gradient_response_G, get_sensitivity_map_S


class HookLayer(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.output = None

    def forward(self, x):
        self.output = x
        return x


class TABlock(nn.Module):
    def __init__(self, dim, drop=0.1):
        super().__init__()
        self.c_q = nn.Linear(dim, dim)
        self.c_k = nn.Linear(dim, dim)
        self.c_v = nn.Linear(dim, dim)
        self.norm_fact = dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x):
        _x = x
        B, C, N = x.shape
        q = self.c_q(x)
        k = self.c_k(x)
        v = self.c_v(x)

        attn = q @ k.transpose(-2, -1) * self.norm_fact
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, C, N)
        x = self.proj_drop(x)
        x = x + _x
        return x


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []


class MANIQA(nn.Module):
    
    def __init__(self, embed_dim=768, num_outputs=1, patch_size=8, drop=0.1,
                 depths=[2, 2], window_size=4, dim_mlp=768, num_heads=[4, 4],
                 img_size=224, num_tab=2, scale=0.8, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.input_size = img_size // patch_size
        self.patches_resolution = (img_size // patch_size, img_size // patch_size)

        self.vit = timm.create_model('vit_base_patch8_224', pretrained=True)
        self.save_output = SaveOutput()
        hook_handles = []
        for layer in self.vit.modules():
            if isinstance(layer, Block):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)

        # 模块 A (HPSM) 配置
        self.hpsm_rms_r = 7
        self.hpsm_dog_sigmas = [1, 2, 4]
        self.hpsm_lambda = 1.0

        self.hpsm_g_layer = 3  # 用第3层算梯度响应
        self.hpsm_beta = nn.Parameter(torch.zeros(1))

        # 模块 B (SCSA) 配置
        self.scsa_f_low_layer = 3   # 浅层纹理
        self.scsa_f_top_layer = 10  # 深层语义
        self.scsa_tau = 0.07        
        self.scsa_lambda = 0.05     # 蒸馏损失权重

        # 模块 C2 (SRA) 配置
        self.c2_e_d_layers = [2, 5]  # 取第2、5层特征算失真因子
        self.c2_z_dim = 32           
        self.sigma = 0.1             # 论文 Eq.9 的缩放因子

        vit_c_dim = self.vit.embed_dim  # 768
        swin_c_dim = embed_dim // 2     # 384

        # 失真编码器 E_d
        self.c2_e_d_head = nn.Sequential(
            nn.Linear(vit_c_dim * len(self.c2_e_d_layers), 128),
            nn.ReLU(),
            nn.Linear(128, self.c2_z_dim),
            nn.BatchNorm1d(self.c2_z_dim)  
        )

        # SRA 残差回归头
        adapter_input_dim = self.c2_z_dim + swin_c_dim + 1
        self.c2_sr_adapter = nn.Sequential(
            nn.Linear(adapter_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        
        self.tablock1 = nn.ModuleList()
        for i in range(num_tab):
            tab = TABlock(self.input_size ** 2)
            self.tablock1.append(tab)

        self.conv1 = nn.Conv2d(embed_dim * 4, embed_dim, 1, 1, 0)
        self.swintransformer1 = SwinTransformer(
            patches_resolution=self.patches_resolution,
            depths=depths,
            num_heads=num_heads,
            embed_dim=embed_dim,
            window_size=window_size,
            dim_mlp=dim_mlp,
            scale=scale
        )

        self.tablock2 = nn.ModuleList()
        for i in range(num_tab):
            tab = TABlock(self.input_size ** 2)
            self.tablock2.append(tab)

        self.conv2 = nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0)
        self.swintransformer2 = SwinTransformer(
            patches_resolution=self.patches_resolution,
            depths=depths,
            num_heads=num_heads,
            embed_dim=embed_dim // 2,
            window_size=window_size,
            dim_mlp=dim_mlp,
            scale=scale
        )

        self.fc_score_core = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim // 2, num_outputs),
            nn.ReLU()
        )
        self.fc_weight_core = nn.Sequential(
            nn.Linear(embed_dim // 2, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop),
            nn.Linear(embed_dim // 2, num_outputs),
            nn.Sigmoid()
        )

        
        self.fc_score = nn.Sequential(self.fc_score_core, HookLayer())
        self.fc_weight = nn.Sequential(self.fc_weight_core, HookLayer())

    def extract_feature(self, save_output):
        x6 = save_output.outputs[6][:, 1:]
        x7 = save_output.outputs[7][:, 1:]
        x8 = save_output.outputs[8][:, 1:]
        x9 = save_output.outputs[9][:, 1:]
        x = torch.cat((x6, x7, x8, x9), dim=2)
        return x

    def forward(self, x, return_maps=False, return_z_d=False, return_delta_score=False):
        # 归一化输入
        x_norm = (x * 0.5) + 0.5

        # 第一阶段：算 HPSM
        M_map = get_masking_factor_M(x_norm, self.hpsm_rms_r, self.hpsm_dog_sigmas)
        M_map_token = F.avg_pool2d(M_map, kernel_size=self.patch_size)

        _x = self.vit(x) 

        f_g_layer = self.save_output.outputs[self.hpsm_g_layer][:, 1:]
        G_map = get_gradient_response_G(f_g_layer, self.input_size)
        S_map = get_sensitivity_map_S(G_map, M_map_token, self.hpsm_lambda)

        self.last_S_hpsm = S_map  

        # 第一阶段：算 SCSA
        f_low = self.save_output.outputs[self.scsa_f_low_layer][:, 1:]
        f_top = self.save_output.outputs[self.scsa_f_top_layer][:, 1:]

        # 第二阶段：提取 SRA 失真因子
        f_e_d_layers = [self.save_output.outputs[i][:, 1:] for i in self.c2_e_d_layers]
        f_e_d_pooled = [torch.mean(f, dim=1) for f in f_e_d_layers]
        f_e_d_cat = torch.cat(f_e_d_pooled, dim=1)

        # 确保 BN 层只在训练时更新统计信息
        if self.training:
            z_d = self.c2_e_d_head(f_e_d_cat)
        else:
            self.c2_e_d_head.eval()  
            z_d = self.c2_e_d_head(f_e_d_cat)
            self.c2_e_d_head.train()  

        # 跑主干特征提取
        x_feat = self.extract_feature(self.save_output)
        self.save_output.outputs.clear()

        x = rearrange(x_feat, 'b (h w) c -> b c (h w)', h=self.input_size, w=self.input_size)
        for tab in self.tablock1:
            x = tab(x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        x = self.conv1(x)

        # 解耦操作1：swin1 只传 SCSA 参数，禁用 HPSM
        x, l_sem_1 = self.swintransformer1(
            x, None, None,               
            f_low, f_top, self.scsa_tau  
        )

        x = rearrange(x, 'b c h w -> b c (h w)', h=self.input_size, w=self.input_size)
        for tab in self.tablock2:
            x = tab(x)
        x = rearrange(x, 'b c (h w) -> b c h w', h=self.input_size, w=self.input_size)
        x = self.conv2(x)

        # 解耦操作2：swin2 只传 HPSM 参数，禁用 SCSA
        x, l_sem_2 = self.swintransformer2(
            x, S_map, self.hpsm_beta,  
            None, None, None           
        )

        x = rearrange(x, 'b c h w -> b (h w) c', h=self.input_size, w=self.input_size)

        # 预测基础分 (Phase I)
        f = self.fc_score(x)   
        w = self.fc_weight(x)  

        baseline_score = torch.sum(f * w, dim=1) / (torch.sum(w, dim=1) + 1e-6)  
        baseline_score_squeezed = torch.squeeze(baseline_score)                  

        # 计算 SRA 残差 (Phase II)
        global_feat = torch.mean(x, dim=1)  

        # 注意：这里 baseline_score 必须 detach，坚决不让梯度往回传破坏骨干
        adapter_input = torch.cat(
            [z_d, global_feat, baseline_score.detach()],
            dim=1
        )

        delta_score = self.c2_sr_adapter(adapter_input)  

        # 结合公式算最终分
        final_score = baseline_score_squeezed + self.sigma * torch.squeeze(delta_score)

        l_sem_total = (l_sem_1 + l_sem_2) * self.scsa_lambda

        # 设置不同返回参数应对不同的脚本
        if return_maps:
            return final_score, l_sem_total, f, w
        elif return_z_d:
            return final_score, l_sem_total, z_d
        elif return_delta_score:
            return final_score, l_sem_total, delta_score
        else:
            return final_score, l_sem_total