import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from torch import nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


# 基础的 FFN 模块
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# 划分窗口
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


# 还原窗口
def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # 相对位置编码表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        # 留个变量存 attention map，方便画图
        self.latest_A_map_avg = None

    def forward(self, x, mask=None, hpsm_bias=None, f_low_win=None, f_top_win=None, tau=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        # 加 HPSM 偏置
        if hpsm_bias is not None:
            attn = attn + hpsm_bias

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            A_map = self.softmax(attn)
        else:
            A_map = self.softmax(attn)

        # 多头取个平均，存下来给可视化用
        A_map_avg_head = torch.mean(A_map, dim=1)
        self.latest_A_map_avg = A_map_avg_head.detach().cpu()

        # 算 SCSA 的交叉层蒸馏损失
        l_sem = 0.0
        if f_low_win is not None and f_top_win is not None:
            f_low_norm = F.normalize(f_low_win, p=2, dim=2)
            f_top_norm = F.normalize(f_top_win, p=2, dim=2)

            # 算 cosine similarity 矩阵
            R_win = torch.bmm(f_low_norm, f_top_norm.transpose(1, 2))
            R_win = F.softmax(R_win / tau, dim=-1)

            # MSE 损失
            l_sem = F.mse_loss(A_map_avg_head, R_win.detach())

        attn = self.attn_drop(A_map)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, l_sem

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        flops = 0
        flops += N * self.dim * 3 * self.dim
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        flops += N * self.dim * self.dim
        return flops


class SwinBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 dim_mlp=1024., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, vit_dim=768):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.dim_mlp = dim_mlp
        self.vit_dim = vit_dim

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = self.dim_mlp
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x, S_map=None, hpsm_beta=None, f_low=None, f_top=None, tau=None):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Swin 的特征 shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # 处理 HPSM 偏置
        hpsm_bias = None
        if S_map is not None:
            S_map_whc = S_map.permute(0, 2, 3, 1)
            if self.shift_size > 0:
                shifted_S_map = torch.roll(S_map_whc, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            else:
                shifted_S_map = S_map_whc

            S_map_windows = window_partition(shifted_S_map, self.window_size)
            S_map_windows_flat = S_map_windows.view(-1, self.window_size * self.window_size)
            S_i = S_map_windows_flat.unsqueeze(2)
            S_j = S_map_windows_flat.unsqueeze(1)
            bias = (S_i + S_j) / 2.0
            hpsm_bias = bias.unsqueeze(1) * hpsm_beta

        # 处理 SCSA 特征的对齐，同步 shift 操作
        f_low_win, f_top_win = None, None
        if f_low is not None and f_top is not None:
            f_low_2d = f_low.view(B, H, W, self.vit_dim)
            f_top_2d = f_top.view(B, H, W, self.vit_dim)

            if self.shift_size > 0:
                shifted_f_low = torch.roll(f_low_2d, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                shifted_f_top = torch.roll(f_top_2d, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            else:
                shifted_f_low = f_low_2d
                shifted_f_top = f_top_2d

            f_low_win_tmp = window_partition(shifted_f_low, self.window_size)
            f_top_win_tmp = window_partition(shifted_f_top, self.window_size)

            f_low_win = f_low_win_tmp.view(-1, self.window_size * self.window_size, self.vit_dim)
            f_top_win = f_top_win_tmp.view(-1, self.window_size * self.window_size, self.vit_dim)

            # 计算 Attention 和 语义损失
        attn_windows, l_sem = self.attn(
            x_windows,
            mask=self.attn_mask,
            hpsm_bias=hpsm_bias,
            f_low_win=f_low_win,
            f_top_win=f_top_win,
            tau=tau
        )

        # 还原窗口
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, l_sem

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        flops += self.dim * H * W
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        flops += self.dim * H * W
        return flops


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size=7,
                 dim_mlp=1024, qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 vit_dim=768):
        super().__init__()
        self.dim = dim
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinBlock(dim=dim, input_resolution=input_resolution,
                      num_heads=num_heads, window_size=window_size,
                      shift_size=0 if (i % 2 == 0) else window_size // 2,
                      dim_mlp=dim_mlp,
                      qkv_bias=qkv_bias, qk_scale=qk_scale,
                      drop=drop, attn_drop=attn_drop,
                      drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                      norm_layer=norm_layer,
                      vit_dim=vit_dim)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, S_map=None, hpsm_beta=None, f_low=None, f_top=None, tau=None):
        l_sem_total = 0.0
        for blk in self.blocks:
            x, l_sem = blk(x, S_map, hpsm_beta, f_low, f_top, tau)
            l_sem_total += l_sem  # 累加每一层的语义损失

        x = rearrange(x, 'b (h w) c -> b c h w', h=self.input_resolution[0], w=self.input_resolution[1])
        x = F.relu(self.conv(x))
        x = rearrange(x, 'b c h w -> b (h w) c')

        return x, l_sem_total

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class SwinTransformer(nn.Module):
    def __init__(self, patches_resolution, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 embed_dim=256, drop=0.1, drop_rate=0., drop_path_rate=0.1, dropout=0., window_size=7,
                 dim_mlp=1024, qkv_bias=True, qk_scale=None, attn_drop_rate=0., norm_layer=nn.LayerNorm,
                 downsample=None, use_checkpoint=False, scale=0.8, **kwargs):
        super().__init__()
        self.scale = scale
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.dropout = nn.Dropout(p=drop)
        self.num_features = embed_dim
        self.num_layers = len(depths)
        self.patches_resolution = (patches_resolution[0], patches_resolution[1])
        self.downsample = nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=3, stride=2, padding=1)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # SCSA里算相似度需要知道 ViT 的维度，这里传进去
        self.vit_dim = dim_mlp

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=self.embed_dim,
                input_resolution=patches_resolution,
                depth=self.depths[i_layer],
                num_heads=self.num_heads[i_layer],
                window_size=self.window_size,
                dim_mlp=dim_mlp,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=dropout,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=downsample,
                use_checkpoint=use_checkpoint,
                vit_dim=self.vit_dim
            )
            self.layers.append(layer)

    def forward(self, x, S_map=None, hpsm_beta=None, f_low=None, f_top=None, tau=None):
        x = self.dropout(x)
        x = rearrange(x, 'b c h w -> b (h w) c')

        l_sem_total = 0.0
        for layer in self.layers:
            _x = x
            x, l_sem = layer(x, S_map, hpsm_beta, f_low, f_top, tau)
            l_sem_total += l_sem

            x = self.scale * x + _x

        x = rearrange(x, 'b (h w) c -> b c h w', h=self.patches_resolution[0], w=self.patches_resolution[1])

        return x, l_sem_total