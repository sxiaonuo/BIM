import torch
from pyexpat import features
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
import math
import torch
from pyexpat import features
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
import math


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        # ViTDet不需要cls token
        # 位置编码信息，只有(img_size // patch_size)**2个位置向量(无cls token)
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        x += self.positions
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class WindowedMultiHeadAttention(nn.Module):
    """ViTDet使用的窗口注意力机制，配合跨窗口传播"""

    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0,
                 window_size: int = 14, use_global: bool = False):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.window_size = window_size
        self.use_global = use_global  # 是否为全局传播块

        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def window_partition(self, x: Tensor) -> Tensor:
        """将特征图划分为非重叠窗口
        输入形状: (B, H, W, C)
        输出形状: (num_windows*B, window_size*window_size, C)
        """
        B, H, W, C = x.shape
        # 确保H和W能被window_size整除
        assert H % self.window_size == 0 and W % self.window_size == 0, \
            f"特征图大小({H},{W})必须能被窗口大小({self.window_size})整除"

        x = x.view(B, H // self.window_size, self.window_size,
                   W // self.window_size, self.window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, self.window_size * self.window_size, C)
        return windows

    def window_reverse(self, windows: Tensor, H: int, W: int) -> Tensor:
        """将窗口重组为完整特征图
        输入形状: (num_windows*B, window_size*window_size, C)
        输出形状: (B, H, W, C)
        """
        B = int(windows.shape[0] / (H * W / self.window_size / self.window_size))
        x = windows.view(B, H // self.window_size, W // self.window_size,
                         self.window_size, self.window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H, W, -1)
        return x

    def forward(self, x: Tensor) -> Tensor:
        if not self.use_global:
            # 窗口注意力模式
            B, N, C = x.shape
            H = W = int(math.sqrt(N))

            # 首先将序列转换为2D特征图
            x = x.view(B, H, W, C)

            # 保存原始输入用于残差连接
            shortcut = x

            # 划分窗口
            x_windows = self.window_partition(x)

            # 计算注意力
            qkv = rearrange(self.qkv(x_windows), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
            queries, keys, values = qkv[0], qkv[1], qkv[2]

            energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
            scaling = self.emb_size ** 0.5
            att = F.softmax(energy, dim=-1) / scaling
            att = self.att_drop(att)

            out = torch.einsum('bhal, bhlv -> bhav ', att, values)
            out = rearrange(out, "b h n d -> b n (h d)")
            out = self.projection(out)

            # 重组窗口
            out = self.window_reverse(out, H, W)

            # 将特征图转换回序列形式
            out = out.view(B, N, C)

            # 残差连接
            out += shortcut.view(B, N, C)
            return out
        else:
            # 全局注意力模式(用于跨窗口传播)
            B, N, C = x.shape

            # 计算全局注意力
            qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
            queries, keys, values = qkv[0], qkv[1], qkv[2]

            energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
            scaling = self.emb_size ** 0.5
            att = F.softmax(energy, dim=-1) / scaling
            att = self.att_drop(att)

            out = torch.einsum('bhal, bhlv -> bhav ', att, values)
            out = rearrange(out, "b h n d -> b n (h d)")
            out = self.projection(out)

            return out

class ConvPropagationBlock(nn.Module):
    """ViTDet的卷积传播块替代方案"""

    def __init__(self, emb_size: int = 768):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(emb_size, emb_size // 4, kernel_size=1),
            nn.LayerNorm(emb_size // 4),
            nn.GELU(),
            nn.Conv2d(emb_size // 4, emb_size // 4, kernel_size=3, padding=1),
            nn.LayerNorm(emb_size // 4),
            nn.GELU(),
            nn.Conv2d(emb_size // 4, emb_size, kernel_size=1)
        )
        # 最后一层初始化为0
        nn.init.zeros_(self.conv_block[-1].weight)
        nn.init.zeros_(self.conv_block[-1].bias)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)  # B C H W
        out = self.conv_block(x)
        out = out.permute(0, 2, 3, 1).view(B, N, C)
        return out

class SimpleFeaturePyramid(nn.Module):
    """ViTDet的简单特征金字塔实现"""

    def __init__(self, emb_size: int = 768, out_channels: int = 256, img_size: int = 1024, patch_size: int = 16):
        super().__init__()
        self.strides = [4, 8, 16, 32]  # 多尺度特征对应的步长
        self.scale_factors = [1 / 4, 1 / 2, 1, 2]  # 相对于1/16特征的缩放因子

        # 计算基础特征图尺寸 (ViT输出的特征图大小)
        self.base_feat_size = img_size // patch_size  # 例如1024/16=64
        self.H = self.W = self.base_feat_size

        self.stages = nn.ModuleList()
        for scale in self.scale_factors:
            layers = []
            if scale == 2.0:
                # 1/32尺度: stride=2的max pooling
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                out_dim = emb_size
                h = w = self.H // 2
            elif scale == 1.0:
                # 1/16尺度: 无操作
                out_dim = emb_size
                h, w = self.H, self.W
            elif scale == 0.5:
                # 1/8尺度: stride=2的反卷积
                layers.append(nn.ConvTranspose2d(emb_size, emb_size // 2, kernel_size=2, stride=2))
                out_dim = emb_size // 2
                h = w = self.H * 2
            elif scale == 0.25:
                # 1/4尺度: 两个stride=2的反卷积
                layers.extend([
                    nn.ConvTranspose2d(emb_size, emb_size // 2, kernel_size=2, stride=2),
                    # 第一个反卷积后的尺寸
                    Rearrange('b c h w -> b (h w) c'),
                    nn.LayerNorm(emb_size // 2),
                    Rearrange('b (h w) c -> b c h w', h=self.H * 2, w=self.W * 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(emb_size // 2, emb_size // 4, kernel_size=2, stride=2)
                ])
                out_dim = emb_size // 4
                h = w = self.H * 4

            # 添加1x1和3x3卷积
            layers.extend([
                nn.Conv2d(out_dim, out_channels, kernel_size=1),
                Rearrange('b c h w -> b (h w) c'),
                nn.LayerNorm(out_channels),
                Rearrange('b (h w) c -> b c h w', h=h, w=w),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                Rearrange('b c h w -> b (h w) c'),
                nn.LayerNorm(out_channels),
                Rearrange('b (h w) c -> b c h w', h=h, w=w)
            ])

            self.stages.append(nn.Sequential(*layers))

    def forward(self, x: Tensor) -> dict:
        """输入为ViT最后一层特征图(B, N, C), 输出多尺度特征字典"""
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)  # B C H W

        features = {}
        for i, (stride, stage) in enumerate(zip(self.strides, self.stages)):
            # 通过stage处理
            # print(stride, x.shape)
            stage_output = stage(x)

            # 懒得确保是四维度了，正常情况下就应该是四维度，直接断言吧
            assert len(stage_output.shape) == 4

            features[f'p{stride}'] = stage_output

        return features

class TransformerEncoderBlock(nn.Module):
    """改造后的Transformer编码块，支持窗口注意力和跨窗口传播"""

    def __init__(self,
                 emb_size: int = 768,
                 num_heads: int = 8,
                 drop_p: float = 0.,
                 window_size: int = 14,
                 is_global_block: bool = False,
                 use_conv_prop: bool = False):
        super().__init__()
        self.is_global_block = is_global_block
        self.use_conv_prop = use_conv_prop

        # 窗口注意力或全局注意力
        self.attention = WindowedMultiHeadAttention(
            emb_size, num_heads, drop_p, window_size, use_global=is_global_block)

        # 跨窗口传播块(卷积或全局注意力)
        if is_global_block and use_conv_prop:
            self.propagation = ConvPropagationBlock(emb_size)
        elif is_global_block:
            self.propagation = WindowedMultiHeadAttention(
                emb_size, num_heads, drop_p, window_size, use_global=True)

        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.ffn = FeedForwardBlock(emb_size)
        self.dropout = nn.Dropout(drop_p)

    def forward(self, x: Tensor) -> Tensor:
        # 注意力部分
        x = self.norm1(x)
        attn_out = self.attention(x)
        x = x + self.dropout(attn_out)

        # 如果是全局传播块
        if self.is_global_block:
            x = self.norm2(x)
            if self.use_conv_prop:
                prop_out = self.propagation(x)
            else:
                prop_out = self.propagation(x)
            x = x + self.dropout(prop_out)

        # FFN部分
        x = self.norm2(x) if not self.is_global_block else x
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)

        return x


class TransformerEncoder(nn.Module):
    """改造后的Transformer编码器，支持跨窗口传播策略"""

    def __init__(self,
                 depth: int = 12,
                 emb_size: int = 768,
                 num_heads: int = 8,
                 window_size: int = 14,
                 drop_p: float = 0.,
                 use_conv_prop: bool = False):
        super().__init__()
        self.layers = nn.ModuleList()

        # 将block分为4个部分，每个部分的最后一个block是全局传播块
        blocks_per_group = depth // 4
        for i in range(depth):
            is_global = (i + 1) % blocks_per_group == 0  # 每个部分的最后一个block
            self.layers.append(
                TransformerEncoderBlock(
                    emb_size, num_heads, drop_p, window_size,
                    is_global, use_conv_prop))

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x


class PredictHead(nn.Module):
    def __init__(self, in_channels=256, num_prototypes=32):
        super().__init__()
        self.num_prototypes = num_prototypes
        # 共享特征提取（移除固定H,W依赖）
        self.base = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True)
        )

        self.scales = [nn.Parameter(torch.ones(1)) for _ in range(4)]  # 可学习参数，用于预测尺度，便于还原坐标原始位置

        # 预测分支
        self.bbox_reg = nn.Conv2d(in_channels, 4, 3, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, 3, padding=1)
        self.mask_conv = nn.Conv2d(in_channels, num_prototypes, 3, padding=1)

    def _create_grid(self, H, W, device):
        """为每个样本生成独立归一化网格"""
        y, x = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'  # 确保H在前，W在后
        )
        # 归一化到0-1范围（基于特征图尺寸）
        grid_x = (x + 0.5) / W  # [H,W] 特征图位置 → 原图比例
        grid_y = (y + 0.5) / H

        # 扩展为 [B, H, W, 2]
        grid = torch.stack([grid_x, grid_y], dim=-1)  # [H,W,2]
        grid = grid.unsqueeze(0)  # 添加批次维度 [1,H,W,2]
        return grid

    def forward(self, x, level_id):
        """
        前向传播
        :param x:
        :param level_id:
        :return: {
            'boxes': boxes,  # 边界框坐标 [B,H*W,6]
            'centerness': centerness,  # 中心度 [B,H*W]
            'mask': mask,  # 掩码 [B,H*W,N]
        }
        """
        B, _, H, W = x.shape

        # 特征提取
        base_feat = self.base(x)  # [B,64,H,W]

        # 预测边界框参数
        bbox_pred = torch.sigmoid(self.bbox_reg(base_feat) * self.scales[level_id].to(x.device))  # [B,4,H,W]
        centerness = torch.sigmoid(self.centerness(base_feat))  # [B,1,H,W]

        # 生成网格
        grid = self._create_grid(H, W, x.device)  # [1,H,W,2]
        grid = grid.expand(B, -1, -1, -1)  # 复制到每个样本 [B,H,W,2]

        # 分解坐标
        cx = grid[..., 0]  # [B,H,W]
        cy = grid[..., 1]  # [B,H,W]

        # 分解预测值
        l = bbox_pred[:, 0, :, :]  # [B,H,W]
        t = bbox_pred[:, 1, :, :]
        r = bbox_pred[:, 2, :, :]
        b = bbox_pred[:, 3, :, :]

        # 计算边界框坐标（保持维度一致）
        boxes = torch.stack([
            cx - l,  # x_min
            cy - t,  # y_min
            cx + r,  # x_max
            cy + b,  # y_max
            cx*512,  # 中心点x坐标
            cy*512,  # 中心点y坐标
        ], dim=-1)  # [B,H,W,6], 6个坐标: x_min,y_min,x_max,y_max,cx,cy

        # 调整维度H，W -> H*W
        boxes = boxes.view(B, -1, 6)  # [B,H*W,6]
        centerness = centerness.view(B, -1)
        mask = self.mask_conv(base_feat)  # [B,32,H,W]
        mask = mask.permute(0, 2, 3, 1).view(B, -1, self.num_prototypes)  # [B,H*W,N]
        mask = mask.sigmoid()

        return {
            'boxes': boxes,  # 边界框坐标 [B,H*W,6]
            'centerness': centerness,  # 中心度 [B,H*W]
            'mask': mask,  # 掩码 [B,H*W,N]
        }


class ViTDet(nn.Module):
    """完整的ViTDet模型实现"""

    def __init__(self,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 emb_size: int = 768,
                 img_size: int = 1024,
                 depth: int = 12,
                 num_heads: int = 8,
                 window_size: int = 14,
                 num_prototypes: int = 32,
                 use_conv_prop: bool = False,
                 ):
        super().__init__()

        self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        self.encoder = TransformerEncoder(
            depth, emb_size, num_heads, window_size, use_conv_prop=use_conv_prop)
        self.fpn = SimpleFeaturePyramid(emb_size, img_size=img_size, patch_size=patch_size)

        # 检测头(Mask R-CNN风格)
        self.rpn = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # 预测头，4个预测头对应P2-P5
        self.predict_head = PredictHead(in_channels=256, num_prototypes=num_prototypes)

    def forward(self, x: Tensor) -> dict:
        # 1. 提取patch嵌入
        x = self.patch_embed(x)  # B, N, C

        # 2. Transformer编码
        x = self.encoder(x)  # B, N, C

        # 3. 构建简单特征金字塔
        features = self.fpn(x)  # 多尺度特征字典

        # 4. 检测头处理
        p4 = features['p4']  # torch.Size([B, 256, 56, 56])
        p8 = features['p8']  # torch.Size([B, 256, 28, 28])
        p16= features['p16'] # torch.Size([B, 256, 14, 14])
        p32= features['p32'] # torch.Size([B, 256, 7, 7])
        print(p4.shape, p8.shape, p16.shape, p32.shape)

        # RPN处理
        # rpn_outs = {k: self.rpn(v) for k, v in features.items()}
        # # 对RPN输出进行下采样到固定尺寸(56x56)
        # rpn_outs = {k: F.interpolate(v, size=(56, 56), mode='bilinear', align_corners=False) for k, v in rpn_outs.items()}
        # rpn_outs = torch.cat([v for v in rpn_outs.values()], dim=1)
        # print("rpn_outs", rpn_outs.shape)

        pred_head_outputs = []

        # 多尺度预测
        for i, p in enumerate([p4, p8, p16, p32]):
            level_pred = self.predict_head(p, level_id=i)
            pred_head_outputs.append(level_pred)

        pred_head_outputs = {
            'boxes': torch.cat([o['boxes'] for o in pred_head_outputs], dim=1),  # [B, num_priors, 6]
            'centerness': torch.cat([o['centerness'] for o in pred_head_outputs], dim=1),  # [B, num_priors]
            'mask': torch.cat([o['mask'] for o in pred_head_outputs], dim=1),  # [B, num_priors, num_prototypes]
        }

        return pred_head_outputs


if __name__ == '__main__':
    # 测试ViTDet
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(2, 3, 224, 224).to(device)
    model = ViTDet(img_size=224, patch_size=16, depth=8, window_size=14).to(device)
    outputs = model(x)
    boxes, centerness, mask = outputs['boxes'], outputs['centerness'], outputs['mask']
    print(boxes.shape, centerness.shape, mask.shape) # 56 * 56 + 28 * 28 + 14 * 14 + 7 * 7 = 4165
