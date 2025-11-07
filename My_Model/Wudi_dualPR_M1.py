import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp  # 添加Mlp导入

class OverlapPatchEmbed(nn.Module):
    """重叠块嵌入层：使用带重叠的卷积操作将图像分割成块"""

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)  # 转换为（高，宽）元组
        patch_size = to_2tuple(patch_size)

        # 计算输出特征图的尺寸
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W

        # 使用卷积实现块嵌入
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2)  # 添加重叠
        )
        self.norm = nn.LayerNorm(embed_dim)  # 标准化层

    def forward(self, x):
        """
        输入形状：(B, C, H, W)
        输出形状：(B, num_patches, embed_dim)
        """
        # 步骤1：通过卷积提取块特征 [形状变化示例：假设输入是(2,3,512,512)]
        x = self.proj(x)  # 输出形状：(B, embed_dim, H_new, W_new)
        # 例如：stride=4时，(2,3,512,512) → (2,64,128,128)

        # 步骤2：展平空间维度
        _, _, H, W = x.shape
        x = x.flatten(2)  # 展平H和W维度 → (B, embed_dim, H*W)
        # (2,64,128,128) → (2,64,16384)

        # 步骤3：转置维度顺序
        x = x.transpose(1, 2)  # → (B, H*W, embed_dim)
        # (2,64,16384) → (2,16384,64)

        # 步骤4：层标准化
        x = self.norm(x)
        return x, H, W  # 返回处理后的特征和原始空间尺寸

class MixFFN(nn.Module):
    def __init__(self, dim, hidden_dim=None, drop=0.):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.fc1(x)
        x = self.drop(x)
        x = x.transpose(1, 2).view(B, -1, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class MixFFNWithCls(nn.Module):
    def __init__(self, dim, hidden_dim=None, drop=0.):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        cls_token, img_tokens = x[:, :1, :], x[:, 1:, :]
        B, N, C = img_tokens.shape

        x = self.fc1(img_tokens)
        x = self.drop(x)
        x = x.transpose(1, 2).view(B, -1, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)

        return torch.cat([cls_token, x], dim=1)
#pointrend
class PointRefineHead(nn.Module):
    def __init__(self, in_channels, hidden_dim=64):
        super().__init__()
        #过程取点
        # self.mlp = nn.Sequential(
        #     nn.Linear(in_channels, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, 1)  # 输出点预测值
        # )
        # 结果取点
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 1)  # 输出点预测值
        )

    def forward(self, point_feats):  # shape: (B, N, C)
        return self.mlp(point_feats).squeeze(-1)  # shape: (B, N)


class LaplaceConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
        super(LaplaceConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)

        # Generate Laplace kernel
        laplace_kernel = torch.tensor([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=torch.float32)  ##8领域
        laplace_kernel = laplace_kernel.unsqueeze(0).unsqueeze(0)
        laplace_kernel = laplace_kernel.repeat((out_channels, in_channels, 1, 1))
        self.conv.weight = nn.Parameter(laplace_kernel)
        self.conv.bias.data.fill_(0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv(x)
        x1 = self.relu(self.bn(x1))

        return x1

class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(CBAMBlock, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.channel_attention(x)
        x = x * avg_out

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_attention_map = self.spatial_attention(torch.cat([max_out, avg_out], dim=1))
        x = x * spatial_attention_map
        return x

class CrossModalCLSToken(nn.Module):
    """
    多模态CLS Token生成模块
    输入形状：(B, C, H, W) → 示例：(2,3,512,512)
    输出形状：(B, 1, embed_dim) → 示例：(2,1,64)
    """

    def __init__(self, in_channels=1, embed_dim=64):
        super().__init__()

        # 第一阶段特征提取
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=7, stride=2, padding=3),  # (H/2, W/2)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # (H/4, W/4)
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # 全局特征聚合
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # (1,1)

        # CLS Token生成
        self.projection = nn.Linear(32, embed_dim)

        # # 可学习的CLS Token（可选）
        # self.learnable_cls = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x):
        """
        前向传播流程：
        输入 → (2,3,512,512)
        ↓ 特征提取
        → (2,16,256,256) → (2,32,128,128)
        ↓ 全局池化
        → (2,32,1,1) → (2,32)
        ↓ 线性投影
        → (2,64) → (2,1,64)
        """
        # 特征提取
        x = self.feature_extractor(x)  # → (2,32,128,128)

        # 全局平均池化
        x = self.global_pool(x)  # → (2,32,1,1)
        x = x.flatten(1)  # → (2,32)

        # 投影到嵌入空间
        cls_token = self.projection(x)  # → (2,64)
        cls_token = cls_token.unsqueeze(1)  # → (2,1,64)

        # 与可学习Token融合（可选）
        # if self.learnable_cls is not None:
        #     learned_cls = self.learnable_cls.expand(x.size(0), -1, -1)  # → (2,1,64)
        #     cls_token = cls_token + learned_cls

        return cls_token


class Make_Transformer_Input(nn.Module):
    """
    改进的块嵌入层（支持CLS Token注入）
    输入形状：
    - 图像输入：(B, C, H, W)
    - CLS Token输入：(B, 1, E)
    输出形状：(B, num_patches + 1, E)
    """

    def __init__(self, img_size=224, patch_size=7, stride=4,
                 in_chans=3, embed_dim=768):
        super().__init__()
        # 原始块嵌入层
        self.patch_embed = OverlapPatchEmbed(img_size, patch_size, stride,
                                             in_chans, embed_dim)



        # # CLS Token位置编码
        # self.cls_pos_embed = nn.Parameter(
        #     torch.randn(1, 1, embed_dim) * 0.02
        # )

    def forward(self, x, BItoken=None):
        """
        处理流程：
        输入图像 → 生成块嵌入 → 拼接CLS Token → 添加位置编码
        """
        # 生成原始块嵌入
        x, H, W = self.patch_embed(x)  # (B, N, E)
        if BItoken is not None:
            x = torch.cat([BItoken, x], dim=1)  # (B, N+1, E)

        # # 如果存在CLS Token
        # if cls_token is not None:
        #     # 调整CLS Token维度
        #     cls_token = cls_token + self.cls_pos_embed  # 添加位置信息
        #     x = torch.cat([cls_token, x], dim=1)  # (B, N+1, E)

        return x, H, W

class Residual(nn.Module):
    def __init__(self, input_dim, output_dim, stride=1, padding=1):
        super(Residual, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0., with_cls_token=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Identity()  # 可插入 DropPath 实现
        self.norm2 = nn.LayerNorm(dim)
        self.with_cls_token = with_cls_token

        if with_cls_token:
            self.mlp = MixFFNWithCls(dim, int(dim * mlp_ratio), drop=drop)
        else:
            self.mlp = MixFFN(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

class FusionAttentionModule(nn.Module):
    def __init__(self, in_channels=480, out_channels =64, reduction=16, factor=8):
        super().__init__()
        # 1*1卷积降低通道数
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False)
        # 2. 全局自注意力
        self.EMA = EMA(out_channels, factor=factor)
        # 3. 通道注意力 空间注意力
        self.CBAM = CBAMBlock(out_channels, reduction)
        # 4. 融合卷积
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels+out_channels, out_channels+out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels+out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels+out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: FM (B, sum(emb_dim), H, W)
        F_reduced = self.conv(x) #(B, half, H, W)
        FM_EMA = self.EMA(F_reduced)  # 全局长程依赖 (B, half, H, W)
        FM_CBAM = self.CBAM(F_reduced)  # 通道加权 (B, half, H, W)
        FM_fused = torch.cat([FM_EMA, FM_CBAM], dim=1) #(B, original, H, W)
        FM_fused= x + FM_fused
        out = self.fuse(FM_fused)  # +残差，局部细节融合 (B, half, H, W)
        return out


class DynamicConvGenerator(nn.Module):
    """
    动态卷积核生成器
    输入形状：(B, 1, emb_dim) → 示例：(2,1,64)
    输出形状：动态卷积核参数 (B, C_out*C_in*K*K)
    """

    def __init__(self,
                 emb_dim=64,
                 in_channels=3,
                 out_channels=3,
                 kernel_size=3,
                 reduction_ratio=4):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 参数计算
        total_params = out_channels * in_channels * kernel_size ** 2
        hidden_dim = emb_dim // reduction_ratio

        # 参数生成网络
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, total_params),
            nn.Tanh()  # 限制参数范围[-1,1]
        )

        # 初始化参数
        self._initialize_weights()

    def _initialize_weights(self):
        # 最后线性层初始化为接近0的值
        nn.init.normal_(self.mlp[-2].weight, mean=0, std=0.01)
        nn.init.constant_(self.mlp[-2].bias, 0)

    def forward(self, cls_token):
        """
        生成动态卷积核参数
        输入：cls_token (B,1,emb_dim)
        输出：卷积核参数 (B, C_out, C_in, K, K)
        """
        B = cls_token.size(0)
        # 压缩序列维度
        x = cls_token.squeeze(1)  # (B, emb_dim)

        # 生成原始参数
        params = self.mlp(x)  # (B, total_params)

        # 重塑为卷积核形状
        kernel = params.view(
            B,
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size
        )  # (B, C_out, C_in, K, K)

        return kernel


class DynamicConvolution(nn.Module):
    """
    动态卷积执行模块
    输入：
    - features: 辅助分支特征 (B, C_in, H, W)
    - kernel: 动态生成的卷积核 (B, C_out, C_in, K, K)
    输出：处理后的特征 (B, C_out, H, W)
    """

    def __init__(self, padding=1):
        super().__init__()
        self.padding = padding

    def forward(self, features, dynamic_kernel):
        B, C_out, C_in, K, _ = dynamic_kernel.shape
        _, _, H, W = features.shape

        # 将特征展开为im2col格式
        unfolded = F.unfold(features, K, padding=self.padding)  # (B, C_in*K*K, L)
        L = unfolded.size(-1)  # L = H' * W'
        # unfolded = unfolded.view(B, 1, C_in * K * K, H * W)  # (B, 1, C_in*K*K, L)

        # 重塑卷积核
        kernel = dynamic_kernel.view(B, C_out, C_in * K * K)  # (B, C_out, C_in*K*K)

        # 批矩阵乘法 (每个样本独立计算)
        output = torch.bmm(kernel, unfolded)  # (B, C_out, L)

        # 执行批量矩阵乘法
        # output = torch.matmul(kernel, unfolded)  # (B, C_out, L)

        # 重塑为特征图格式
        output = output.view(B, C_out, H, W)

        return output


class CLSControlledDynamicBlock(nn.Module):
    """
    完整的CLS控制动态卷积模块
    流程：
    CLS Token → 生成动态卷积核 → 应用动态卷积 → 残差连接
    """

    def __init__(self,
                 emb_dim=64,
                 in_channels=3,
                 out_channels=3,
                 kernel_size=3):
        super().__init__()
        self.generator = DynamicConvGenerator(
            emb_dim=emb_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size
        )
        self.conv = DynamicConvolution(padding=kernel_size // 2)

        # 通道对齐（可选）
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, cls_token, features):
        """
        输入：
        - cls_token: (B,1,emb_dim)
        - features: 辅助分支特征 (B,C,H,W)
        输出：处理后的特征 (B,C_out,H,W)
        """
        # 生成动态卷积核
        dynamic_kernel = self.generator(cls_token)  # (B,C_out,C_in,K,K)

        # 应用动态卷积
        conv_feat = self.conv(features, dynamic_kernel)

        # 残差连接
        shortcut = self.shortcut(features)
        output = conv_feat + shortcut

        return output
class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        [b, c, row, col] = x.size()
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)

        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result

class CrossGuidanceBlock(nn.Module):
    def __init__(self, low_channels, high_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(low_channels, low_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(high_channels, low_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(high_channels, low_channels, kernel_size=1)
        self.proj = nn.Conv2d(low_channels, low_channels, kernel_size=1)

    def forward(self, x_low, x_high):
        _, _, H, W = x_low.shape
        x_high_up = F.interpolate(x_high, size=(H, W), mode='bilinear', align_corners=False)
        query = self.query_conv(x_low)
        key = self.key_conv(x_high_up)
        value = self.value_conv(x_high_up)
        attn = torch.sigmoid(query * key)
        out = attn * value
        out = self.proj(out)
        return x_low + out
class Wudi(nn.Module):
    def __init__(self, img_size=512, patch_size=16, stride=16, in_chans=3, num_classes=2, embed_dims=[64,96,128,192], d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True, depths=[3, 4, 6, 3], mlp_ratios=[4, 4, 4, 4],  num_heads=[1, 2, 4, 8],
                 use_checkpoint=False, **kwargs):
        super(Wudi, self).__init__()
        # Define Parameters
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.img_size = img_size

        # Define Functions
        self.edge_lap = LaplaceConv2d(in_channels=3, out_channels=1)
        self.Boundary_Info_Tokenizer = CrossModalCLSToken(in_channels=1, embed_dim=embed_dims[0])
        self.patch_embed1 = Make_Transformer_Input(
            img_size, patch_size=7, stride=4,
            in_chans=in_chans, embed_dim=embed_dims[0]
        )
        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.block1 = nn.ModuleList([
            Block(
                dim=embed_dims[0], num_heads=num_heads[0],
                mlp_ratio=mlp_ratios[0], drop_path=dpr[i],with_cls_token=True
            ) for i in range(depths[0])
        ])
        self.patch_embed2 = Make_Transformer_Input(
            img_size//4, patch_size=3, stride=2,
            in_chans=embed_dims[0], embed_dim=embed_dims[1]
        )
        self.patch_embed3 = Make_Transformer_Input(
            img_size//8, patch_size=3, stride=2,
            in_chans=embed_dims[1], embed_dim=embed_dims[2]
        )
        self.patch_embed4 = Make_Transformer_Input(
            img_size//16, patch_size=3, stride=2,
            in_chans=embed_dims[2], embed_dim=embed_dims[3]
        )
        self.block2 = nn.ModuleList([
            Block(
                dim=embed_dims[1], num_heads=num_heads[1],
                mlp_ratio=mlp_ratios[1], drop_path=dpr[i + depths[0]]
            ) for i in range(depths[1])
        ])

        # 阶段3的Transformer块
        self.block3 = nn.ModuleList([
            Block(
                dim=embed_dims[2], num_heads=num_heads[2],
                mlp_ratio=mlp_ratios[2], drop_path=dpr[i + sum(depths[:2])]
            ) for i in range(depths[2])
        ])

        # 阶段4的Transformer块
        self.block4 = nn.ModuleList([
            Block(
                dim=embed_dims[3], num_heads=num_heads[3],
                mlp_ratio=mlp_ratios[3], drop_path=dpr[i + sum(depths[:3])]
            ) for i in range(depths[3])
        ])
        # self.FusionAttentionModule = FusionAttentionModule(in_channels=sum(embed_dims), out_channels =64, reduction=16, factor=32)
        self.stage_fusions = nn.ModuleList([
            FusionAttentionModule(
                in_channels=dim,  # 各阶段输入通道
                out_channels=dim//2,  # 通道数减半
                reduction=16,
                factor=8
            ) for dim in embed_dims
        ])
        #setting 1
        # self.midconv = nn.Sequential(
        #     nn.Conv2d(sum(embed_dims)//2, 32, 3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True))
        #setting 2:
        self.midconv = nn.Sequential(
        nn.Conv2d(sum(embed_dims)//2, 48, 3, padding=1),
        nn.BatchNorm2d(48),
        nn.ReLU(inplace=True))

        self.bdr_reduce = nn.Conv2d(64, 16,1)
        self.ASPP = ASPP(64, 64, rate=1)
        self.finalCBAM = CBAMBlock(64, reduction_ratio=16)

        #输出各预测最终结果的卷积层
        self.out_feature = nn.Conv2d(64,1,1)
        self.edge_feature = nn.Conv2d(64, self.num_classes, 1)
        #输出距离图
        self.dis_feature = nn.Conv2d(64,1,1)
        self.conv1 = Residual(1, 32) # 定义laplace卷积后的残差卷积
        self.bdrCBAM = CBAMBlock(32, reduction_ratio=16)
        self.dynamic_block = CLSControlledDynamicBlock(
        emb_dim=embed_dims[0],
        in_channels=1,
        out_channels=32,
        kernel_size=3)
        self.cross1 = CrossGuidanceBlock(embed_dims[2], embed_dims[3])  # 2 <- 3
        self.cross2 = CrossGuidanceBlock(embed_dims[1], embed_dims[2])  # 1 <- 2
        self.cross3 = CrossGuidanceBlock(embed_dims[0], embed_dims[1])  # 0 <- 1
        self.refine_headbdr = PointRefineHead(in_channels=2)
        self.refine_headmain = PointRefineHead(in_channels=1)

    def forward(self, x):
        B = x.shape[0]
        Trans_features = []
        Bdr_Branch = self.edge_lap(x)  # (B, 3, 256, 256) -- (B, 1, 256, 256)
        Bdr_token = self.Boundary_Info_Tokenizer(Bdr_Branch)  #(B, 1, emb_dim)

        #Transformer主分支
        Trans_Branch, H1, W1= self.patch_embed1(x, BItoken=Bdr_token) #（B, 4096(H*W)+1,64(emb_dim) )
        for blk in self.block1:
            Trans_Branch = blk(Trans_Branch, H1, W1)  # 保持形状
            # 重塑为图像格式
        Trans_Branch_lv1 = Trans_Branch[:, :-1, :]#取(B, 4096(H*W),64(emb_dim) )作为第一层输出和下一层输入

        Feedback_token = Trans_Branch[:, -1, :] #取(B, 1, 64(emb_dim) )作为
        Trans_Branch_lv1 = Trans_Branch_lv1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2)  # → (B,64,64,64)
        Trans_features.append(Trans_Branch_lv1)  # 保存1/4分辨率特征
        Trans_Branch_lv2, H2, W2 = self.patch_embed2(Trans_Branch_lv1, None) # → (B,1024(32*32),96(第二层emb_dim))

        for blk in self.block2:
            Trans_Branch_lv2 = blk(Trans_Branch_lv2, H2, W2)  # 保持形状
            # 重塑为图像格式
        Trans_Branch_lv2 = Trans_Branch_lv2.reshape(B, H2, W2, -1).permute(0, 3, 1, 2)  # → (B,96,32,32)
        Trans_features.append(Trans_Branch_lv2)  # 保存1/8分辨率特征
        Trans_Branch_lv3, H3, W3 = self.patch_embed3(Trans_Branch_lv2, None) # → (B,256(16*16),128(第三层emb_dim))

        for blk in self.block3:
            Trans_Branch_lv3 = blk(Trans_Branch_lv3, H3, W3)  # 保持形状
            # 重塑为图像格式
        Trans_Branch_lv3 = Trans_Branch_lv3.reshape(B, H3, W3, -1).permute(0, 3, 1, 2)  # → (B,128,16,16)
        Trans_features.append(Trans_Branch_lv3)  # 保存1/8分辨率特征
        Trans_Branch_lv4, H4, W4 = self.patch_embed4(Trans_Branch_lv3, None)  # → (B,64(8*8),192(第四层emb_dim))

        for blk in self.block4:
            Trans_Branch_lv4 = blk(Trans_Branch_lv4, H4, W4)  # 保持形状
            # 重塑为图像格式
        Trans_Branch_lv4 = Trans_Branch_lv4.reshape(B, H4, W4, -1).permute(0, 3, 1, 2)  # → (B,192,8,8)
        Trans_features.append(Trans_Branch_lv4)  # 保存1/16分辨率特征
        #接下来，进行针对多层Transformer输出的上采样工作
        fused_features = []


        fused2 = self.cross1(Trans_features[2], Trans_features[3])  # 2引导3
        fused1 = self.cross2(Trans_features[1], fused2)
        fused0 = self.cross3(Trans_features[0], fused1)
        fused_features.append(fused0)
        fused_features.append(fused1)
        fused_features.append(fused2)
        fused_features.append(Trans_features[3])
        up_features=[]



        for i, (feat, fusion_module) in enumerate(zip(fused_features, self.stage_fusions)):
            # 对每个阶段的特征分别进行注意力融合
            combined = fusion_module(feat)
            up_feat = F.interpolate(
                combined,
                size=(self.img_size, self.img_size),
                mode='bilinear',
                align_corners=True
            )
            up_features.append(up_feat)  #第一层（B,dim[0]/2,64,64);第二层（B,dim[1]/2,32,32)；第三层（B,dim[2]/2,16,16)；第四层（B,dim[3]/2,8,8)


        combined_feature = torch.cat(up_features, dim=1) # (B,64/2+96/2+128/2+192/2,512,512) = (B,240,512,512)

        FM = self.midconv(combined_feature) #(B,32,512,512)
        #setting 2:(B,48,512,512)


        # 边界分支
        Bdr_A = self.dynamic_block(Feedback_token, Bdr_Branch) #边界分支再分为两支，一支用由feedbacktoken控制的动态卷积决定. (B,1,512,512)--(B,32,512,512)
        Bdr_B = self.conv1(Bdr_Branch)

        Bdr_B = self.bdrCBAM(Bdr_B) #另一支由残差卷积+CBAM决定. (B,1,512,512)--(B,32,512,512)
        Bdr_Branch = torch.cat((Bdr_A, Bdr_B), dim=1) #拼接(B,64,512,512)
        #setting 2
        reduced_Bdr = self.bdr_reduce(Bdr_Branch) #(B,16,512,512)
        FM =torch.cat((FM, reduced_Bdr), dim=1)

        # FM =torch.cat((FM, Bdr_B), dim=1)  #Bdr_B分支与主干进行特征融合(B,32,512,512)拼(B,32,512,512)--(B,64,512,512)
        FM = self.ASPP(FM)  # (B,64,512,512)
        FM = self.finalCBAM(FM)  # (B,64,512,512)




        # 统一整合输出
        edge_out = self.edge_feature(Bdr_Branch)
        # edge_out = F.log_softmax(edge_out, dim=1)
        mask_out = self.out_feature(FM)
        dis_out = self.dis_feature(FM)
        point_feat_Main = mask_out.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        point_logits_Main = self.refine_headmain(point_feat_Main).squeeze(-1)  # (B, H*W)
        point_feat_bdr = edge_out.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        point_logits_bdr = self.refine_headbdr(point_feat_bdr).squeeze(-1)  # (B, H*W)
        # point_feat_all = edge_out.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        # point_logits_all = self.refine_head(point_feat_all).squeeze(-1)  # (B, H*W)

        return [mask_out,edge_out,dis_out], [point_logits_bdr, point_logits_Main]