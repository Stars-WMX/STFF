import math
import torch
import torch.nn as nn
import numpy as np
from einops.layers.torch import Rearrange, Reduce


class SKFF(nn.Module):
    def __init__(self, in_channels, height=2, reduction=8, bias=False):
        super(SKFF, self).__init__()
        self.height = height

        d = max(int(in_channels / reduction), 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.PReLU())

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats = inp_feats[0].shape[1]
        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])
        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)
        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)

        return feats_V

#  将输入的图像按照指定的窗口大小分割成多个子窗口，并重新排列它们的维度
def img2windows(img, H_sp, W_sp):
    """
    Input: Image (B, C, H, W)
    Output: Window Partition (B', N, C)
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp* W_sp, C)

    return img_perm # 输出的是按照指定的窗口大小分割后的子窗口

# 将窗口分割的图像重新组合成原始的图像形状
def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    Input: Window Partition (B', N, C)
    Output: Image (B, H, W, C)
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img

# 多层感知器（MLP）模型
class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.hidden_dims = hidden_features
        self.in_dims = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        _, n, _ = x.shape
        self.N = n
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class DynamicPosBias(nn.Module):
    # The implementation builds on Crossformer code https://github.com/cheerss/CrossFormer/blob/main/models/crossformer.py
    """ Dynamic Relative Position Bias.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        residual (bool):  If True, use residual strage to connect conv.
    """
    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )
    def forward(self, biases):
        self.l, self.c = biases.shape
        if self.residual:
            pos = self.pos_proj(biases) # 2Gh-1 * 2Gw-1, heads
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos


class Attention_regular(nn.Module):
    """ Regular Rectangle-Window (regular-Rwin) self-attention with dynamic relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        resolution (int): Input resolution.
        idx (int): The identix of V-Rwin and H-Rwin, 0 is H-Rwin, 1 is Vs-Rwin. (different order from Attention_axial)
        split_size (tuple(int)): Height and Width of the regular rectangle window (regular-Rwin).
        dim_out (int | None): The dimension of the attention output. Default: None
        num_heads (int): Number of attention heads. Default: 6
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set
        position_bias (bool): The dynamic relative position bias. Default: True
    """
    def __init__(self, dim, idx, split_size=None, dim_out=None, num_heads=6, qk_scale=None, position_bias=True):
        super().__init__()
        if split_size is None:
            split_size = [2, 4]
        self.dim = dim
        self.dim_out = dim_out or dim
        self.split_size = split_size
        self.num_heads = num_heads
        self.idx = idx
        self.position_bias = position_bias

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        if idx == 0:
            H_sp, W_sp = self.split_size[0], self.split_size[1]
        elif idx == 1:
            W_sp, H_sp = self.split_size[0], self.split_size[1]
        else:
            print("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp

        self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)
        self.softmax = nn.Softmax(dim=-1)


    def im2win(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp* self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    # 它接受三个输入张量 qkv，其中包含查询（q）、键（k）和值（v），以及输入图像的高度（H）和宽度（W）。
    # 在前向传播过程中，模块首先将输入的查询、键和值通过 im2win 方法转换为窗口形式，并计算自注意力权重。
    # 然后，模块将动态相对位置偏置应用于注意力权重，并通过 softmax 层进行归一化。最后，模块将注意力权重乘以值张量，并将结果转换回图像形式，并返回输出张量
    def forward(self, qkv, H, W, mask=None, rpi=None, rpe_biases=None):
        """
        Input: qkv: (B, 3*L, C), H, W, mask: (B, N, N), N is the window size
        Output: x (B, H, W, C)
        """
        q, k, v = qkv[0], qkv[1], qkv[2]

        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        self.N = L//(self.H_sp * self.W_sp)

        # partition the q,k,v, image to window
        q = self.im2win(q, H, W)
        k = self.im2win(k, H, W)
        v = self.im2win(v, H, W)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N

        # calculate drpe
        pos = self.pos(rpe_biases)
        # select position bias
        relative_position_bias = pos[rpi.view(-1)].view(self.H_sp * self.W_sp, self.H_sp * self.W_sp, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        N = attn.shape[3]

        # use mask for shift window
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)

        x = (attn @ v)
        x = x.transpose(1, 2).reshape(-1, self.H_sp* self.W_sp, C)  # B head N N @ B head N C

        # merge the window, window to image
        x = windows2img(x, self.H_sp, self.W_sp, H, W)  # B H' W' C

        return x

# def forward(self, x, x_size, params, attn_mask=NotImplementedError): 接受四个参数
class SRWINBlock(nn.Module):
    r""" Shift Rectangle Window Attention Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        split_size (int): Define the window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, split_size=(2,2), shift_size=(0,0), mlp_ratio=2., qkv_bias=True, qk_scale=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.split_size = split_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.branch_num = 2
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim) # DW Conv

        self.attns = nn.ModuleList([Attention_regular(dim//2, idx=i, split_size=split_size, num_heads=num_heads//2, dim_out=dim//2, qk_scale=qk_scale, position_bias=True)
                for i in range(self.branch_num)])

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x, x_size, params, attn_mask=NotImplementedError):
        h, w = x_size
        self.h, self.w = x_size

        b, l, c = x.shape
        shortcut = x
        x = self.norm1(x)
        qkv = self.qkv(x).reshape(b, -1, 3, c).permute(2, 0, 1, 3) # 3, B, HW, C
        v = qkv[2].transpose(-2,-1).contiguous().view(b, c, h, w)

        # cyclic shift
        if self.shift_size[0] > 0 or self.shift_size[1] > 0:
            qkv = qkv.view(3, b, h, w, c)
            # H-Shift
            qkv_0 = torch.roll(qkv[:,:,:,:,:c//2], shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(2, 3))
            qkv_0 = qkv_0.view(3, b, h*w, c//2)
            # V-Shift
            qkv_1 = torch.roll(qkv[:,:,:,:,c//2:], shifts=(-self.shift_size[1], -self.shift_size[0]), dims=(2, 3))
            qkv_1 = qkv_1.view(3, b, h*w, c//2)

            # H-Rwin
            x1_shift = self.attns[0](qkv_0, h, w, mask=attn_mask[0], rpi=params['rpi_sa_h'], rpe_biases=params['biases_h'])
            # V-Rwin
            x2_shift = self.attns[1](qkv_1, h, w, mask=attn_mask[1], rpi=params['rpi_sa_v'], rpe_biases=params['biases_v'])

            x1 = torch.roll(x1_shift, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
            x2 = torch.roll(x2_shift, shifts=(self.shift_size[1], self.shift_size[0]), dims=(1, 2))
            # Concat
            attened_x = torch.cat([x1,x2], dim=-1)
        else:
            # H-Rwin
            x1 = self.attns[0](qkv[:,:,:,:c//2], h, w, rpi=params['rpi_sa_h'], rpe_biases=params['biases_h'])
            # V-Rwin
            x2 = self.attns[1](qkv[:,:,:,c//2:], h, w, rpi=params['rpi_sa_v'], rpe_biases=params['biases_v'])
            # Concat
            attened_x = torch.cat([x1,x2], dim=-1)

        attened_x = attened_x.view(b, -1, c).contiguous()

        # Locality Complementary Module
        lcm = self.get_v(v)
        lcm = lcm.permute(0, 2, 3, 1).contiguous().view(b, -1, c)

        attened_x = attened_x + lcm

        attened_x = self.proj(attened_x)

        # FFN
        x = shortcut + attened_x
        x = x + self.mlp(self.norm2(x))
        return x  #  b, l, c # # def forward(self, x, x_size, params, attn_mask=NotImplementedError): #

class ChannelAtten(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAtten, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, (1, 1), 1, 0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, (1, 1), 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.attention(x)

class RDCAB(nn.Module):
    def __init__(self, nf, gc, bias=True):
        super(RDCAB, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.se = ChannelAtten(nf, reduction=16)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x6 = self.se(x5)
        return x6 * 0.2 + x

class ConvTransBlock(nn.Module):
    def __init__(self, dim, depth, num_heads, split_size_0=2, split_size_1=2, mlp_ratio=2., qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm, gc=32):
        """
        SRwinTransformer and RConv Block
        """
        super(ConvTransBlock, self).__init__()
        self.dim = dim
        self.conv_dim = dim // 2
        self.trans_dim = dim // 2
        self.depth = depth
        self.gc = gc

        # SRwinT blocks  Shift Rectangle window attention blocks
        self.trans_block = SRWINBlock(
                                    dim=self.trans_dim,
                                    num_heads=num_heads,
                                    split_size=[split_size_0, split_size_1],
                                    shift_size=[split_size_0 // 2, split_size_1 // 2],
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    norm_layer=norm_layer
                                )

        self.conv1_1 = nn.Conv2d(self.conv_dim + self.trans_dim, self.conv_dim + self.trans_dim, 1, 1, 0, bias=True) # 最前面的 1*1
        self.conv1_2 = nn.Conv2d(self.conv_dim + self.trans_dim, self.conv_dim + self.trans_dim, 1, 1, 0, bias=True)# 最后面的 1*1

        # AdaResidual rrdb block 局部建模
        self.conv_block = RDCAB(nf=self.conv_dim, gc=self.conv_dim //2)


    def forward(self, x, x_size, params): # n 128 h w

        conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)
        conv_x = self.conv_block(conv_x)  # n 64 h w

        trans_x = Rearrange('b c h w -> b h w c')(trans_x)
        b, h, w, c = trans_x.shape
        h, w = x_size
        trans_x = trans_x.reshape(b, h*w, c)
        trans_x = self.trans_block(trans_x, x_size, params, params['attn_mask'])
        trans_x = trans_x.reshape(b, h, w, c)
        trans_x = Rearrange('b h w c -> b c h w')(trans_x) #  n 64 h w

        res = self.conv1_2(torch.cat((conv_x, trans_x), dim=1)) # n 128 h w
        x = x + res
        return x


class SRUNet(nn.Module):
    def __init__(self, dim=64,
                 depth=2,
                 num_heads=8,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 split_size_0=2,
                 split_size_1=2,
                 norm_layer=nn.LayerNorm):
        super(SRUNet, self).__init__()
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.split_size_0 = split_size_0
        self.split_size_1 = split_size_1
        self.split_size=(2, 2)
        self.activation = nn.LeakyReLU(0.2, True)
        # relative position index
        self.calculate_rpi_v_sa()

        # m_down1
        self.en_layer1_1 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1), self.activation)

        # m_down2
        self.en_layer2_1 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1), self.activation)

        # m_down3
        self.en_layer3_1 = nn.Sequential(nn.Conv2d(dim, 2*dim, kernel_size=3, stride=2, padding=1), self.activation)

        # # body
        self.body = ConvTransBlock(dim=2*dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale, split_size_0=split_size_0, split_size_1=split_size_1, norm_layer=norm_layer)

        #m_up1
        self.de_layer1_1 = nn.Sequential(nn.ConvTranspose2d(2*dim, dim, kernel_size=4, stride=2, padding=1), self.activation)

        # m_up2
        self.de_layer2_1 = nn.Sequential(nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1), self.activation)

        # m_up3
        self.de_layer3_1 = nn.Sequential(nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1), self.activation)

        self.SKFF_1 = SKFF(in_channels=dim, height=2, reduction=8)
        self.SKFF_2 = SKFF(in_channels=dim, height=2, reduction=8)

        # 计算垂直和水平方向上的相对位置索引和位置偏置
    def calculate_rpi_v_sa(self):
            # generate mother-set
            H_sp, W_sp = self.split_size[0], self.split_size[1]
            position_bias_h = torch.arange(1 - H_sp, H_sp)
            position_bias_w = torch.arange(1 - W_sp, W_sp)
            biases_h = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))
            biases_h = biases_h.flatten(1).transpose(0, 1).contiguous().float()

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(H_sp)
            coords_w = torch.arange(W_sp)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += H_sp - 1
            relative_coords[:, :, 1] += W_sp - 1
            relative_coords[:, :, 0] *= 2 * W_sp - 1
            relative_position_index_h = relative_coords.sum(-1)

            H_sp, W_sp = self.split_size[1], self.split_size[0]
            position_bias_h = torch.arange(1 - H_sp, H_sp)
            position_bias_w = torch.arange(1 - W_sp, W_sp)
            biases_v = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))
            biases_v = biases_v.flatten(1).transpose(0, 1).contiguous().float()

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(H_sp)
            coords_w = torch.arange(W_sp)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += H_sp - 1
            relative_coords[:, :, 1] += W_sp - 1
            relative_coords[:, :, 0] *= 2 * W_sp - 1
            relative_position_index_v = relative_coords.sum(-1)
            self.register_buffer('relative_position_index_h', relative_position_index_h)
            self.register_buffer('relative_position_index_v', relative_position_index_v)
            self.register_buffer('biases_v', biases_v)
            self.register_buffer('biases_h', biases_h)

            return biases_v, biases_h

    @torch.jit.ignore
    def no_weight_decay(self):
            return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        params = {'attn_mask': (None, None), 'rpi_sa_h': self.relative_position_index_h, 'rpi_sa_v': self.relative_position_index_v, 'biases_v': self.biases_v, 'biases_h': self.biases_h}
        x = self.body(x, x_size,  params)
        return x

    def forward(self, x0):
        h, w = x0.size()[-2:]  # n 64 h w
        paddingBottom = int(np.ceil(h / 64) * 64 - h)
        paddingRight = int(np.ceil(w / 64) * 64 - w)
        x0 = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x0)

        x2 = x0 # n 64 h w
        #x2 = self.ConvTransBlock1(x1) # n 64 h w

        # m_down1
        x3 = self.en_layer1_1(x2) # n 64 h/2 w/2
        # m_down2
        x4 = self.en_layer2_1(x3) # n 64 h/4 w/4
        # m_down1=3
        x5 = self.en_layer3_1(x4) # n 64 h/8 w/8
        # body
        x = self.forward_features(x5) # n 128 h/8 w/8

        # m_up1
        x6 = self.de_layer1_1(x)  # n 64 h/4 w/4
        x7 = self.SKFF_1([x4, x6])  # n 64 h/4 w/4
        # m_up2
        x8 = self.de_layer2_1(x7) # n 64 h/2 w/2
        x9 = self.SKFF_2([x3, x8])  # n 64 h/2 w/2
        # m_up3
        x10 = self.de_layer3_1(x9) # n 64 h w

        x11 = x10 + x2  # n 64 h w

        # 在这里进行切片以去掉填充
        x_final = x11[:, :, :h, :w]

        return x_final
