import torch
from torch import nn

class ChannelAttention(nn.Module):
    def __init__(self, num_features, reduction):
        super(ChannelAttention, self).__init__()
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, num_features // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // reduction, num_features, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.module(x)

class HFERB(nn.Module):
    def __init__(self, dim, in_nc) -> None:
        super().__init__()
        self.mid_dim = dim//2
        self.dim = dim
        self.in_nc = in_nc
        self.act = nn.ReLU(inplace=True)

        self.first_fc = nn.Conv2d(self.in_nc, self.dim, 1)
        # Local feature extraction branch
        self.conv = nn.Conv2d(self.mid_dim, self.mid_dim, 3, 1, 1)

        # High-frequency enhancement branch
        self.fc = nn.Conv2d(self.mid_dim, self.mid_dim, 1)
        self.max_pool = nn.MaxPool2d(3, 1, 1)

        self.last_fc = nn.Conv2d(self.dim, self.dim, 1)

        self.se = ChannelAttention(dim, reduction=16)

    def forward(self, x):
        short = self.first_fc(x)
        # Local feature extraction branch
        lfe = self.act(self.conv(x[:,:self.mid_dim,:,:]))

        # High-frequency enhancement branch
        hfe = self.act(self.fc(self.max_pool(x[:,self.mid_dim:,:,:])))

        x = torch.cat([lfe, hfe], dim=1)
        x = self.se(x)
        x = short + self.last_fc(x) * 0.2
        return x


class RDCAB(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(RDCAB, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.se = ChannelAttention(nf, reduction=16)

    def forward(self, x): # n 64 h w
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1)) # n 64 h w
        out = self.se(x5) # n 64 h w
        return out * 0.2 + x

class QE_HFERB_RDCAB(nn.Module):
    """Residual in Residual Dense Block"""
    def __init__(self, in_nc, nf=64, gc=32, out_nc=1, base_ks=3):
        super(QE_HFERB_RDCAB, self).__init__()

        self.nf = nf
        self.gc = nf // 2

        self.in_conv = nn.Sequential(nn.Conv2d(in_nc, nf, base_ks, padding=1), nn.ReLU(inplace=True))
        self.out_conv = nn.Conv2d(nf, out_nc, base_ks, padding=1)

        self.HFERB1 = HFERB(nf, nf)
        self.RDCAB1 = RDCAB(nf, gc)
        self.HFERB2 = HFERB(nf, nf)

    def forward(self, x):  # n 64 h w

        x = self.in_conv(x)

        out = self.HFERB1(x) #
        out = self.RDCAB1(out)
        out = self.HFERB2(out)

        out = self.out_conv(out)

        return out
