import torch
import numpy as np
import torch.nn as nn
from model.FEA import FEA
from model.QE_HFERB1_RDB6_HFERB1_Relu import QE_HFERB_RDCAB

def generate_it(x, t=0, nf=3, f=7):
    index = np.array([t - nf // 2 + i for i in range(nf)])
    index = np.clip(index, 0, f-1).tolist()
    it = x[:, index, :, :]
    return it

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
    def __init__(self, in_nc, dim) -> None:
        super().__init__()
        self.in_nc = in_nc
        self.dim = dim
        self.mid_dim = dim//2
        self.act = nn.ReLU(inplace=True)

        self.first_fc = nn.Conv2d(self.in_nc, self.dim, 1)
        self.conv = nn.Conv2d(self.mid_dim, self.mid_dim, 3, 1, 1)
        self.fc = nn.Conv2d(self.mid_dim, self.mid_dim, 1)
        self.max_pool = nn.MaxPool2d(3, 1, 1)

        self.last_fc = nn.Conv2d(self.dim, self.dim, 1)

        self.se = ChannelAttention(dim, reduction=32)

    def forward(self, x):
        short = self.first_fc(x)
        lfe = self.act(self.conv(short[:,:self.mid_dim,:,:]))
        hfe = self.act(self.fc(self.max_pool(short[:,self.mid_dim:,:,:])))

        x = torch.cat([lfe, hfe], dim=1)
        x = self.se(x)
        x = short + self.last_fc(x) * 0.2
        return x

class STFF_L(nn.Module):
    def __init__(self, in_nc=7, nf=32, out_nc=64, cpu_cache_length=15):
        super(STFF_L, self).__init__()

        self.out_nc = out_nc  # 64
        self.cpu_cache_length = cpu_cache_length
        self.FEA = FEA(in_nc=in_nc, out_nc=out_nc, nf=nf, deform_ks=3)

        self.HFERB = HFERB(in_nc=out_nc, dim=out_nc)

        self.backward_Align = FEA(in_nc=2 * out_nc, out_nc=out_nc, nf=nf, deform_ks=1)
        self.forward_Align = FEA(in_nc=3 * out_nc, out_nc=out_nc, nf=nf, deform_ks=1)

        self.fuse = nn.Sequential(nn.Conv2d(3 * self.out_nc, self.out_nc, 3, padding=1), nn.ReLU(inplace=True))
        self.qenet = QE_HFERB_RDCAB(in_nc=2 * self.out_nc, nf=self.out_nc, gc=self.out_nc // 2, out_nc=1)

    def forward(self, inputs):
        n, t, h, w = inputs.size()
        #n, t, c, h, w = inputs.size()
        if t > self.cpu_cache_length:
            self.cpu_cache = True
        else:
            self.cpu_cache = False
        #####################################Backward Propagation############################################
        Backward_List = []
        feat = inputs.new_zeros(n, self.out_nc, h, w)
        for i in range(t - 1, -1, -1):
            out1 = generate_it(inputs, i, 7, t) # n t h w
            out = self.FEA(out1)
            feat = torch.cat([out, feat], dim=1)
            feat = self.backward_Align(feat)
            HF = self.HFERB(feat)
            feat = self.fuse(torch.cat([out, HF, feat], dim=1)) + out
            if self.cpu_cache:
                Backward_List.append(feat.cpu())
                torch.cuda.empty_cache()
            else:
                Backward_List.append(feat)
        Backward_List = Backward_List[::-1]
        #####################################Forward Propagation############################################
        Enhanced = []
        feat = inputs.new_zeros(n, self.out_nc, h, w)

        for i in range(0, t):
            future = Backward_List[i] if i == t - 1 else Backward_List[i + 1]
            present = Backward_List[i]

            if self.cpu_cache:
                present = present.cuda()
                future = future.cuda()

            feat = torch.cat([feat, present, future], dim=1)
            feat = self.forward_Align(feat)
            HF = self.HFERB(feat)
            feat = self.fuse(torch.cat([present, HF, feat], 1)) + present
            if self.cpu_cache:
                out = self.qenet(torch.cat([Backward_List[i].cuda(), feat], dim=1)) + inputs[:, i:i + 1, :, :]
                Enhanced.append(out.cpu())
                torch.cuda.empty_cache()
            else:
                out = self.qenet(torch.cat([Backward_List[i], feat], dim=1)) + inputs[:, i:i + 1, :, :]
                Enhanced.append(out)

        return torch.stack(Enhanced, dim=1)

if __name__ == "__main__":
    torch.cuda.set_device(3)
    net = STFF_L().cuda()

    from thop import profile

    with torch.no_grad():
        input = torch.randn(1, 1, 1280, 720).cuda()
        flops, params = profile(net, inputs=(input,))
        total = sum([param.nelement() for param in net.parameters()])
        print('Number of params: %.2fM' % (total / 1e6))
        print('Number of FLOPs: %.2fTFLOPs' % (flops / (1e9 * 1024)))










