import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
from model.SRU import SRUNet

class FEA(nn.Module):
    def __init__(self, in_nc, out_nc, nf, base_ks=3, deform_ks=3):
        super(FEA, self).__init__()

        self.in_nc = in_nc
        self.deform_ks = deform_ks
        self.size_dk = deform_ks ** 2
        self.in_conv = nn.Sequential(nn.Conv2d(in_nc, 2*nf, base_ks, padding=base_ks // 2), nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.unet = SRUNet(dim=2*nf)
        self.out_conv = nn.Sequential(nn.Conv2d(2*nf, nf, base_ks, padding=base_ks // 2), nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.offset_mask = nn.Conv2d(nf, in_nc * 3 * self.size_dk, base_ks, padding=base_ks // 2)
        self.deform_conv = DeformConv2d(in_nc, out_nc, deform_ks, padding=deform_ks // 2)

    def forward(self, inputs):
        out = self.in_conv(inputs)
        out = self.unet(out)
        off_msk = self.offset_mask(self.out_conv(out))
        off = off_msk[:, :self.in_nc * 2 * self.size_dk, ...]
        msk = torch.sigmoid(off_msk[:, self.in_nc * 2 * self.size_dk:, ...])
        fused_feat = F.relu(self.deform_conv(inputs, off, msk), inplace=True)
        return fused_feat
