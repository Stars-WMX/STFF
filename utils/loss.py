import torch
import torch.nn as nn
import torch.nn.functional as F

class CharbonnierLoss(torch.nn.Module):
    def __init__(self, eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, X, Y):
        diff = X - Y
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss

class FFTLoss(torch.nn.Module):   
    def __init__(self):
        super(FFTLoss, self).__init__()
        self.L1 = nn.L1Loss(reduction='mean')

    def forward(self, img1, img2):
        img1=torch.stack([torch.fft.fft2(img1, dim=(-2, -1)).real, torch.fft.fft2(img1, dim=(-2, -1)).imag], -1)
        img2=torch.stack([torch.fft.fft2(img2, dim=(-2, -1)).real, torch.fft.fft2(img2, dim=(-2, -1)).imag], -1)
        return self.L1(img1, img2)

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(1,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.to('cuda')
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)
        down        = filtered[:,:,::2,::2]
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4
        filtered    = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, x, y):
        nx, tx, cx, hx, wx = x.size()
        ny, ty, cy, hy, wy = y.size()
        x = x.view(nx * tx, cx, hx, wx)  #  (n * t, c, h, w)
        y = y.view(ny * ty, cy, hy, wy)  #  (n * t, c, h, w)
        loss = self.loss(self.laplacian_kernel(x.to('cuda')), self.laplacian_kernel(y.to('cuda')))
        return loss

class CVQE_Loss(nn.Module):
    def __init__(self):
        super(CVQE_Loss, self).__init__()

        self.char = CharbonnierLoss()
        self.fft = FFTLoss()
        self.Edge = EdgeLoss()

    def forward(self, img1, img2):
        char = self.char(img1, img2)
        FFT = 0.01 * self.fft(img1, img2)
        Edge = 0.05 * self.Edge(img1, img2)
        loss = char + FFT + Edge
        return loss

class CVQE_Loss1(nn.Module):
    def __init__(self):
        super(CVQE_Loss1, self).__init__()
        self.char = CharbonnierLoss()
        self.Edge = EdgeLoss()

    def forward(self, img1, img2):
        char = self.char(img1, img2)
        Edge = 0.05 * self.Edge(img1, img2)
        loss = char + Edge
        return loss

class CVQE_Loss2(nn.Module):
    def __init__(self):
        super(CVQE_Loss2, self).__init__()
        self.char = CharbonnierLoss()
        self.fft = FFTLoss()

    def forward(self, img1, img2):
        char = self.char(img1, img2)
        FFT = 0.01 * self.fft(img1, img2)
        loss = char + FFT
        return loss

