import torch
from torch import nn, Tensor
import numpy as np
from typing import Optional

class GaussianEncoding(nn.Module):
    """Layer for mapping coordinates using random Fourier features"""

    def __init__(self, sigma: Optional[float] = None,
                 input_size: Optional[float] = None,
                 encoded_size: Optional[float] = None,
                 b: Optional[Tensor] = None):
        super().__init__()
        if b is None:
            if sigma is None or input_size is None or encoded_size is None:
                raise ValueError(
                    'Arguments "sigma," "input_size," and "encoded_size" are required.')
            b = self.sample_b(sigma, (encoded_size, input_size))
        elif sigma is not None or input_size is not None or encoded_size is not None:
            raise ValueError('Only specify the "b" argument when using it.')
        self.register_buffer('b', b)

    def forward(self, v: Tensor) -> Tensor:
        v = v.double()  # Ensure v is double
        vp = 2 * np.pi * torch.matmul(v, self.b.T)
        return torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)

    @staticmethod
    def sample_b(sigma, size):
        return torch.randn(size) * sigma

class MultiLayerNet(nn.Module):
    def __init__(self, D_in, H, D_out, act_func, CNN_dev, rff_dev, N_Layers):
        super(MultiLayerNet, self).__init__()

        H = int(H)
        self.encoding = GaussianEncoding(sigma=rff_dev, input_size=D_in, encoded_size=H // 2)

        N_Layers = int(N_Layers)
        self.linear = nn.ModuleList()

        for ii in range(N_Layers):
            if ii == 0:
                self.linear.append(nn.Linear(D_in, H))
            elif ii == (N_Layers - 1):
                self.linear.append(nn.Linear(H, D_out))
            else:
                self.linear.append(nn.Linear(H, H))

            nn.init.constant_(self.linear[ii].bias, 0.)
            nn.init.normal_(self.linear[ii].weight, mean=0, std=CNN_dev)

    def forward(self, x, N_Layers, act_fn):
        activation_fn = getattr(torch, act_fn)
        y_auto = []

        for ii in range(N_Layers):
            if ii == 0:
                y_auto.append(self.encoding(x.double()))  # Ensure x is double
            elif ii == (N_Layers - 1):
                y_auto.append(self.linear[-1](y_auto[-1]))
            else:
                y_auto.append(activation_fn(self.linear[ii](y_auto[ii - 1])))

        return y_auto[-1]

