import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv(dims):
    return F.conv3d if dims == 3 else F.conv2d


class RandomWavelet(nn.Module):
    """Stationary (a trous) db2 wavelet with 4 levels; at each level one random unit direction u mixes the
    first-order detail bands (high-pass along one axis, low-pass along the others), u0 * LH + u1 * HL in 2D,
    per channel. dims = 3 filters volumes, with three detail bands per level."""
    LO = [-0.12940952255092145, 0.22414386804185735, 0.8365163037378079, 0.48296291314469025]
    HI = [-0.48296291314469025, 0.8365163037378079, -0.22414386804185735, -0.12940952255092145]

    def __init__(self, levels=4, dims=2):
        super().__init__()
        self.levels, self.dims = levels, dims
        self.register_buffer('h0', torch.tensor(self.LO).flip(0))
        self.register_buffer('h1', torch.tensor(self.HI).flip(0))
        self.register_buffer('u', torch.zeros(levels, dims))
        self.redraw()

    def redraw(self):
        u = torch.randn(self.levels, self.dims, device=self.u.device)
        self.u.copy_(u / u.norm(dim=1, keepdim=True).clamp_min(1e-12))

    def _sep(self, x, filters, dil):
        C, L = x.shape[1], filters[0].numel()
        pl = dil * ((L - 1) // 2); pr = dil * (L - 1) - pl
        for axis, f in enumerate(filters):
            shape = [1] * self.dims; shape[axis] = L
            k = f.to(x.dtype).view(1, 1, *shape).expand(C, 1, *shape)
            pad = [0] * (2 * self.dims); pad[2 * (self.dims - 1 - axis)] = pl; pad[2 * (self.dims - 1 - axis) + 1] = pr
            dilation = [1] * self.dims; dilation[axis] = dil
            x = _conv(self.dims)(F.pad(x, pad, mode='circular'), k, groups=C, dilation=tuple(dilation))
        return x

    def forward(self, x):
        a, outs = x, []
        for j in range(1, self.levels + 1):
            if j > 1:
                a = self._sep(a, [self.h0] * self.dims, 2 ** (j - 2))
            d = 2 ** (j - 1)
            u = self.u[j - 1].to(x.dtype)
            out = 0
            for i in range(self.dims):
                high_axis = self.dims - 1 - i
                filters = [self.h1 if axis == high_axis else self.h0 for axis in range(self.dims)]
                out = out + u[i] * self._sep(a, filters, d)
            outs.append(out)
        return torch.cat(outs, dim=1)
