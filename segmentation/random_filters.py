import math
import torch
import torch.nn.functional as F
from torch import nn
from monai.transforms import MapTransform, RandomizableTransform
from monai.utils import convert_to_dst_type, convert_to_tensor
from rafl.rafl import RandomWavelet, _conv

TRUNCATED_STD_RATIO = 0.87962566103423978

WAVELET_LOW_PASS = {
    "db1": [0.7071067811865476, 0.7071067811865476],
    "db2": RandomWavelet.LO,
    "db4": [-0.010597401785069032, 0.0328830116668852, 0.030841381835560764, -0.18703481171909309,
            -0.027983769416859854, 0.6308807679298589, 0.7148465705529157, 0.2303778133088965],
    "sym4": [-0.07576571478927333, -0.02963552764599851, 0.49761866763201545, 0.8037387518059161,
             0.29785779560527736, -0.09921954357684722, -0.012603967262037833, 0.0322231006040427],
}


class ConfigurableWavelet(RandomWavelet):
    """rafl `RandomWavelet` with a choice of wavelet.

    wavelet: key of WAVELET_LOW_PASS; high-pass is its quadrature mirror
    levels: number of à trous levels
    Source: rafl/rafl.py
    """

    def __init__(self, wavelet="db2", levels=4, dims=3):
        super().__init__(levels=levels, dims=dims)
        low_pass = torch.tensor(WAVELET_LOW_PASS[wavelet])
        signs = (-1.0) ** torch.arange(1, low_pass.numel() + 1)
        high_pass = signs * low_pass.flip(0)
        self.h0 = low_pass.flip(0)
        self.h1 = high_pass.flip(0)

    def _sep(self, x, filters, dil):
        """Separable à trous filtering, circularly padded.

        x: (B, C, *spatial)
        filters: one 1D filter per spatial axis
        dil: dilation of every filter
        -> (B, C, *spatial)
        """
        channels = x.shape[1]
        for axis, taps in enumerate(filters):
            length = taps.numel()
            shape = [1] * self.dims
            shape[axis] = length
            kernel = taps.to(x.dtype).view(1, 1, *shape).expand(channels, 1, *shape)
            pad_left = dil * ((length - 1) // 2)
            pad_right = dil * (length - 1) - pad_left
            pad = [0] * (2 * self.dims)
            pad_index = 2 * (self.dims - 1 - axis)
            pad[pad_index] = pad_left
            pad[pad_index + 1] = pad_right
            dilation = [1] * self.dims
            dilation[axis] = dil
            padded = F.pad(x, pad, mode="circular")
            x = _conv(self.dims)(padded, kernel, groups=channels, dilation=tuple(dilation))
        return x


class StackedRandomConv(nn.Module):
    """Linear random convolution: `layers` stacked k^dims convolutions, zero "same" padding, no bias or nonlinearity.

    channels: input channels
    k: kernel size of every layer, an int or one size per spatial axis
    layers: number of convolutions; the last has one output channel
    filters: output channels of every other layer
    dims: 3 convolves volumes, 2 slices
    Source: randConv_linear and Keras HeNormal Scholz et al., ai-idt/rc_aug randconv.py.
    """

    def __init__(self, channels=1, k=3, layers=3, filters=1, dims=3):
        super().__init__()
        kernel = (k,) * dims if isinstance(k, int) else tuple(k)
        self.padding = tuple(size // 2 for size in kernel)
        self.layers = layers
        self.dims = dims
        in_channels = channels
        for layer in range(layers):
            out_channels = 1 if layer == layers - 1 else filters
            self.register_buffer(f"w{layer}", torch.empty(out_channels, in_channels, *kernel))
            in_channels = out_channels
        self.redraw()

    def redraw(self):
        """Draw every layer's taps from Keras' HeNormal: a normal cut at ±2 std, widened so its std after the cut is
        the He std sqrt(2 / fan_in).

        Source: TRUNCATED_STD_RATIO from keras.initializers.VarianceScaling.
        """
        for layer in range(self.layers):
            weight = getattr(self, f"w{layer}")
            fan_in = weight[0].numel()
            he_std = math.sqrt(2.0 / fan_in)
            std_before_cut = he_std / TRUNCATED_STD_RATIO
            nn.init.trunc_normal_(weight, mean=0.0, std=std_before_cut, a=-2 * std_before_cut, b=2 * std_before_cut)

    def forward(self, x):
        for layer in range(self.layers):
            weight = getattr(self, f"w{layer}").to(x.dtype)
            x = _conv(self.dims)(x, weight, padding=self.padding)
        return x


FILTER_FAMILIES = {
    "conv_scholz": StackedRandomConv,
    "wavelet": RandomWavelet,
    "configurable_wavelet": ConfigurableWavelet,
}


def filtered_image(x, filter_module):
    """One filter draw per (sample, channel), its responses randomly projected to one image.

    x: (B, C, X, Y, Z)
    filter_module: single-channel 3D rafl filter; responses weighted by v ~ N(0, 1)
    -> (B, C, X, Y, Z)
    """
    filtered = torch.empty_like(x)
    for sample in range(x.shape[0]):
        for channel in range(x.shape[1]):
            volume = x[sample, channel][None, None]
            filter_module.redraw()
            responses = filter_module(volume)[0]
            weights = torch.randn(responses.shape[0], device=x.device, dtype=x.dtype)
            filtered[sample, channel] = torch.einsum("fxyz,f->xyz", responses, weights)
    return filtered


def augment_with_filter(x, filter_module, nonzero_rescale=False):
    """Replace each (sample, channel) with its randomly filtered version, kept in its own intensity range.

    x: (B, C, X, Y, Z)
    filter_module: single-channel rafl filter (`filtered_image`), applied to x min-max scaled to [0, 1]
    nonzero_rescale: filtered output to [0, 1] as Scholz et al.: shift only its non-zero voxels by their min, then
        divide by the max, so exact zeros (background far from tissue) stay 0; else plain min-max
    -> (B, C, X, Y, Z), min-max scaled back to each (sample, channel)'s input range
    Source: ai-idt/rc_aug 
    """
    spatial_dims = (2, 3, 4)
    low = x.amin(dim=spatial_dims, keepdim=True)
    span = (x.amax(dim=spatial_dims, keepdim=True) - low).clamp_min(1e-8)
    unit = (x - low) / span
    filtered = filtered_image(unit, filter_module)
    if nonzero_rescale:
        nonzero = filtered != 0
        nonzero_low = filtered.masked_fill(~nonzero, float("inf")).amin(dim=spatial_dims, keepdim=True)
        shifted = torch.where(nonzero, filtered - nonzero_low, filtered)
        filtered_unit = shifted / shifted.amax(dim=spatial_dims, keepdim=True).clamp_min(1e-8)
    else:
        filtered_low = filtered.amin(dim=spatial_dims, keepdim=True)
        filtered_span = (filtered.amax(dim=spatial_dims, keepdim=True) - filtered_low).clamp_min(1e-8)
        filtered_unit = (filtered - filtered_low) / filtered_span
    return filtered_unit * span + low


class RandFilterd(RandomizableTransform, MapTransform):
    """With probability `prob`, `augment_with_filter` on each key's (C, X, Y, Z) image, seeded from MONAI's RNG.

    family: "conv_scholz", "wavelet" (rafl filter) or "configurable_wavelet"
    nonzero_rescale: see `augment_with_filter`
    params: keyword arguments of the filter
    """

    def __init__(self, keys, family, prob, nonzero_rescale=False, **params):
        RandomizableTransform.__init__(self, prob)
        MapTransform.__init__(self, keys)
        filter_class = FILTER_FAMILIES[family]
        if family in ("wavelet", "configurable_wavelet"):
            self.filter_module = filter_class(dims=3, **params)
        else:
            self.filter_module = filter_class(channels=1, dims=3, **params)
        self.nonzero_rescale = nonzero_rescale

    def __call__(self, data):
        d = dict(data)
        self.randomize(None)
        if not self._do_transform:
            return d
        seed = int(self.R.randint(2**31))
        cuda_devices = []
        for key in self.key_iterator(d):
            if d[key].is_cuda:
                cuda_devices.append(d[key].device)
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            for key in self.key_iterator(d):
                image = convert_to_tensor(d[key], track_meta=False, dtype=torch.float32)
                self.filter_module.to(image.device)
                augmented = augment_with_filter(image[None], self.filter_module, self.nonzero_rescale)
                d[key] = convert_to_dst_type(augmented[0], d[key])[0]
        return d
