import numpy as np
import torch
import tsaug
from scipy.fft import fft

def DataTransform(sample, config):

    weak_aug = scaling(sample, config.augmentation.jitter_scale_ratio)
    strong_aug = scaling(reverse_time(sample), 0.2)

    return weak_aug, strong_aug


def jitter(x, sigma=0.05):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=1.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    ai = []
    for i in range(x.shape[1]):
        xi = x[:, i, :]
        ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    return np.concatenate((ai), axis=1)

def time_shift(x, shift_range=150):
    shifted_x = np.copy(x)
    for i in range(x.shape[0]):
        shift = np.random.randint(-shift_range, shift_range + 1)
        shifted_x[i, 0, :] = np.roll(x[i, 0, :], shift)
        shifted_x[i, 1, :] = np.roll(x[i, 1, :], shift)
    return shifted_x

def permutation(x, max_segments=9, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])
    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i, :, :] = pat[:, warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)

def time_warp(x):
    x = x.transpose(1, 2).numpy()
    y = tsaug.TimeWarp(n_speed_change=7, max_speed_ratio=4).augment(x)
    y = torch.from_numpy(y)

    return y.transpose(1, 2)

def reverse_time(x):
    x = x.transpose(1, 2).numpy()
    y = tsaug.Reverse().augment(x)
    y = torch.from_numpy(y)

    return y.transpose(1, 2)

def conv(x):
    x = x.transpose(1, 2).numpy()
    y = tsaug.Convolve(size=14).augment(x)
    y = torch.from_numpy(y)

    return y.transpose(1, 2)

def frequency(x):
    x_rearrange = torch.permute(x, (0, 2, 1)).contiguous()
    signals_time = torch.view_as_complex(x_rearrange)
    signals_freq = torch.fft.fft(signals_time, norm='ortho')
    signals_fft = torch.stack((signals_freq.real, signals_freq.imag), dim=1)

    return signals_fft

def pool(x):
    x = x.transpose(1, 2).numpy()
    y = tsaug.Pool(size=2).augment(x)
    y = torch.from_numpy(y)
    return y.transpose(1, 2)


def quantize(x):
    x = x.transpose(1, 2).numpy()
    y = tsaug.Quantize(n_levels=20).augment(x)
    y = torch.from_numpy(y)
    return y.transpose(1, 2)