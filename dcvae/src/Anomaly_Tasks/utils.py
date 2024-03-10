import atexit
import itertools
import shutil
import tempfile
import torch

import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 600
plt.rcParams['savefig.dpi'] = 600
font = {'weight' : 'bold',
        'size'   : 6}

import matplotlib
matplotlib.rc('font', **font)
import torch.nn as nn


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def freeze_layer(m):
    """Freezes the given layer for updates."""
    if isinstance(m, nn.Linear):
        if m.weight is not None:
            m.weight.requires_grad_(False)
        if m.bias is not None:
            m.bias.requires_grad_(False)
    elif isinstance(m, nn.BatchNorm1d):
        if m.weight is not None:
            m.weight.requires_grad_(False)
        if m.bias is not None:
            m.bias.requires_grad_(False)
        m.eval()


def tempdir():
    tempdir_path = tempfile.mkdtemp()
    atexit.register(shutil.rmtree, tempdir_path)

    return tempdir_path


def get_axes_grid(num_subplots, ncols, ax_size):
    fig, axes = _get_grid(num_subplots, ncols, ax_size)
    _deactivate_unused_axes(axes, num_subplots)

    return fig, axes


def _get_grid(num_subplots, ncols, ax_size):
    nrows = num_subplots // ncols
    nrows += 1 if nrows * ncols < num_subplots else 0
    figsize = (ax_size * ncols, ax_size * nrows)
    plt.subplots_adjust(left=0.1, bottom=1, right=0.9, top=1.5, wspace=0.4, hspace=0.8)
    fig, axes = plt.subplots(nrows, ncols,
                             sharey='all',
                             sharex='all',
                             figsize=figsize)
    axes = axes.ravel()
    return fig, axes


def _deactivate_unused_axes(axes, num_subplots):
    unused_axes = axes[num_subplots:]
    for unused_ax in unused_axes:
        unused_ax.set_axis_off()


def idx2onehot(idx, n):
    assert torch.max(idx).item() < n

    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n).to(idx.device)
    onehot.scatter_(1, idx, 1)

    return onehot