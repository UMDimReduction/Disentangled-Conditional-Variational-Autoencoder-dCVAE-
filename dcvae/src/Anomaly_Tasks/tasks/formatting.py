import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 600
plt.rcParams['savefig.dpi'] = 600
font = {'weight' : 'bold',
        'size'   : 6}

import matplotlib
matplotlib.rc('font', **font)
import numpy as np
from PIL import Image


def save_imagegrid(image_grid, file_name):
    image_grid *= 255
    image_grid = image_grid.astype(np.uint8)
    image_grid = image_grid.transpose(1, 2, 0)
    img = Image.fromarray(image_grid)
    img.save(file_name)


def save_oscillating_video(video, file_name, duration=None):
    oscillation = _build_oscillating_video(video)
    save_video(oscillation, file_name, duration, loop=True)


def _build_oscillating_video(video):
    start_frame = video[0][None]
    end_frame = video[-1][None]
    num_frames = video.shape[0]
    num_still_fames = (5 * num_frames) // 4 - num_frames
    oscillation = _build_oscillation(video, start_frame, end_frame, num_still_fames)

    return oscillation


def _build_oscillation(video, start_frame, end_frame, num_still_fames):
    start_still = np.tile(start_frame, (num_still_fames // 2, 1, 1, 1))
    end_still = np.tile(end_frame, (num_still_fames - 2, 1, 1, 1))
    reverse_video = video[::-1]
    oscillation = [start_still, video, end_still, reverse_video, start_still]
    oscillation = np.concatenate(oscillation)

    return oscillation


def save_video(video, file_name, duration=None, loop=False):
    video *= 255
    video = video.transpose(0, 2, 3, 1)
    video = video.astype(np.uint8)
    duration = duration or video.shape[0] / 25
    _save_gif(video, file_name, duration, loop)


def _save_gif(video, file_name, duration, loop):
    loop = 0 if loop else 1
    duration = 1000 * duration / video.shape[0]
    img, *imgs = [Image.fromarray(frame) for frame in video]
    img.save(fp=file_name, format='GIF', append_images=imgs,
             save_all=True, duration=duration, loop=loop)


def save_roc_plot(tpr, fpr, auc, file_name):
    fig = plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=0.1, bottom=1, right=0.7, top=1.5, wspace=0.4, hspace=0.8)
    plot_roc(plt.gca(), fpr, tpr, auc)
    plt.savefig(file_name)
    plt.close(fig)


def plot_roc(ax, fpr, tpr, auc, title=None):
    ax.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right')
    if title is not None:
        ax.set_title(title)


def plot_reduction(ax, features, labels, title=None):
    classes = np.unique(labels)
    for cls, color in zip(classes, plt.cm.get_cmap('tab10').colors):
        class_features = features[labels == cls]
        ax.scatter(class_features[:, 0], class_features[:, 1], c=[color], label=cls, s=[2], alpha=0.5)
    if title is not None:
        ax.set_title(title)
