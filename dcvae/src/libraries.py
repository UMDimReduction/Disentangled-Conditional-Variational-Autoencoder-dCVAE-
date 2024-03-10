import warnings
import argparse
import os; os.system('')
import mlflow
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 600
plt.rcParams['savefig.dpi'] = 600
font = {'weight' : 'bold',
        'size'   : 6}

import matplotlib
matplotlib.rc('font', **font)
import tensorflow_probability as tfp
import tensorflow as tf
from time import time
import glob
import imageio
import PIL
import PIL.Image
from IPython import display
import os, inspect, time, math
#import utils

from src.utils.loss_functions import log_bernouli_pdf, kl_divergence_standard_prior, total_correlation
from src.utils.utils import output_dims, save_images, image_generation, plot_latent_images
from src.utils.utils import plot_latent_images
from src.utils.data_loader import *
from sklearn.metrics import roc_curve, roc_auc_score
import json



from src.models.betaVAE import BetaVAE
from src.models.cvae import cvae
from src.models.dcvae import dcvae
from src.models.FactorVAE import FactorVAE
from src.models.static import VAE
import src.models.RFVAE as RFVAE


import pytorch_lightning as pl
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
import atexit
import itertools
import shutil
import tempfile
import torch.nn as nn
from tensorboard.backend.event_processing import event_accumulator
#from anomaly_task import load_ae_from_checkpoint
import pytorch_lightning.loggers as loggers
import umap
import datetime
from functools import reduce
from math import pow
from tensorboard.backend.event_processing import event_accumulator
import pytablewriter
#import data
import torchvision
from pytorch_lightning.core import lightning





