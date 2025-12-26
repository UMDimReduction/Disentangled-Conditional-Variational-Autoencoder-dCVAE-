"""
Models Package for dCVAE Project
=================================

Available models:
- VAE: Standard Variational Autoencoder
- CVAE: Conditional VAE (baseline, no TC)
- dCVAE: CORRECTED Disentangled CVAE with TC computation
- BetaVAE: β-VAE with weighted KL
- FactorVAE: VAE with discriminator-based TC
- RFVAE: Relevance Factor VAE
"""

from .vae import VAE
from .dcvae import dCVAE
from .factorvae import FactorVAE
from .betavae import BetaVAE
from .rfvae import RFVAE
from .cvae import CVAE
from .tc_utils import compute_tc_minibatch, compute_mutual_info

AUTOENCODERS = {
    'vae': VAE,
    'dcvae': dCVAE,
    'cvae': CVAE,
    'factorvae': FactorVAE,
    'betavae': BetaVAE,
    'rfvae': RFVAE
}

__all__ = [
    'VAE', 'CVAE', 'dCVAE', 'BetaVAE', 'FactorVAE', 'RFVAE',
    'compute_tc_minibatch', 'compute_mutual_info',
    'AUTOENCODERS'
]
