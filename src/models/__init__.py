from .vae import VAE
from .dcvae import dCVAE
from .factorvae import FactorVAE
from .betavae import BetaVAE
from .rfvae import RFVAE
from .cvae import CVAE

AUTOENCODERS = {
    'vae': VAE,
    'dcvae': dcvae,
    'factorvae': FactorVAE,
    'betavae': BetaVAE,
    'rfvae': RFVAE,
    'cvae': CVAE
}
