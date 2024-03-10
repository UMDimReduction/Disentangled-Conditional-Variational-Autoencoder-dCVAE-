#%%
from src.libraries import *

import warnings

class VAE(object):
    @staticmethod
    def create_model(model='VAE', kwargs={}):
      model = model.lower()
      if model in ['beta-vae', 'bvae', 'b-vae']:
        return BetaVAE(**kwargs)
      elif model in ['dcvae', 'd-cvae']:
        return dcvae(**kwargs)
      elif model in ['cvae', 'conditional-vae']:
        return cvae(**kwargs)
      elif model in ['RFVAE', 'Relevance-FactorVAE']:
        return RFVAE(**kwargs)
      elif model in ['FactorVAE', 'factorvae']:
        return FactorVAE(**kwargs)

      #else:
      #  warnings.warn(f'no matched model name can be found for {model}, use vae instead')
       # return dcvae(**kwargs)
