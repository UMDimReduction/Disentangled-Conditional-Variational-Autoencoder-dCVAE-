"""
CORRECTED Disentangled Conditional Variational Autoencoder (dCVAE)
===================================================================

This is the CORRECTED implementation that ACTUALLY computes Total Correlation!

The original implementation was identical to CVAE (no TC computation).

Corrected Loss Function:
    L_dCVAE = -E[log p(x|z,c)] + D_KL(q(z|x,c) || p(z)) + (β-1)·TC(z|c)

Key Changes from Original:
1. Added compute_tc_minibatch() for TC estimation
2. Added (β-1)·TC term to loss
3. Logs TC separately for tracking
4. Added beta parameter to control disentanglement strength

Author: [Your Name]
Date: [Date]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

# Import TC computation - THE KEY ADDITION!
from models.tc_utils import compute_tc_minibatch


class dCVAE(pl.LightningModule):
    """
    CORRECTED Disentangled Conditional Variational Autoencoder.
    
    The original dCVAE implementation was IDENTICAL to CVAE - it never computed TC!
    This corrected version actually computes and minimizes Total Correlation.
    
    Args:
        input_shape: Shape of input data (e.g., (1, 32, 32))
        latent_dim: Dimension of latent space
        condition_dim: Dimension of condition (number of classes)
        beta: TC weight parameter (β > 1 for stronger disentanglement)
        lr: Learning rate
    """
    
    def __init__(self, input_shape, latent_dim, condition_dim, beta=4.0, lr=1e-3):
        super(dCVAE, self).__init__()
        self.input_shape = input_shape
        self.input_dim = int(torch.prod(torch.tensor(input_shape)))
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.beta = beta  # NEW: TC weight parameter
        self.lr = lr

        # Encoder network: q_φ(z|x,c)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim + condition_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * latent_dim)  # mu and log_var
        )

        # Decoder network: p_θ(x|z,c)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.input_dim),
            nn.Sigmoid()
        )

    def encode(self, x, condition):
        """Encode input and condition to latent distribution parameters."""
        if condition.dim() == 1:
            condition = F.one_hot(condition, num_classes=self.condition_dim).float()
        condition = condition.view(-1, self.condition_dim)
        x = torch.cat((x.view(-1, self.input_dim), condition), dim=1)
        encoded = self.encoder(x)
        mu, log_var = torch.chunk(encoded, 2, dim=1)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        """Reparameterization trick: z = μ + σ·ε"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z, condition):
        """Decode latent and condition to reconstruction."""
        if condition.dim() == 1:
            condition = F.one_hot(condition, num_classes=self.condition_dim).float()
        condition = condition.view(-1, self.condition_dim)
        z = torch.cat((z, condition), dim=1)
        return self.decoder(z)

    def forward(self, x, condition):
        """Full forward pass."""
        mu, log_var = self.encode(x, condition)
        z = self.reparameterize(mu, log_var)
        return self.decode(z, condition), mu, log_var

    def training_step(self, batch, batch_idx):
        """
        CORRECTED training step with TC computation.
        
        Original: loss = recon_loss + kl_loss  (NO TC!)
        Corrected: loss = recon_loss + kl_loss + (β-1)·tc_loss
        """
        x, y = batch
        y_one_hot = F.one_hot(y, num_classes=self.condition_dim).float()
        
        # Forward pass
        mu, log_var = self.encode(x, y_one_hot)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z, y_one_hot)
        
        # 1. Reconstruction loss: -E[log p(x|z,c)]
        recon_loss = F.binary_cross_entropy(
            x_hat.view(-1, self.input_dim), 
            x.view(-1, self.input_dim), 
            reduction='sum'
        ) / x.shape[0]
        
        # 2. KL divergence: D_KL(q(z|x,c) || p(z))
        kl_loss = -0.5 * torch.sum(
            1 + log_var - mu.pow(2) - log_var.exp()
        ) / x.shape[0]
        
        # 3. Total Correlation: TC(z|c) - THE KEY CORRECTION!
        # Original code: This was NEVER computed!
        tc_loss = compute_tc_minibatch(z, mu, log_var)
        
        # Total loss: L = Recon + KL + (β-1)·TC
        # Note: Standard KL already contains TC with weight 1
        # Adding (β-1)·TC gives total weight β on the TC term
        loss = recon_loss + kl_loss + (self.beta - 1) * tc_loss
        
        # Log all components separately
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_recon', recon_loss)
        self.log('train_kl', kl_loss)
        self.log('train_tc', tc_loss, prog_bar=True)  # NEW: Track TC!
        
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step with TC tracking."""
        x, y = batch
        y_one_hot = F.one_hot(y, num_classes=self.condition_dim).float()
        
        mu, log_var = self.encode(x, y_one_hot)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z, y_one_hot)
        
        recon_loss = F.binary_cross_entropy(
            x_hat.view(-1, self.input_dim), 
            x.view(-1, self.input_dim), 
            reduction='sum'
        ) / x.shape[0]
        
        kl_loss = -0.5 * torch.sum(
            1 + log_var - mu.pow(2) - log_var.exp()
        ) / x.shape[0]
        
        tc_loss = compute_tc_minibatch(z, mu, log_var)
        
        val_loss = recon_loss + kl_loss + (self.beta - 1) * tc_loss
        
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_tc', tc_loss)
        
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# =============================================================================
# ORIGINAL (WRONG) IMPLEMENTATION - FOR REFERENCE ONLY
# =============================================================================
"""
The original dCVAE training_step was:

def training_step(self, batch, batch_idx):
    x, y = batch
    y_one_hot = F.one_hot(y, num_classes=self.condition_dim).float()
    x_hat, mu, log_var = self(x, y_one_hot)
    recon_loss = F.binary_cross_entropy(...)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    loss = recon_loss + kl_loss  # ← NO TC TERM! This is just CVAE!
    self.log('train_loss', loss)
    return loss

This was IDENTICAL to CVAE! The paper claims TC minimization but never implemented it!
"""
