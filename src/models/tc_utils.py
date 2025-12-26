"""
Total Correlation Computation Utilities
========================================

This module provides the TC computation that was MISSING from the original dCVAE.

The original paper claims to minimize TC but never actually computes it!
This module fixes that by implementing minibatch-weighted TC estimation.

Usage:
    from models.tc_utils import compute_tc_minibatch
    
    tc_loss = compute_tc_minibatch(z, mu, logvar)
"""

import torch
import numpy as np


def compute_tc_minibatch(z: torch.Tensor, 
                         mu: torch.Tensor, 
                         logvar: torch.Tensor) -> torch.Tensor:
    """
    Compute Total Correlation using minibatch-weighted sampling (β-TCVAE method).
    
    TC(z) = E_q(z)[log q(z) - log Π_j q(z_j)]
    
    where q(z) ≈ (1/N) Σ_i q(z|x_i) is approximated using minibatch samples.
    
    This is THE KEY CORRECTION - the original dCVAE paper claims to minimize TC
    but never actually computes it!
    
    Args:
        z: Sampled latent vectors [batch_size, latent_dim]
        mu: Encoder means [batch_size, latent_dim]
        logvar: Encoder log-variances [batch_size, latent_dim]
    
    Returns:
        tc: Scalar TC estimate
        
    Reference:
        Chen et al. (2018) "Isolating Sources of Disentanglement in VAEs" (β-TCVAE)
    """
    batch_size, latent_dim = z.shape
    
    # Clamp logvar for numerical stability
    logvar = torch.clamp(logvar, min=-20, max=20)
    
    # Expand dimensions for broadcasting
    z_expand = z.unsqueeze(1)  # [B, 1, D]
    mu_expand = mu.unsqueeze(0)  # [1, B, D]
    logvar_expand = logvar.unsqueeze(0)  # [1, B, D]
    
    # Log probability under Gaussian: log N(z; μ, σ²)
    var_expand = torch.exp(logvar_expand) + 1e-8
    log_qz_given_x = -0.5 * (
        ((z_expand - mu_expand) ** 2 / var_expand) +
        logvar_expand +
        np.log(2 * np.pi)
    ).sum(dim=2)  # [B, B]
    
    # log q(z) = logsumexp - log(batch_size)
    log_qz = torch.logsumexp(log_qz_given_x, dim=1) - np.log(batch_size)
    
    # log Π_j q(z_j) = Σ_j log q(z_j)
    log_prod_qzj = torch.zeros(batch_size, device=z.device)
    
    for j in range(latent_dim):
        z_j = z[:, j:j+1]
        mu_j = mu[:, j:j+1]
        logvar_j = logvar[:, j:j+1]
        
        z_j_expand = z_j.unsqueeze(1)
        mu_j_expand = mu_j.unsqueeze(0)
        logvar_j_expand = logvar_j.unsqueeze(0)
        var_j_expand = torch.exp(logvar_j_expand) + 1e-8
        
        log_qzj_given_x = -0.5 * (
            ((z_j_expand - mu_j_expand) ** 2 / var_j_expand) +
            logvar_j_expand +
            np.log(2 * np.pi)
        ).squeeze(-1)
        
        log_qzj = torch.logsumexp(log_qzj_given_x, dim=1) - np.log(batch_size)
        log_prod_qzj += log_qzj
    
    # TC = E[log q(z) - log Π_j q(z_j)]
    tc = (log_qz - log_prod_qzj).mean()
    
    return tc


def compute_mutual_info(z: torch.Tensor,
                        mu: torch.Tensor,
                        logvar: torch.Tensor) -> torch.Tensor:
    """
    Compute Mutual Information I(x; z) using minibatch estimation.
    
    I(x; z) = E_q(x,z)[log q(z|x) - log q(z)]
    """
    batch_size, latent_dim = z.shape
    
    logvar = torch.clamp(logvar, min=-20, max=20)
    var = torch.exp(logvar) + 1e-8
    
    # log q(z|x) for diagonal
    log_qz_given_x_diag = -0.5 * (
        ((z - mu) ** 2 / var) + logvar + np.log(2 * np.pi)
    ).sum(dim=1)
    
    # log q(z) via minibatch
    z_expand = z.unsqueeze(1)
    mu_expand = mu.unsqueeze(0)
    logvar_expand = logvar.unsqueeze(0)
    var_expand = torch.exp(logvar_expand) + 1e-8
    
    log_qz_given_x_all = -0.5 * (
        ((z_expand - mu_expand) ** 2 / var_expand) +
        logvar_expand +
        np.log(2 * np.pi)
    ).sum(dim=2)
    
    log_qz = torch.logsumexp(log_qz_given_x_all, dim=1) - np.log(batch_size)
    
    mi = (log_qz_given_x_diag - log_qz).mean()
    return mi
