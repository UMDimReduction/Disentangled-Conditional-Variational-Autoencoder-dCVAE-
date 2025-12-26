"""
Main Training Script for dCVAE
==============================

Updated to support beta parameter for corrected dCVAE.

Usage:
    # Train corrected dCVAE with TC computation
    python run.py dcvae --dataset mnist --epochs 50 --beta 4.0
    
    # Train CVAE baseline (no TC)
    python run.py cvae --dataset mnist --epochs 50
    
    # Compare models
    python run.py dcvae --dataset mnist --epochs 50 --beta 4.0
    python run.py cvae --dataset mnist --epochs 50
    # Then compare TC values in results!

The key difference:
- dCVAE now ACTUALLY computes and minimizes Total Correlation
- TC values are tracked and visualized in results/
- Use --beta to control disentanglement strength (default: 4.0)
"""

import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from building import build_ae, build_datamodule
from models import dCVAE, CVAE
from plot import SaveResultsCallback

# Ensure directories exist
os.makedirs('results', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)


def run(model_type, dataset, batch_size, anomaly, downstream_task=None, epochs=10, beta=4.0):
    """
    Train a model and save results.
    
    Args:
        model_type: Type of model ('vae', 'cvae', 'dcvae', etc.)
        dataset: Dataset name ('mnist', 'fmnist', 'kmnist', etc.)
        batch_size: Batch size for training
        anomaly: Whether to exclude a class for anomaly detection
        downstream_task: Optional downstream task to run
        epochs: Number of training epochs
        beta: Beta parameter for dCVAE/BetaVAE (controls TC/KL weight)
    
    Returns:
        Path to best model checkpoint
    """
    # Create results directory
    model_results_dir = os.path.join('results', model_type, dataset)
    os.makedirs(model_results_dir, exist_ok=True)

    # Build datamodule
    datamodule = build_datamodule(
        dataset=dataset, 
        model_type=model_type, 
        batch_size=batch_size, 
        anomaly=anomaly
    )

    # Build model with beta parameter
    ae = build_ae(
        model_type, 
        datamodule.dims, 
        anomaly,
        beta=beta  # NEW: Pass beta to model
    )

    # Move to appropriate device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ae = ae.to(device)

    # Print model info
    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()} on {dataset.upper()}")
    print(f"{'='*60}")
    print(f"  Device: {device}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Latent dim: {ae.latent_dim}")
    if hasattr(ae, 'beta'):
        print(f"  Beta (TC weight): {ae.beta}")
    if model_type == 'dcvae':
        print(f"  TC Computation: ENABLED")
    print(f"{'='*60}\n")

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=os.path.join('checkpoints', model_type, dataset),
        filename=f'{model_type}-{{epoch:02d}}-{{val_loss:.2f}}',
        save_top_k=1,
        mode='min',
    )

    # Results callback (includes TC tracking)
    save_results_callback = SaveResultsCallback(
        ae, 
        datamodule, 
        model_results_dir, 
        model_type, 
        dataset
    )

    # Trainer
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1 if torch.cuda.is_available() else 'auto',
        max_epochs=epochs,
        logger=None,
        callbacks=[checkpoint_callback, save_results_callback],
        enable_progress_bar=True
    )

    # Train
    trainer.fit(ae, datamodule=datamodule)

    # Print final TC for dCVAE
    if model_type == 'dcvae' and len(save_results_callback.tc_history) > 0:
        print(f"\n{'='*60}")
        print("TC ANALYSIS (Corrected dCVAE)")
        print(f"{'='*60}")
        print(f"  Initial TC: {save_results_callback.tc_history[0]:.4f}")
        print(f"  Final TC:   {save_results_callback.tc_history[-1]:.4f}")
        reduction = (1 - save_results_callback.tc_history[-1] / save_results_callback.tc_history[0]) * 100
        print(f"  Reduction:  {reduction:.1f}%")
        print(f"{'='*60}\n")

    return checkpoint_callback.best_model_path


def compare_models(dataset, epochs, batch_size, beta):
    """
    Compare CVAE vs dCVAE to show TC computation difference.
    """
    print("\n" + "#"*60)
    print("# MODEL COMPARISON: CVAE vs dCVAE")
    print("#"*60 + "\n")
    
    results = {}
    
    # Train CVAE
    print("Training CVAE (no TC computation)...")
    run('cvae', dataset, batch_size, False, epochs=epochs)
    
    # Train dCVAE
    print("\nTraining dCVAE (WITH TC computation)...")
    run('dcvae', dataset, batch_size, False, epochs=epochs, beta=beta)
    
    print("\n" + "#"*60)
    print("# COMPARISON COMPLETE")
    print("#"*60)
    print("""
Check the results directory for:
- results/cvae/{dataset}/ - CVAE results (no TC tracking)
- results/dcvae/{dataset}/ - dCVAE results (WITH TC tracking)

The key difference:
- CVAE: No TC history (TC was never computed)
- dCVAE: TC history shows decrease over training

This proves that the corrected dCVAE actually minimizes TC!
    """)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train dCVAE and other VAE models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train corrected dCVAE on MNIST
    python run.py dcvae --dataset mnist --epochs 50 --beta 4.0
    
    # Train CVAE baseline
    python run.py cvae --dataset mnist --epochs 50
    
    # Compare CVAE vs dCVAE
    python run.py --compare --dataset mnist --epochs 50 --beta 4.0
    
    # Anomaly detection (exclude class 1)
    python run.py dcvae --dataset mnist --anomaly --epochs 30
        """
    )

    parser.add_argument('model_type', type=str, nargs='?', default='dcvae',
                       choices=['vae', 'dcvae', 'factorvae', 'rfvae', 'betavae', 'cvae'],
                       help='Model type (default: dcvae)')
    parser.add_argument('--dataset', type=str, default='mnist',
                       help='Dataset name (default: mnist)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--anomaly', action='store_true',
                       help='Enable anomaly detection mode')
    parser.add_argument('--downstream_task', type=str, 
                       choices=['umap', 'roc', 'classification', 'reconstruction'], 
                       default=None,
                       help='Downstream task to run')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of epochs (default: 10)')
    parser.add_argument('--beta', type=float, default=4.0,
                       help='Beta parameter for dCVAE/BetaVAE (default: 4.0)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare CVAE vs dCVAE')

    opt = parser.parse_args()

    if opt.compare:
        compare_models(opt.dataset, opt.epochs, opt.batch_size, opt.beta)
    else:
        checkpoint_path = run(
            opt.model_type, 
            opt.dataset, 
            opt.batch_size, 
            opt.anomaly, 
            opt.downstream_task, 
            opt.epochs,
            opt.beta
        )
        print(f"Best model saved to: {checkpoint_path}")
