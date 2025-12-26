"""
Plotting and Evaluation Module for dCVAE
=========================================

Updated to include TC tracking and visualization.

New features:
- TC history tracking during training
- TC comparison plots
- Loss component visualization
"""

import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend BEFORE importing pyplot
import matplotlib.pyplot as plt
import torch
import umap.umap_ as umap
import torch.nn as nn
from pytorch_lightning.callbacks import Callback
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np
from models import dCVAE, CVAE


class SaveResultsCallback(Callback):
    """
    Callback for saving results after each epoch.
    
    Updated to track TC for dCVAE models.
    """
    
    def __init__(self, ae, datamodule, results_dir, model_type, dataset_name):
        super().__init__()
        self.ae = ae
        self.datamodule = datamodule
        self.results_dir = results_dir
        self.model_type = model_type
        self.dataset_name = dataset_name

        # Create subfolder for each dataset and model type
        self.model_results_dir = os.path.join(results_dir, model_type, dataset_name)
        if not os.path.exists(self.model_results_dir):
            os.makedirs(self.model_results_dir)
        
        # NEW: Track TC history for dCVAE
        self.tc_history = []
        self.loss_history = {'total': [], 'recon': [], 'kl': [], 'tc': []}

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        print(f"Saving results for Epoch {epoch+1}")
        
        # NEW: Track TC if available (for dCVAE)
        if hasattr(trainer, 'logged_metrics'):
            metrics = trainer.logged_metrics
            if 'train_tc' in metrics:
                self.tc_history.append(float(metrics['train_tc']))
            if 'train_loss' in metrics:
                self.loss_history['total'].append(float(metrics['train_loss']))
            if 'train_recon' in metrics:
                self.loss_history['recon'].append(float(metrics['train_recon']))
            if 'train_kl' in metrics:
                self.loss_history['kl'].append(float(metrics['train_kl']))
            if 'train_tc' in metrics:
                self.loss_history['tc'].append(float(metrics['train_tc']))

        # Save latent manifold visualization
        plot_latent_manifold(self.ae, self.datamodule.test_dataloader(), self.model_results_dir, epoch)

        # Save reconstruction results
        plot_reconstruction(self.ae, self.datamodule.test_dataloader(), self.model_results_dir, epoch, self.model_type)

        # Save UMAP visualization
        handle_downstream_task(self.ae, self.datamodule, 'umap', self.model_results_dir, epoch)

        # Save ROC analysis
        handle_downstream_task(self.ae, self.datamodule, 'roc', self.model_results_dir, epoch)

        # Save classification accuracy
        handle_downstream_task(self.ae, self.datamodule, 'classification', self.model_results_dir, epoch)
        
        # NEW: Save TC history plot for dCVAE
        if len(self.tc_history) > 0:
            plot_tc_history(self.tc_history, self.model_results_dir, epoch)
        
        # NEW: Save loss components plot
        if len(self.loss_history['total']) > 0:
            plot_loss_components(self.loss_history, self.model_results_dir, epoch)


def plot_tc_history(tc_history, model_results_dir, epoch):
    """
    NEW: Plot Total Correlation over training epochs.
    
    This shows that dCVAE actually minimizes TC!
    """
    plt.figure(figsize=(10, 6), facecolor='white')
    
    epochs = range(1, len(tc_history) + 1)
    plt.plot(epochs, tc_history, 'b-', linewidth=2, marker='o', markersize=4)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Total Correlation', fontsize=12)
    plt.title(f'Total Correlation Over Training (Epoch {epoch + 1})', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add annotation showing TC reduction
    if len(tc_history) > 1:
        reduction = (1 - tc_history[-1] / tc_history[0]) * 100
        plt.annotate(f'TC Reduction: {reduction:.1f}%', 
                    xy=(len(tc_history), tc_history[-1]),
                    xytext=(len(tc_history) * 0.6, tc_history[0] * 0.8),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red')
    
    output_path = os.path.join(model_results_dir, f'epoch_{epoch + 1}_tc_history.png')
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved TC history plot to {output_path}")


def plot_loss_components(loss_history, model_results_dir, epoch):
    """
    NEW: Plot all loss components over training.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), facecolor='white')
    
    epochs = range(1, len(loss_history['total']) + 1)
    
    # Total loss
    if loss_history['total']:
        axes[0, 0].plot(epochs, loss_history['total'], 'b-', linewidth=2)
        axes[0, 0].set_title('Total Loss', fontsize=12)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Reconstruction loss
    if loss_history['recon']:
        axes[0, 1].plot(epochs, loss_history['recon'], 'g-', linewidth=2)
        axes[0, 1].set_title('Reconstruction Loss', fontsize=12)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
    
    # KL divergence
    if loss_history['kl']:
        axes[1, 0].plot(epochs, loss_history['kl'], 'r-', linewidth=2)
        axes[1, 0].set_title('KL Divergence', fontsize=12)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Total Correlation
    if loss_history['tc']:
        axes[1, 1].plot(epochs, loss_history['tc'], 'm-', linewidth=2)
        axes[1, 1].set_title('Total Correlation', fontsize=12)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('TC')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'TC not tracked\n(not dCVAE model)', 
                       ha='center', va='center', fontsize=12)
        axes[1, 1].set_title('Total Correlation', fontsize=12)
    
    plt.tight_layout()
    output_path = os.path.join(model_results_dir, f'epoch_{epoch + 1}_loss_components.png')
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()


def handle_downstream_task(ae, datamodule, task, model_results_dir, epoch):
    """Handle downstream evaluation tasks."""
    dataloader = datamodule.test_dataloader()
    ae.eval()

    all_inputs, all_latents, all_labels = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            x, labels = batch
            x, labels = x.to(ae.device), labels.to(ae.device)
            condition = torch.nn.functional.one_hot(labels, num_classes=10).float().to(ae.device)

            if isinstance(ae, (CVAE, dCVAE)):
                mu, log_var = ae.encode(x, condition)
            else:
                mu, log_var = ae.encode(x)
            z = ae.reparameterize(mu, log_var)
            all_inputs.append(x.cpu())
            all_latents.append(z.cpu())
            all_labels.append(labels.cpu())

    all_inputs = torch.cat(all_inputs, dim=0)
    all_latents = torch.cat(all_latents, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    if task == 'umap':
        plot_umap(all_latents, all_labels, model_results_dir, epoch)
    elif task == 'roc':
        run_roc_analysis(ae, datamodule, model_results_dir, epoch)
    elif task == 'classification':
        run_classification_error(ae, datamodule, model_results_dir, epoch)


def plot_latent_manifold(ae, dataloader, model_results_dir, epoch):
    """Generate and plot the latent manifold."""
    ae.eval()

    n = 10
    digit_size = 32
    figure = np.zeros((digit_size * n, digit_size * n))

    grid_x = np.linspace(-2, 2, n)
    grid_y = np.linspace(-2, 2, n)

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.zeros((1, ae.latent_dim))
            z_sample[0, 0] = xi
            z_sample[0, 1] = yi

            z_sample = torch.from_numpy(z_sample).float().to(ae.device)

            if isinstance(ae, (CVAE, dCVAE)):
                condition = torch.eye(10)[i % 10].unsqueeze(0).to(ae.device)
                with torch.no_grad():
                    x_decoded = ae.decode(z_sample, condition)
            else:
                with torch.no_grad():
                    x_decoded = ae.decode(z_sample)

            output_size = x_decoded.numel()
            digit_size = int(np.sqrt(output_size))

            if output_size != digit_size * digit_size:
                raise ValueError(f"Expected image size {digit_size}x{digit_size}, but got {output_size} elements")

            x_decoded = x_decoded.view(digit_size, digit_size).cpu().numpy()
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = x_decoded

    plt.figure(figsize=(10, 10), facecolor='white')
    plt.imshow(figure, vmin=0, vmax=1, cmap=plt.cm.binary)

    # Scale from 0 to 250 on both axes
    total_size = digit_size * n  # 320 pixels total
    tick_positions = np.linspace(0, total_size - 1, 6)  # 6 ticks: 0, 50, 100, 150, 200, 250
    tick_labels = [0, 50, 100, 150, 200, 250]
    plt.xticks(tick_positions, tick_labels)
    plt.yticks(tick_positions, tick_labels)

    plt.gca().set_facecolor('white')
    plt.grid(False)

    plt.title(f'Latent Manifold (Epoch {epoch + 1})', color='black')
    output_path = os.path.join(model_results_dir, f'epoch_{epoch + 1}_latent_manifold.png')
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_umap(latents, labels, model_results_dir, epoch):
    """Create UMAP visualization of latent space."""
    if latents is None or len(latents) == 0:
        print("No latents found, skipping UMAP plot.")
        return

    reducer = umap.UMAP()
    latent_reduced = reducer.fit_transform(latents)

    if len(latent_reduced) != len(labels):
        print(f"Warning: Latent representations ({len(latent_reduced)}) do not match labels ({len(labels)}).")
        labels = labels[:len(latent_reduced)]

    # Scale to 0-1 range
    min_vals = latent_reduced.min(axis=0)
    max_vals = latent_reduced.max(axis=0)
    latent_reduced_scaled = (latent_reduced - min_vals) / (max_vals - min_vals + 1e-8)

    # Color palette matching original paper
    class_colors = {
        0: '#98accd', 1: '#ea8024', 2: '#4ca947', 3: '#d12227', 4: '#824098',
        5: '#a1766e', 6: '#f5a0c5', 7: '#d2cdcc', 8: '#fce274', 9: '#aeddf7'
    }

    colors = [class_colors[int(label)] for label in labels]

    plt.figure(facecolor='white')
    plt.scatter(latent_reduced_scaled[:, 0], latent_reduced_scaled[:, 1], c=colors)

    # Scale from 0 to 1, with intervals of 0.2
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=class_colors[i], markersize=8) for i in range(10)]
    plt.legend(handles, [f'Class {i}' for i in range(10)], title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title(f'UMAP of Latent Space (Epoch {epoch + 1})')
    output_path = os.path.join(model_results_dir, f'epoch_{epoch + 1}_umap_latent_space.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def run_roc_analysis(ae, datamodule, model_results_dir, epoch):
    """Run ROC analysis for anomaly detection."""
    dataloader = datamodule.test_dataloader()
    all_labels, all_scores = [], []

    ae.eval()

    with torch.no_grad():
        for batch in dataloader:
            x, labels = batch
            x, labels = x.to(ae.device), labels.to(ae.device)
            condition = torch.nn.functional.one_hot(labels, num_classes=10).float().to(ae.device)

            if isinstance(ae, (CVAE, dCVAE)):
                reconstructions, _, _ = ae(x, condition)
            else:
                reconstructions, _, _ = ae(x)

            reconstructions = reconstructions.view_as(x)
            recon_loss = torch.mean((reconstructions - x) ** 2, dim=(1, 2, 3))
            all_labels.append(labels.cpu())
            all_scores.append(recon_loss.cpu())

    all_labels = torch.cat(all_labels, dim=0)
    all_scores = torch.cat(all_scores, dim=0)
    num_classes = 10
    all_labels_bin = label_binarize(all_labels.numpy(), classes=range(num_classes))

    plt.figure()
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(all_labels_bin[:, i], all_scores.numpy())
        roc_auc = roc_auc_score(all_labels_bin[:, i], all_scores.numpy())
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC - Epoch {epoch+1}')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(model_results_dir, f'epoch_{epoch+1}_roc_auc_multiclass.png'))
    plt.close()


def run_classification_error(ae, datamodule, model_results_dir, epoch):
    """Evaluate classification accuracy using latent representations."""
    dataloader = datamodule.test_dataloader()
    correct = 0
    total = 0

    classifier = nn.Linear(ae.latent_dim, 10)
    classifier.to(ae.device)

    with torch.no_grad():
        for batch in dataloader:
            x, labels = batch
            x, labels = x.to(ae.device), labels.to(ae.device)

            if isinstance(ae, (CVAE, dCVAE)):
                condition = torch.nn.functional.one_hot(labels, num_classes=10).float().to(ae.device)
                mu, log_var = ae.encode(x, condition)
            else:
                mu, log_var = ae.encode(x)

            z = ae.reparameterize(mu, log_var)
            logits = classifier(z)
            predicted = torch.argmax(logits, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    with open(os.path.join(model_results_dir, f'epoch_{epoch+1}_classification_accuracy.txt'), 'w') as f:
        f.write(f'Classification Accuracy: {accuracy * 100:.2f}%')


def plot_reconstruction(ae, dataloader, model_results_dir, epoch, model_type):
    """Plot original vs reconstructed images."""
    ae.eval()
    batch = next(iter(dataloader))
    x, labels = batch
    x = x.to(ae.device)
    labels = labels.to(ae.device)

    if isinstance(ae, (CVAE, dCVAE)):
        condition = torch.nn.functional.one_hot(labels, num_classes=10).float().to(ae.device)
        x_hat, _, _ = ae(x, condition)
    else:
        x_hat, _, _ = ae(x)

    image_shape = x.shape[1:]
    x_hat = x_hat.view(-1, *image_shape).cpu().detach().numpy()

    num_images = 25
    grid_size = 5

    plt.figure(figsize=(10, 10), facecolor='white')

    for i in range(num_images):
        ax = plt.subplot(grid_size, grid_size, i + 1)
        plt.subplots_adjust(left=0.1, bottom=1, right=0.9, top=1.5, wspace=0.4, hspace=0.8)
        plt.title(f'{model_type}', fontsize=10, fontweight='bold')

        img = x_hat[i].squeeze()
        plt.imshow(img, vmin=0, vmax=1, cmap=plt.cm.binary)

        ax.set_xticks([0, 10, 20])
        ax.set_yticks([0, 10, 20])
        ax.set_xticklabels([0, 10, 20], fontsize=10, fontweight='bold')
        ax.set_yticklabels([0, 10, 20], fontsize=10, fontweight='bold')

        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)

    plt.tight_layout()
    plt.savefig(os.path.join(model_results_dir, f'epoch_{epoch + 1}_reconstruction.png'), bbox_inches='tight')
    plt.close()