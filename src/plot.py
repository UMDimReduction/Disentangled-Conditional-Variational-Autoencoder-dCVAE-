import os
import torch
import umap.umap_ as umap
import torch.nn as nn
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks import Callback
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np
from models import dCVAE, CVAE

# Callback for saving results after each epoch
class SaveResultsCallback(Callback):
    def __init__(self, ae, datamodule, results_dir, model_type, dataset_name):
        super().__init__()
        self.ae = ae
        self.datamodule = datamodule
        self.results_dir = results_dir
        self.model_type = model_type
        self.dataset_name = dataset_name

        # Create a subfolder for each dataset and model type inside the results directory
        self.model_results_dir = os.path.join(results_dir, model_type, dataset_name)
        if not os.path.exists(self.model_results_dir):
            os.makedirs(self.model_results_dir)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        print(f"Saving results for Epoch {epoch+1}")

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

def handle_downstream_task(ae, datamodule, task, model_results_dir, epoch):
    dataloader = datamodule.test_dataloader()
    ae.eval()

    all_inputs, all_latents, all_labels = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            x, labels = batch
            x, labels = x.to(ae.device), labels.to(ae.device)
            condition = torch.nn.functional.one_hot(labels, num_classes=10).float().to(ae.device)

            if isinstance(ae, (CVAE, dCVAE)):  # Only pass condition if it's a CVAE or dCVAE model
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
    """ Generate and plot the latent manifold with axis intervals 0-255 """
    ae.eval()

    n = 10  # Grid size for latent space sampling (10x10 for 10 classes)
    digit_size = 32  # Assuming the decoder outputs 32x32 images
    figure = np.zeros((digit_size * n, digit_size * n))  # Updated to match 32x32 image sizes

    grid_x = np.linspace(-2, 2, n)
    grid_y = np.linspace(-2, 2, n)

    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.zeros((1, ae.latent_dim))  # latent_dim should match your model
            z_sample[0, 0] = xi
            z_sample[0, 1] = yi

            z_sample = torch.from_numpy(z_sample).float().to(ae.device)

            # Conditionally generate samples based on the model type
            if isinstance(ae, (CVAE, dCVAE)):
                condition = torch.eye(10)[i % 10].unsqueeze(0).to(ae.device)  # One-hot encode for class i
                with torch.no_grad():
                    x_decoded = ae.decode(z_sample, condition)  # Pass the condition for conditional models
            else:
                with torch.no_grad():
                    x_decoded = ae.decode(z_sample)  # No condition for standard VAE

            # Dynamically check the size of the output and reshape accordingly
            output_size = x_decoded.numel()  # Get the number of elements in x_decoded tensor
            digit_size = int(np.sqrt(output_size))  # Calculate the image size (28, 32, etc.)

            # Ensure the output size matches the expected image size
            if output_size != digit_size * digit_size:
                raise ValueError(f"Expected image size {digit_size}x{digit_size}, but got {output_size} elements")

            # Reshape the output to the correct 2D image size
            x_decoded = x_decoded.view(digit_size, digit_size).cpu().numpy()

            # Ensure the figure array has the correct shape for the output
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = x_decoded

    plt.figure(figsize=(10, 10), facecolor='white')  # Explicitly setting the facecolor to white
    plt.imshow(figure, vmin=0, vmax=1, cmap=plt.cm.binary)  # Ensure the colormap is 'gray' to get black and white images

    # Adjust the number of ticks to match the number of labels (n = 10)
    plt.xticks(np.arange(0, digit_size * n, step=digit_size), np.linspace(0, 255, n).astype(int))
    plt.yticks(np.arange(0, digit_size * n, step=digit_size), np.linspace(0, 255, n).astype(int))

    plt.gca().set_facecolor('white')  # Ensure the axes background is white
    plt.grid(False)  # Disable the grid

    plt.title(f'Latent Manifold (Epoch {epoch + 1})', color='black')
    output_path = os.path.join(model_results_dir, f'epoch_{epoch + 1}_latent_manifold.png')
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')  # Save with white background
    plt.close()

def plot_umap(latents, labels, model_results_dir, epoch):
    if latents is None or len(latents) == 0:
        print("No latents found, skipping UMAP plot.")
        return

    reducer = umap.UMAP()
    latent_reduced = reducer.fit_transform(latents)

    # Ensure the number of labels matches the number of latent representations
    if len(latent_reduced) != len(labels):
        print(f"Warning: Latent representations ({len(latent_reduced)}) do not match labels ({len(labels)}).")
        labels = labels[:len(latent_reduced)]

    # Find min and max values to normalize and scale the data
    min_vals = latent_reduced.min(axis=0)
    max_vals = latent_reduced.max(axis=0)
    latent_reduced_scaled = (latent_reduced - min_vals) / (max_vals - min_vals)  # Normalize between 0 and 1

    # Now scale to the [0, 0.4] range
    latent_reduced_scaled = latent_reduced_scaled * 0.4

    # Define colors for each class
    class_colors = {
        0: '#98accd',
        1: '#ea8024',
        2: '#4ca947',
        3: '#d12227',
        4: '#824098',
        5: '#a1766e',
        6: '#f5a0c5',
        7: '#d2cdcc',
        8: '#fce274',
        9: '#aeddf7'
    }

    # Assign colors based on labels
    colors = [class_colors[int(label)] for label in labels]

    plt.figure(facecolor='white')
    plt.scatter(latent_reduced_scaled[:, 0], latent_reduced_scaled[:, 1], c=colors)

    # Set axis limits and ticks for UMAP plot (zoomed out to fit everything nicely)
    plt.xlim(0, 0.4)
    plt.ylim(0, 0.4)
    plt.xticks([0.1, 0.2, 0.3, 0.4])
    plt.yticks([0.1, 0.2, 0.3, 0.4])

    # Create a custom legend for the classes
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=class_colors[i], markersize=8) for i in
               range(10)]
    plt.legend(handles, [f'Class {i}' for i in range(10)], title='Classes', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title(f'UMAP of Latent Space (Epoch {epoch + 1})')
    output_path = os.path.join(model_results_dir, f'epoch_{epoch + 1}_umap_latent_space.png')
    print(f"Saving UMAP plot at {output_path}")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def run_roc_analysis(ae, datamodule, model_results_dir, epoch):
    dataloader = datamodule.test_dataloader()
    all_labels, all_scores = [], []

    ae.eval()

    with torch.no_grad():
        for batch in dataloader:
            x, labels = batch
            x, labels = x.to(ae.device), labels.to(ae.device)
            condition = torch.nn.functional.one_hot(labels, num_classes=10).float().to(ae.device)
            reconstructions, _, _ = ae(x, condition) if isinstance(ae, (CVAE, dCVAE)) else ae(x)
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
    dataloader = datamodule.test_dataloader()
    correct = 0
    total = 0

    classifier = nn.Linear(ae.latent_dim, 10)
    classifier.to(ae.device)

    with torch.no_grad():
        for batch in dataloader:
            x, labels = batch
            x, labels = x.to(ae.device), labels.to(ae.device)

            # Pass the condition if the model is a CVAE or dCVAE
            if isinstance(ae, (CVAE, dCVAE)):
                condition = torch.nn.functional.one_hot(labels, num_classes=10).float().to(ae.device)
                mu, log_var = ae.encode(x, condition)  # Pass condition here
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
    ae.eval()
    batch = next(iter(dataloader))
    x, labels = batch
    x = x.to(ae.device)
    labels = labels.to(ae.device)

    # Pass the condition if the model is a CVAE or dCVAE
    if isinstance(ae, (CVAE, dCVAE)):
        condition = torch.nn.functional.one_hot(labels, num_classes=10).float().to(ae.device)
        x_hat, _, _ = ae(x, condition)
    else:
        x_hat, _, _ = ae(x)

    # Get the shape of the input images
    image_shape = x.shape[1:]

    # Reshape the images based on their actual shape
    x_hat = x_hat.view(-1, *image_shape).cpu().detach().numpy()

    num_images = 25  # Display a 5x5 grid of images
    grid_size = 5

    plt.figure(figsize=(10, 10), facecolor='white')

    for i in range(num_images):
        ax = plt.subplot(grid_size, grid_size, i + 1)

        # Custom layout and design incorporation
        plt.subplots_adjust(left=0.1, bottom=1, right=0.9, top=1.5, wspace=0.4, hspace=0.8)
        plt.title(f'{model_type}', fontsize=10, fontweight='bold')  # Larger and bold font

        # Plot the predicted/reconstructed image, removing the unnecessary dimension
        img = x_hat[i].squeeze()  # This will remove the extra dimension
        plt.imshow(img, vmin=0, vmax=1, cmap=plt.cm.binary)

        # Set axis intervals (0, 20) and make them bold
        ax.set_xticks([0, 10, 20])
        ax.set_yticks([0, 10, 20])
        ax.set_xticklabels([0, 10, 20], fontsize=10, fontweight='bold')
        ax.set_yticklabels([0, 10, 20], fontsize=10, fontweight='bold')

        # Show the spines (axis borders)
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)

    # Ensure tight layout
    plt.tight_layout()

    # Save the figure with the desired layout
    plt.savefig(os.path.join(model_results_dir, f'epoch_{epoch + 1}_reconstruction.png'), bbox_inches='tight')
    plt.close()
