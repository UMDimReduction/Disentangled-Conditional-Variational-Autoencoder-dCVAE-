import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from building import build_ae, build_datamodule
from models import dCVAE, CVAE
from plot import SaveResultsCallback  # Importing the plotting and callback logic

# Ensure the results directory exists
os.makedirs('results', exist_ok=True)

def run(model_type, dataset, batch_size, anomaly, downstream_task=None, epochs=1):
    # Create a folder structure based on model type and dataset
    model_results_dir = os.path.join('results', model_type, dataset)
    os.makedirs(model_results_dir, exist_ok=True)

    # Build the datamodule
    datamodule = build_datamodule(dataset=dataset, model_type=model_type, batch_size=batch_size, anomaly=anomaly)

    # Build the autoencoder
    ae = build_ae(model_type, datamodule.dims, anomaly)

    # Move the model to the appropriate device
    ae = ae.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Setup checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=os.path.join('checkpoints', model_type, dataset),
        filename=f'{model_type}-{{epoch:02d}}-{{val_loss:.2f}}',
        save_top_k=1,
        mode='min',
    )

    # SaveResults callback to store results after each epoch
    save_results_callback = SaveResultsCallback(ae, datamodule, model_results_dir, model_type, dataset)

    # Trainer configuration
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[0],
        max_epochs=epochs,
        logger=None,
        callbacks=[checkpoint_callback, save_results_callback]
    )

    # Train the model
    trainer.fit(ae, datamodule=datamodule)

    return checkpoint_callback.best_model_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('model_type', type=str, choices=['vae', 'dcvae', 'factorvae', 'rfvae', 'betavae', 'cvae'])
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--anomaly', action='store_true')
    parser.add_argument('--downstream_task', type=str, choices=['umap', 'roc', 'classification', 'reconstruction'], default=None)
    parser.add_argument('--epochs', type=int, default=10)

    opt = parser.parse_args()

    print(run(opt.model_type, opt.dataset, opt.batch_size, opt.anomaly, opt.downstream_task, opt.epochs))
