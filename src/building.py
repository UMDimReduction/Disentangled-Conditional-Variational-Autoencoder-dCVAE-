import os
import torch
import pytorch_lightning as pl
import data
from models.vae import VAE
from models.dcvae import dCVAE
from models.cvae import CVAE
from models.factorvae import FactorVAE
from models.rfvae import RFVAE
from models.betavae import BetaVAE

# Define the available autoencoders
AUTOENCODERS = {
    'vae': VAE,
    'cvae': CVAE,
    'dcvae': dCVAE,
    'rfvae': RFVAE,
    'betavae': BetaVAE,
    'factorvae': FactorVAE
}

def build_datamodule(dataset=None, model_type=None, batch_size=32, anomaly=False):
    """
    Build the appropriate datamodule for loading data.
    Args:
        dataset: Name of the dataset (e.g., MNIST).
        model_type: Type of model (e.g., classification, anomaly).
        batch_size: Batch size for data loading.
        anomaly: Whether the data is for anomaly detection.
    Returns:
        datamodule: The appropriate datamodule for loading data.
    """
    exclude = 1 if anomaly else None
    train_size = 550 if model_type == 'classification' else None
    dataset_constructor = _get_dataset_constructor(dataset)
    datamodule = dataset_constructor('../data', batch_size=batch_size, train_size=train_size, exclude=exclude)
    return datamodule

def _get_dataset_constructor(dataset):
    """
    Get the appropriate dataset constructor.
    Args:
        dataset: Name of the dataset.
    Returns:
        Constructor for the dataset.
    """
    if dataset == 'mnist' or dataset is None:
        return data.MNISTDataModule
    elif dataset in data.AVAILABLE_DATASETS:
        return data.AVAILABLE_DATASETS[dataset]
    else:
        raise ValueError(f'The dataset {dataset} is not supported. Choose one of {list(data.AVAILABLE_DATASETS.keys())}')


def build_ae(model_type, input_shape, anomaly):
    latent_dim = 20  # Default latent dimension
    condition_dim = 10  # Example condition dimension (can be adjusted based on the dataset)

    # Map model_type to the correct class
    ae_class = {
        'vae': VAE,
        'dcvae': dCVAE,
        'cvae': CVAE,
        'factorvae': FactorVAE,
        'rfvae': RFVAE,
        'betavae': BetaVAE,
    }.get(model_type.lower())

    if ae_class is None:
        raise ValueError(f"Unknown model_type: {model_type}")

    if model_type == 'dcvae' or model_type == 'cvae':
        # Return the conditional model with condition_dim
        ae = ae_class(input_shape, latent_dim=latent_dim, condition_dim=condition_dim)
    else:
        # Return the regular autoencoder model
        ae = ae_class(input_shape, latent_dim=latent_dim)

    return ae


def load_ae_from_checkpoint(model_type, input_shape, anomaly, checkpoint_path):
    """
    Load an autoencoder model from a checkpoint.
    Args:
        model_type: Type of model (e.g., VAE, CVAE, etc.).
        input_shape: Shape of the input data.
        anomaly: Whether the model is used for anomaly detection.
        checkpoint_path: Path to the checkpoint file.
    Returns:
        model: The loaded autoencoder model.
    """
    checkpoint = torch.load(checkpoint_path)
    model = build_ae(model_type, input_shape, anomaly)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def build_logger(model_type, dataset, task=None):
    """
    Build a logger for tracking training with TensorBoard.
    Args:
        model_type: Type of model (e.g., VAE, CVAE).
        dataset: The dataset being used.
        task: Optional task name.
    Returns:
        logger: The TensorBoard logger.
    """
    log_dir = _get_log_dir(dataset)
    experiment_name = _get_experiment_name(model_type, task)
    logger = pl.loggers.TensorBoardLogger(log_dir, experiment_name)
    return logger

def _get_log_dir(dataset):
    """
    Get the directory for logging experiments.
    Args:
        dataset: Name of the dataset.
    Returns:
        log_dir: Directory path for logging.
    """
    script_path = os.path.dirname(__file__)
    log_dir = os.path.normpath(os.path.join(script_path, '..', 'logs', dataset))
    return log_dir

def _get_experiment_name(model_type, task):
    """
    Generate an experiment name for logging.
    Args:
        model_type: Type of model.
        task: Task name.
    Returns:
        experiment_name: Formatted experiment name.
    """
    task = task or 'general'
    experiment_name = f'{model_type}_{task}'
    return experiment_name
