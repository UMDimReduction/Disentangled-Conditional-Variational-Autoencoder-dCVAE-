import os

import pytorch_lightning.loggers as loggers
import torch

import data
import lightning


def build_datamodule(dataset=None, model_type=None, batch_size=32, anomaly=False):
    exclude = 1 if anomaly else None
    train_size = 550 if model_type == 'classification' else None
    dataset_constructor = _get_dataset_constructor(dataset)
    datamodule = dataset_constructor('../data',
                                     batch_size=batch_size,
                                     train_size=train_size,
                                     exclude=exclude)

    return datamodule


def _get_dataset_constructor(dataset):
    if dataset == 'mnist' or dataset is None:
        return data.MNISTDataModule
    elif dataset in data.AVAILABLE_DATASETS:
        return data.AVAILABLE_DATASETS[dataset]
    else:
        raise ValueError(f'The dataset {dataset} is not supported. Choose one of {data.AVAILABLE_DATASETS.keys()}')


def build_ae(model_type, input_shape, anomaly=False):
    latent_dim = 2 if anomaly else 20
    noise_ratio = 0.5 if model_type == 'conditional' else None
    encoder, decoder = _build_networks(model_type, input_shape, latent_dim)
    bottleneck = _build_bottleneck(model_type, latent_dim)
    ae = lightning.Autoencoder(encoder, bottleneck, decoder, lr=0.0001, noise_ratio=noise_ratio)

    return ae


def _build_networks(model_type, input_shape, latent_dim):
    enc_dim = dec_dim = latent_dim
    if model_type == 'vae' or model_type.startswith('betaVAE'):
        enc_dim *= 2

    num_layers = 3
    if model_type == 'ConditionalVAE':
        encoder = encoders.ReducedEncoder(input_shape, enc_dim)
        decoder = decoders.ReducedDecoder(dec_dim, input_shape)
    elif model_type == 'CorrelatedVAE':
        encoder = encoders.MergedEncoder(input_shape, num_layers, enc_dim)
        decoder = decoders.MergedDecoder(dec_dim, num_layers, input_shape)
    else:
        encoder = encoders.LayeredEncoder(input_shape, num_layers, enc_dim)
        decoder = decoders.LayeredDecoder(dec_dim, num_layers, input_shape)

    return encoder, decoder


def _build_bottleneck(model_type, latent_dim):
    if model_type == 'dcvae' or model_type == 'FactorVAE' or model_type == 'RFVAE':
        bottleneck = bottlenecks.CorrelatedBottleneck(latent_dim, num_categories=512)
    elif model_type == 'VAE':
        bottleneck = bottlenecks.VariationalBottleneck(latent_dim)
    elif model_type == 'betaVAE':
        bottleneck = bottlenecks.VariationalBottleneck(latent_dim, beta=2.0)
    elif model_type == 'cvae':
        bottleneck = bottlenecks.ConditionedBottleneck(latent_dim, sparsity=0.20)
    else:
        raise ValueError(f'Unknown model type {model_type}.')

    return bottleneck


def load_ae_from_checkpoint(model_type, input_shape, anomaly, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model = build_ae(model_type, input_shape, anomaly)
    model.load_state_dict(checkpoint['state_dict'])

    return model


def build_logger(model_type, dataset, task=None):
    log_dir = _get_log_dir(dataset)
    experiment_name = _get_experiment_name(model_type, task)
    logger = loggers.TensorBoardLogger(log_dir, experiment_name)

    return logger


def _get_log_dir(dataset):
    script_path = os.path.dirname(__file__)
    log_dir = os.path.normpath(os.path.join(script_path, '..', 'logs', dataset))

    return log_dir


def _get_experiment_name(model_type, task):
    task = task or 'general'
    experiment_name = f'{model_type}_{task}'

    return experiment_name
