import os
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST, KMNIST, EMNIST, CIFAR10


class MNISTDerivativeDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = './', batch_size=32, train_size=None, exclude=None):
        """
        Base data module class for handling MNIST-like datasets.

        Args:
            data_dir: Directory to store/load data.
            batch_size: Batch size for training/testing.
            train_size: Size of the training set.
            exclude: Class to exclude (optional, for anomaly detection).
        """
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.Pad(2),
                                             transforms.ToTensor()])

        self.dims = (1, 32, 32)  # Image dimensions
        self.num_classes = 10 if exclude is None else 9
        self.batch_size = batch_size
        self.train_size = train_size
        self.exclude = exclude
        self.num_workers = 1

        self.mnist_train = None
        self.mnist_val = None
        self.mnist_test = None

    def _get_mnist(self, train, transform=None, download=False):
        """
        Method to be implemented in child classes to return the dataset.

        Args:
            train: Whether to load the training set.
            transform: Transformations to apply to the dataset.
            download: Whether to download the dataset.
        """
        raise NotImplementedError

    def prepare_data(self):
        """
        Download the MNIST dataset (if necessary).
        """
        self._get_mnist(train=True, download=True)
        self._get_mnist(train=False, download=True)

    def setup(self, stage=None):
        """
        Split the data into training, validation, and test sets.
        """
        if stage == 'fit' or stage is None:
            mnist_full = self._get_mnist(train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = self._split_train_val(mnist_full)

        if stage == 'test' or stage == 'fit' or stage is None:  # Add 'fit' here to ensure it gets initialized for testing
            self.mnist_test = self._get_mnist(train=False, transform=self.transform)

    def _split_train_val(self, mnist_full):
        """
        Split the dataset into training and validation sets, optionally excluding a class.
        """
        filter_mask = torch.zeros(len(mnist_full), dtype=torch.int)
        split_idx = torch.randperm(len(mnist_full), generator=torch.Generator().manual_seed(42))
        bootstrap_size = self.train_size if self.train_size is not None else 55000
        filter_mask.scatter_(0, split_idx[:bootstrap_size], 1)

        # Exclude a class (if needed for anomaly detection)
        if self.exclude is not None:
            filter_mask[mnist_full.targets == self.exclude] = 0

        mnist_train = Subset(mnist_full, filter_mask.nonzero(as_tuple=False).squeeze())
        mnist_val = Subset(mnist_full, split_idx[55000:])

        return mnist_train, mnist_val

    def train_dataloader(self):
        """
        Return the DataLoader for training.
        """
        return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                          persistent_workers=True)

    def val_dataloader(self):
        """
        Return the DataLoader for validation.
        """
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=self.num_workers,
                          persistent_workers=True)

    def test_dataloader(self):
        """
        Return the DataLoader for testing.
        """
        if self.mnist_test is None:
            raise ValueError("Test dataset not initialized properly.")

        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers,
                          persistent_workers=True)


# Specific Dataset Modules
class MNISTDataModule(MNISTDerivativeDataModule):
    def _get_mnist(self, train, transform=None, download=False):
        """
        Load MNIST dataset.
        """
        return MNIST(self.data_dir, train=train, transform=transform, download=download)


class FashionMNISTDataModule(MNISTDerivativeDataModule):
    def _get_mnist(self, train, transform=None, download=False):
        """
        Load FashionMNIST dataset.
        """
        return FashionMNIST(self.data_dir, train=train, transform=transform, download=download)


class KMNISTDataModule(MNISTDerivativeDataModule):
    def _get_mnist(self, train, transform=None, download=False):
        """
        Load KMNIST dataset.
        """
        return KMNIST(self.data_dir, train=train, transform=transform, download=download)


class EMNISTDataModule(MNISTDerivativeDataModule):
    def _get_mnist(self, train, split='byclass', transform=None, download=False):
        """
        Load EMNIST dataset.
        """
        return EMNIST(self.data_dir, split=split, train=train, transform=transform, download=download)


class CIFAR10DataModule(MNISTDerivativeDataModule):
    def _get_mnist(self, train, transform=None, download=False):
        """
        Load CIFAR10 dataset.
        """
        return CIFAR10(self.data_dir, train=train, transform=transform, download=download)


# Available Datasets Mapping
AVAILABLE_DATASETS = {
    'mnist': MNISTDataModule,
    'fmnist': FashionMNISTDataModule,
    'kmnist': KMNISTDataModule,
    'emnist': EMNISTDataModule,
    'cifar10': CIFAR10DataModule
}
