"""
Download Datasets for dCVAE
===========================

Run this script to download all datasets before training.

Usage:
    python download_data.py
    python download_data.py --dataset mnist
    python download_data.py --dataset all
"""

import os
import argparse

# Create data directory
os.makedirs('./data', exist_ok=True)
os.makedirs('../data', exist_ok=True)

print("=" * 60)
print("Downloading datasets for dCVAE")
print("=" * 60)


def download_mnist():
    print("\n[1/5] Downloading MNIST...")
    from torchvision.datasets import MNIST
    MNIST('./data', train=True, download=True)
    MNIST('./data', train=False, download=True)
    print("      ✓ MNIST complete")


def download_fashion_mnist():
    print("\n[2/5] Downloading Fashion-MNIST...")
    from torchvision.datasets import FashionMNIST
    FashionMNIST('./data', train=True, download=True)
    FashionMNIST('./data', train=False, download=True)
    print("      ✓ Fashion-MNIST complete")


def download_kmnist():
    print("\n[3/5] Downloading KMNIST...")
    from torchvision.datasets import KMNIST
    KMNIST('./data', train=True, download=True)
    KMNIST('./data', train=False, download=True)
    print("      ✓ KMNIST complete")


def download_emnist():
    print("\n[4/5] Downloading EMNIST...")
    from torchvision.datasets import EMNIST
    EMNIST('./data', split='byclass', train=True, download=True)
    EMNIST('./data', split='byclass', train=False, download=True)
    print("      ✓ EMNIST complete")


def download_cifar10():
    print("\n[5/5] Downloading CIFAR-10...")
    from torchvision.datasets import CIFAR10
    CIFAR10('./data', train=True, download=True)
    CIFAR10('./data', train=False, download=True)
    print("      ✓ CIFAR-10 complete")


def download_all():
    download_mnist()
    download_fashion_mnist()
    download_kmnist()
    download_emnist()
    download_cifar10()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download datasets')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['all', 'mnist', 'fmnist', 'kmnist', 'emnist', 'cifar10'],
                        help='Dataset to download (default: all)')
    args = parser.parse_args()

    if args.dataset == 'all':
        download_all()
    elif args.dataset == 'mnist':
        download_mnist()
    elif args.dataset == 'fmnist':
        download_fashion_mnist()
    elif args.dataset == 'kmnist':
        download_kmnist()
    elif args.dataset == 'emnist':
        download_emnist()
    elif args.dataset == 'cifar10':
        download_cifar10()

    print("\n" + "=" * 60)
    print("✓ Download complete!")
    print("=" * 60)
    print("\nData saved to: ./data/")
    print("\nYou can now run training:")
    print("  python run.py dcvae --dataset mnist --epochs 50 --beta 4.0")
    print("=" * 60)
