# Disentangled Conditional Variational Autoencoder (dCVAE) for Unsupervised Anomaly Detection


Recently, generative models have shown promising performance in anomaly detection tasks. Specifically, autoencoders learn representations of high-dimensional data, and their reconstruction ability can be used to assess whether a new instance is likely to be anomalous. However, the primary challenge of unsupervised anomaly detection (UAD) is in learning appropriate disentangled features and avoiding information loss, while incorporating known sources of variation to improve the reconstruction. In this paper, we propose a novel architecture of generative autoencoder by combining the frameworks of $\beta$-VAE, conditional variational autoencoder (CVAE), and the principle of total correlation (TC). We show that our architecture improves the disentanglement of latent features, optimizes TC loss more efficiently, and improves the ability to detect anomalies in an unsupervised manner with respect to high-dimensional instances, such as in imaging datasets. Through both qualitative and quantitative experiments on several benchmark datasets, we demonstrate that our proposed method excels in terms of both anomaly detection and capturing disentangled features. Our analysis underlines the importance of learning disentangled features for UAD tasks.


## Installation
```python
   pip install -r requirements.txt
```

## Usage

```python
   cd ./dcvae
```
Run:
```python
   python main.py --model <model> --task <dataset> --beta 2.0 --num_epochs 100  --batch_size 64 [--gpu]
```

## Results

### Reconstructions

![Reconstructions](https://github.com/UMDimReduction/Disentangled-Conditional-Variational-Autoencoder-dCVAE-/tree/main/dcvae/results/Figures/reconstruction.png?raw=true)

### Latent Representation (MNIST)

![Latent Representation (MNIST)](https://github.com/UMDimReduction/Disentangled-Conditional-Variational-Autoencoder-dCVAE-/tree/main/dcvae/results/Figures/Figures/Latent_MNIST.png?raw=true)

### Latent Representation (FMNIST)

![Latent Representation (FMNIST)](https://github.com/UMDimReduction/Disentangled-Conditional-Variational-Autoencoder-dCVAE-/tree/main/dcvae/results/Figures/Latent_FMNIST.png?raw=true)

### Latent Representation (EMNIST)

![Latent Representation (EMNIST)](https://github.com/UMDimReduction/Disentangled-Conditional-Variational-Autoencoder-dCVAE-/tree/main/dcvae/results/Figures/Latent_EMNIST.png?raw=true)

### Latent Representation (KMNIST)

![Latent Representation (KMNIST)](https://github.com/UMDimReduction/Disentangled-Conditional-Variational-Autoencoder-dCVAE-/tree/main/dcvae/results/FiguresLatent_KMNIST.png?raw=true)


## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)