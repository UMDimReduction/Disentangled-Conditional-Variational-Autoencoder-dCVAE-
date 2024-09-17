import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class BetaVAE(pl.LightningModule):
    def __init__(self, input_shape, latent_dim, beta=4, lr=1e-3):
        super(BetaVAE, self).__init__()
        self.input_shape = input_shape
        self.input_dim = int(torch.prod(torch.tensor(input_shape)))
        self.latent_dim = latent_dim
        self.beta = beta
        self.lr = lr

        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * latent_dim)  # mu and log_var for latent space
        )

        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = x.view(-1, self.input_dim)  # Flatten the input
        encoded = self.encoder(x)
        mu, log_var = torch.chunk(encoded, 2, dim=1)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z):
        decoded = self.decoder(z)
        return decoded.view(-1, *self.input_shape)  # Reshape back to original input shape

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, mu, log_var = self(x)
        recon_loss = F.binary_cross_entropy(x_hat.view(-1, self.input_dim), x.view(-1, self.input_dim), reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recon_loss + self.beta * kl_loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, mu, log_var = self(x)
        recon_loss = F.binary_cross_entropy(x_hat.view(-1, self.input_dim), x.view(-1, self.input_dim), reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        val_loss = recon_loss + self.beta * kl_loss
        self.log('val_loss', val_loss)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
