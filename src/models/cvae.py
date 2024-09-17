import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class CVAE(pl.LightningModule):
    def __init__(self, input_shape, latent_dim, condition_dim, lr=1e-3):
        super(CVAE, self).__init__()
        self.input_shape = input_shape
        self.input_dim = int(torch.prod(torch.tensor(input_shape)))
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.lr = lr

        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim + condition_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2 * latent_dim)  # mu and log_var for latent space
        )

        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.input_dim),
            nn.Sigmoid()
        )

    def encode(self, x, condition):
        # Combine input and condition
        condition = condition.view(-1, self.condition_dim)
        x = torch.cat((x.view(-1, self.input_dim), condition), dim=1)
        encoded = self.encoder(x)
        mu, log_var = torch.chunk(encoded, 2, dim=1)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z, condition):
        z = torch.cat((z, condition), dim=1)
        decoded = self.decoder(z)
        return decoded.view(-1, *self.input_shape)

    def forward(self, x, condition):
        mu, log_var = self.encode(x, condition)
        z = self.reparameterize(mu, log_var)
        return self.decode(z, condition), mu, log_var

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_one_hot = F.one_hot(y, num_classes=self.condition_dim).float()
        x_hat, mu, log_var = self(x, y_one_hot)
        recon_loss = F.binary_cross_entropy(x_hat.view(-1, self.input_dim), x.view(-1, self.input_dim), reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recon_loss + kl_loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_one_hot = F.one_hot(y, num_classes=self.condition_dim).float()
        x_hat, mu, log_var = self(x, y_one_hot)
        recon_loss = F.binary_cross_entropy(x_hat.view(-1, self.input_dim), x.view(-1, self.input_dim), reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        val_loss = recon_loss + kl_loss
        self.log('val_loss', val_loss)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
