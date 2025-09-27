import torch
import torch.nn as nn
import yaml


# ----- Hyperparameters -----
with open("config.yaml", 'r') as file:
    config = yaml.safe_load(file)
    

IMAGE_SIZE = config['dataset']['image_size']
CHANNELS = config['dataset']['channels']

EMBEDDING_DIM = config['model']['embedding_dim']
DROPOUT = config['model']['dropout_rate']

class Encoder(nn.Module):
    def __init__(self, image_Size, n_channels, dropout_rate, embedding_dim):
        super().__init__()
        self.wh = image_Size
        self.n_channels = n_channels
        self.dropout_rate = dropout_rate
        self.embedding_dim = embedding_dim
        
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels = self.n_channels, out_channels = 32, kernel_size=3, stride=1, padding=1), #28x28x1 -> 28x28x32
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.MaxPool2d(kernel_size=2, stride=2), #32x28x28 -> 32x14x14
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=3, stride=1, padding=1), #32x14x14 -> 64x14x14
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.MaxPool2d(kernel_size=2, stride=2), #64x14x14 -> 64x7x7
            nn.Flatten(), #64x7x7 -> 3136
            nn.Linear(self.wh//4 * self.wh//4 * 64, 128), #3136 -> 128
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate)
        )

        # Layers to output mean and log variance
        self.fc_mean = nn.Linear(128, self.embedding_dim)
        self.fc_log_var = nn.Linear(128, self.embedding_dim)
        
    def forward(self, x):
        x = self.encoder_conv(x)
        z_mean = self.fc_mean(x)
        z_log_var = self.fc_log_var(x)
        return z_mean, z_log_var

class Sampling(nn.Module):
    """Uses the reparameterization trick to sample from a normal distribution."""
    def forward(self, z_mean, z_log_var):
        # Epsilon is sampled from a standard normal distribution
        epsilon = torch.randn_like(z_mean) 
        # The formula for sampling z
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon

class Decoder(nn.Module):
    def __init__(self, image_Size, n_channels, dropout_rate, embedding_dim):
        super().__init__()
        self.wh = image_Size
        self.n_channels = n_channels
        self.dropout_rate = dropout_rate
        self.embedding_dim = embedding_dim
        
        self.decoder = nn.Sequential(
            nn.Linear(self.embedding_dim, 128), #embedding_dim -> 128
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(128, self.wh//4 * self.wh//4 * 64), #128 -> 3136
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.Unflatten(1, (64, self.wh//4, self.wh//4)), #3136 -> 64x7x7
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2), #7x7x64 -> 14x14x32
            nn.ReLU(),
            nn.Dropout(p=self.dropout_rate),
            nn.ConvTranspose2d(in_channels=32, out_channels=self.n_channels, kernel_size=2, stride=2), #14x14x32 -> 28x28x1
            nn.Sigmoid() # To ensure output is between 0 and 1
        )
    def forward(self, x):
        self.recon = self.decoder(x) #reconstruction
        return self.recon

class VariationalAutoencoder(nn.Module):
    def __init__(self, enc, dec):
        super().__init__()
        self.encoder = enc
        self.decoder = dec
        self.sampling = Sampling()
        
    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = self.sampling(z_mean, z_log_var)
        recon = self.decoder(z)
        return recon, z_mean, z_log_var
    
if __name__ == "__main__":
    encoder = Encoder(IMAGE_SIZE, CHANNELS, DROPOUT, EMBEDDING_DIM)
    decoder = Decoder(IMAGE_SIZE, CHANNELS, DROPOUT, EMBEDDING_DIM)
    model = VariationalAutoencoder(encoder, decoder)
    print(model)
    
    sample_input = torch.randn((32, CHANNELS, IMAGE_SIZE, IMAGE_SIZE))
    recon_output, z_mean, z_log_var = model(sample_input)
    
    print(f"\nInput shape: {sample_input.shape}")
    print(f"Reconstruction shape: {recon_output.shape}")
    print(f"Z_mean shape: {z_mean.shape}")
    print(f"Z_log_var shape: {z_log_var.shape}")