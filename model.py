import torch
import torch.nn as nn
#from torch.utils import Dataset, DataLoader
#from torchvision import datasets, transforms


# ----- Hyperparameters -----
IMAGE_SIZE = 28
CHANNELS = 1
BATCH_SIZE = 100
BUFFER_SIZE = 1000
VALIDATION_SPLIT = 0.2
EMBEDDING_DIM = 2
EPOCHS = 3

class Encoder(nn.Module):
    def __init__(self, image_Size, n_channels, embedding_dim):
        super().__init__()
        self.wh = image_Size
        self.n_channels = n_channels
        self.embedding_dim = embedding_dim
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels = self.n_channels, out_channels = 32, kernel_size=3, stride=1, padding=1), #28x28x1 -> 28x28x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #32x28x28 -> 32x14x14
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=3, stride=1, padding=1), #32x14x14 -> 64x14x14
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #64x14x14 -> 64x7x7
            nn.Flatten(), #64x7x7 -> 3136
            nn.Linear(self.wh//4 * self.wh//4 * 64, 128), #3136 -> 128
            nn.ReLU(),
            nn.Linear(128, self.embedding_dim) #128 -> embedding_dim    
        )
        
    def forward(self, x):
        self.embedding = self.encoder(x)
        return self.embedding

class Decoder(nn.Module):
    def __init__(self, image_Size, n_channels, embedding_dim):
        super().__init__()
        self.wh = image_Size
        self.n_channels = n_channels
        self.embedding_dim = embedding_dim
        
        self.decoder = nn.Sequential(
            nn.Linear(self.embedding_dim, 128), #embedding_dim -> 128
            nn.ReLU(),
            nn.Linear(128, self.wh//4 * self.wh//4 * 64), #128 -> 3136
            nn.ReLU(),
            nn.Unflatten(1, (64, self.wh//4, self.wh//4)), #3136 -> 64x7x7
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2), #7x7x64 -> 14x14x32
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=self.n_channels, kernel_size=2, stride=2), #14x14x32 -> 28x28x1
            nn.Sigmoid() # To ensure output is between 0 and 1
        )
    def forward(self, x):
        self.recon = self.decoder(x) #reconstruction
        return self.recon

class Autoencoder(nn.Module):
    def __init__(self, image_Size, n_channels, embedding_dim):
        super().__init__()
        self.encoder = Encoder(image_Size, n_channels, embedding_dim)
        self.decoder = Decoder(image_Size, n_channels, embedding_dim)
        
    def forward(self, x):
        embedding = self.encoder(x)
        recon = self.decoder(embedding)
        return recon
    
if __name__ == "__main__":
    model = Autoencoder(IMAGE_SIZE, CHANNELS, EMBEDDING_DIM)
    print(model)
    sample_input = torch.randn((BATCH_SIZE, CHANNELS, IMAGE_SIZE, IMAGE_SIZE))
    sample_output = model(sample_input)
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {sample_output.shape}")