import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import Encoder, Decoder, Autoencoder
from data import train_loader, val_loader, test_loader
from utils import plot_images, plot_images_2
from tqdm import tqdm
import yaml

# ----- Hyperparameters -----
with open("config.yaml", 'r') as file:
    config = yaml.safe_load(file)

IMAGE_SIZE = config['dataset']['image_size']
CHANNELS = config['dataset']['channels']
VALIDATION_SPLIT = config['dataset']['validation_split']

EMBEDDING_DIM = config['model']['embedding_dim']
DROPOUT = config['model']['dropout_rate']

BATCH_SIZE = config['training']['batch_size']
EPOCHS = config['training']['epochs']
LEARNING_RATE = config['training']['learning_rate']

# ----- Datasets -----

test_images, _ = next(iter(test_loader))
plot_images(test_images)


# ----- Model -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = Encoder(IMAGE_SIZE, CHANNELS, DROPOUT, EMBEDDING_DIM)
decoder = Decoder(IMAGE_SIZE, CHANNELS, DROPOUT, EMBEDDING_DIM)

model = Autoencoder(encoder, decoder).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Started Training...")
# ----- Training Loop -----
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for images, _ in tqdm(train_loader, desc=f'TRAINING: Epoch {epoch+1}/{EPOCHS}'):
        images = images.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, images)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * images.size(0)
    
    train_loss /= len(train_loader.dataset)
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, _ in tqdm(val_loader, desc=f'VALIDATION: Epoch {epoch+1}/{EPOCHS}'):
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, images)
            val_loss += loss.item() * images.size(0)
    
    val_loss /= len(val_loader.dataset)
    
    print(f'Epoch [{epoch+1}/{EPOCHS}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

print("Training Finished. Saving Model...")


torch.save(model.state_dict(), 'encoder.pth')
torch.save(model.state_dict(), 'decoder.pth')
torch.save(model.state_dict(), 'autoencoder.pth')

test_output = model(test_images)
plot_images_2(test_images, test_output)
