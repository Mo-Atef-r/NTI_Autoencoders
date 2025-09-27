import matplotlib.pyplot as plt

def plot_images(images, n=10):
    """Plot n images from a batch of images."""
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        plt.imshow(images[i].cpu().numpy().squeeze(), cmap='gray')
        plt.axis('off')
    plt.show()
    

def plot_images_2(images, generated, n=10):
    """Plot n images from a batch of images."""
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        plt.imshow(images[i].cpu().numpy().squeeze(), cmap='gray')
        plt.axis('off')
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(generated[i].cpu().detach().numpy().squeeze(), cmap='gray')
        plt.axis('off')
    plt.show()