import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

# return the tensor of our image
def open_image(path: str):
    image = Image.open(path).convert("RGB")
    transform = T.ToTensor()  
    tensor = transform(image)
    return tensor
 
# add values from a Normal (Gaussian) Distribution to each pixel in the image
@torch.no_grad()
def add_gaussian_noise(tensor, mean=0, std=0.1):
    noise = torch.randn_like(tensor) * std + mean
    noisy_tensor = torch.clamp(tensor + noise, 0.0, 1.0)
    return noisy_tensor


# add white(salt) and black(pepper) pixels to the image
@torch.no_grad()
def add_salt_and_pepper_noise(tensor, noise_ratio=0.02):
    noisy_tensor = tensor.clone()
    mask = torch.rand(tensor.shape[0], 1, tensor.shape[2], tensor.shape[3], device=tensor.device)

    salt_mask = (mask < (noise_ratio / 2))
    pepper_mask = (mask > 1 - (noise_ratio / 2))

    noisy_tensor[salt_mask.expand_as(tensor)] = 1.0
    noisy_tensor[pepper_mask.expand_as(tensor)] = 0.0

    return noisy_tensor

# add completely random noise to the image, given an intensity ceiling
@torch.no_grad()
def add_random_noise(tensor, intensity=0.1):
    noise = (torch.rand_like(tensor) * 2 - 1) * intensity  # (-intensity, +intensity)
    return torch.clamp(tensor + noise, 0.0, 1.0)




def testing():
    original_tensor = open_image("image.png").unsqueeze(0)
    if original_tensor is None:
        raise Exception("Image not loaded properly. Check the file path.")
 

    # TESTING NORMAL (GAUSSIAN) NOISE
    noisy_tensor = add_gaussian_noise(original_tensor, mean=0, std=0.1)
    to_pil = T.ToPILImage()
    
    plt.subplot(1,2,1)
    plt.title("Original")
    plt.imshow(to_pil(original_tensor.squeeze(0)))
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.title("Gaussian Noise")
    plt.imshow(to_pil(noisy_tensor.squeeze(0)))
    plt.axis('off')
    plt.show()


    # TESTING SALT AND PEPPER
    noisy_tensor = add_salt_and_pepper_noise(original_tensor, 0.02)
    to_pil = T.ToPILImage()
    
    plt.subplot(1,2,1)
    plt.title("Original")
    plt.imshow(to_pil(original_tensor.squeeze(0)))
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.title("Salt and Pepper")
    plt.imshow(to_pil(noisy_tensor.squeeze(0)))
    plt.axis('off')
    plt.show()

    # TESTING RANDOM NOISE
    noisy_tensor = add_random_noise(original_tensor, 0.1)
    to_pil = T.ToPILImage()
    
    plt.subplot(1,2,1)
    plt.title("Original")
    plt.imshow(to_pil(original_tensor.squeeze(0)))
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.title("Random Noise")
    plt.imshow(to_pil(noisy_tensor.squeeze(0)))
    plt.axis('off')
    plt.show()



testing()