import torch

def add_gaussian_noise(image, mean, std):
    std = std/255.0
    noise = torch.randn_like(image) * std + mean
    return torch.clamp(image + noise, 0, 1)

def add_salt_and_pepper_noise(image, density):
    noise_mask = torch.rand_like(image)
    salt = (noise_mask < density/2).float()
    pepper = (noise_mask > (1 - density/2)).float()
    noisy_image = image.clone()
    noisy_image[salt > 0] = 1
    noisy_image[pepper > 0] = 0
    return noisy_image

def add_speckle_noise(image, intensity):
    noise = torch.randn_like(image) * intensity
    noisy_image = image + image * noise
    return torch.clamp(noisy_image, 0, 1)

def add_poisson_noise(image, scale):
    scaled_image = image * 255.0 * scale
    noisy_image = torch.poisson(scaled_image) / (255.0 * scale)
    return torch.clamp(noisy_image, 0, 1)

def add_noise(noise_type, original_image, **noise_params):
    if noise_type == "gaussian":
        return add_gaussian_noise(image=original_image, 
                                mean=noise_params.get('mean', 0), 
                                std=noise_params.get('std', 0.1))
    elif noise_type == "salt_and_pepper":
        return add_salt_and_pepper_noise(image=original_image, 
                                       density=noise_params.get('density', 0.1))
    elif noise_type == "speckle":
        return add_speckle_noise(image=original_image, 
                               intensity=noise_params.get('intensity', 0.1))
    elif noise_type == "poisson":
        return add_poisson_noise(image=original_image, 
                               scale=noise_params.get('scale', 1.0))
    else:
        return original_image