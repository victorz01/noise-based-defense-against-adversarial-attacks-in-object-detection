import torch 
from .ae_model import DenoisingAE
from .hgd_trainer import HGDTrainer
from .adversarial_detection import AdversarialObjectDetection
from .noise_functions import add_gaussian_noise

def train_denoiser(image_paths,target_class=1,epochs=10):
    detector = AdversarialObjectDetection()
    ae = DenoisingAE()
    trainer = HGDTrainer(ae,detector)

    for epoch in range(epochs):
        for img_path in image_paths:

            clean, adv, _ = detector.generate_adversarial_patch(img_path, target_class, num_iterations=10)
            noisy_adv = add_gaussian_noise(adv, mean=0, std=0.1) 
            loss, p_loss, h_loss = trainer.train_step(clean, noisy_adv)
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Total Loss: {loss:.4f}, Pixel Loss: {p_loss:.4f}, HGD Loss: {h_loss:.4f}")

    torch.save(ae.state_dict(), "hgd_denoiser.pth")
    return ae