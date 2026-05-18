import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from .ae_model import DenoisingAE
from .hgd_trainer import HGDTrainer
from .adversarial_detection import AdversarialObjectDetection
import random

#Dataset class to create patches form the clean and adversairal images 
class PatchDataset(Dataset):
    def __init__(self, data_list, patch_size=256):
        self.data_list = data_list
        self.crop = T.RandomCrop(patch_size)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        clean, adv = self.data_list[idx]
        
        if clean.dim() == 4: clean = clean.squeeze(0)
        if adv.dim() == 4: adv = adv.squeeze(0)
            
        combined = torch.cat([clean, adv], dim=0)
        cropped = self.crop(combined)
        
        clean_crop = cropped[:3, ...]
        adv_crop = cropped[3:, ...]
        
        return clean_crop, adv_crop

#Main function to train the denoiser with the adversarial examples
def train_denoiser(image_paths, target_class=1, epochs=10,
                   noise_std=15,
                   p_identity=0.34,
                   p_noisy_clean=0.33,
                   p_noisy_adv=0.33,
                   attack_iterations=5,
                   alpha=0.7,
                   batch_size=4,
                   patch_size=256):
    detector = AdversarialObjectDetection()
    ae = DenoisingAE()
    trainer = HGDTrainer(ae,detector)
    
    adv_cache = {}

    data_list = []
    print("Preparing dataset with adversarial examples...")
    for img_path in image_paths:
        clean = detector.load_image(img_path).to(detector.device)
        if img_path not in adv_cache:
            clean2, adv, _ = detector.generate_adversarial_patch(
                img_path, target_class, num_iterations=attack_iterations
            )
            adv_cache[img_path] = (clean2.cpu() if clean2 is not None else clean.cpu(), adv.cpu())
            
        data_list.append(adv_cache[img_path])
        
    dataset = PatchDataset(data_list, patch_size=patch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    for epoch in range(epochs):
        epoch_loss = 0
        p_loss_total = 0
        h_loss_total = 0
        steps = 0
        
        for clean_batch, adv_batch in dataloader:
            clean_batch = clean_batch.to(detector.device)
            adv_batch = adv_batch.to(detector.device)

            r = random.random()
            if r < p_identity + p_noisy_clean:
                input_img = clean_batch
                target_img = clean_batch
            else:
                input_img = adv_batch
                target_img = clean_batch

            loss, p_loss, h_loss = trainer.train_step_pair(
                target_clean_img=target_img,
                input_img=input_img,
                alpha=alpha,
            )
            epoch_loss += loss
            p_loss_total += p_loss
            h_loss_total += h_loss
            steps += 1
            
        if steps > 0 and (epoch+1) % 10 == 0:
             print(f"Epoch {epoch+1}/{epochs}, Total Loss: {epoch_loss/steps:.4f}, Pixel Loss: {p_loss_total/steps:.4f}, HGD Loss: {h_loss_total/steps:.4f}")

    torch.save(ae.state_dict(), "hgd_denoiser.pth")
    return ae