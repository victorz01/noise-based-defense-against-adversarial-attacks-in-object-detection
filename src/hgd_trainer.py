import torch 
import torch.nn as nn
from .adversarial_detection import AdversarialObjectDetection

class HGDTrainer:
    def __init__(self,ae,detector_wrapper:AdversarialObjectDetection,lr=1e-4):
        self.ae = ae.to(detector_wrapper.device)
        self.detector= detector_wrapper.model 
        self.device = detector_wrapper.device
        self.optimizer = torch.optim.Adam(self.ae.parameters(), lr=lr)

        self.pixel_loss = nn.MSELoss()
        self.feature_loss = nn.L1Loss()

    def  get_features(self,x):
        self.detector.eval()
        with torch.no_grad():
            images,_ = self.detector.transform([x[0] if x.dim() == 4 else x])
            features = self.detector.backbone(images.tensors)
            return features['0']
        
    def train_step(self, clean_img, noisy_adv_img, alpha=0.7):
        self.ae.train() 
        self.optimizer.zero_grad()
        denoised = self.ae(noisy_adv_img) 

        with torch.no_grad():

            clean_transformed, _ = self.detector.transform([clean_img[0] if clean_img.dim()==4 else clean_img])
            clean_feats = self.detector.backbone(clean_transformed.tensors)['0']

    
        denoised_transformed, _ = self.detector.transform([denoised[0] if denoised.dim()==4 else denoised])
        denoised_feats = self.detector.backbone(denoised_transformed.tensors)['0']


        l_pixel = self.pixel_loss(denoised, clean_img)
        l_hgd = self.feature_loss(denoised_feats, clean_feats)
        total_loss = (1 - alpha) * l_pixel + alpha * l_hgd


        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), l_pixel.item(), l_hgd.item()