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

    def _to_batched(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is NCHW for the autoencoder."""
        if x.dim() == 3:
            return x.unsqueeze(0)
        return x

    def _from_batched_like(self, y: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        """Undo batching if `like` was unbatched."""
        if like.dim() == 3 and y.dim() == 4 and y.size(0) == 1:
            return y.squeeze(0)
        return y

    def  get_features(self,x):
        self.detector.eval()
        with torch.no_grad():
            images,_ = self.detector.transform([x[0] if x.dim() == 4 else x])
            features = self.detector.backbone(images.tensors)
            return features['0']
        
    def train_step_pair(self, target_clean_img, input_img, alpha=0.7, grad_clip=None):
        """
        Generic training step for the denoiser.

        The denoiser learns to map `input_img` -> `target_clean_img`.
        This supports:
        - identity: input=clean, target=clean (don't change clean)
        - denoise:  input=noisy_clean, target=clean
        - purify:   input=noisy_adv, target=clean
        """
        self.ae.train()
        self.optimizer.zero_grad()

        x = self._to_batched(input_img)
        y = self.ae(x)
        denoised = self._from_batched_like(y, input_img)

        with torch.no_grad():
            clean_transformed, _ = self.detector.transform([
                target_clean_img[0] if target_clean_img.dim() == 4 else target_clean_img
            ])
            clean_feats = self.detector.backbone(clean_transformed.tensors)['0']

        denoised_transformed, _ = self.detector.transform([
            denoised[0] if denoised.dim() == 4 else denoised
        ])
        denoised_feats = self.detector.backbone(denoised_transformed.tensors)['0']

        l_pixel = self.pixel_loss(denoised, target_clean_img)
        l_hgd = self.feature_loss(denoised_feats, clean_feats)
        total_loss = (1 - alpha) * l_pixel + alpha * l_hgd

        total_loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.ae.parameters(), grad_clip)
        self.optimizer.step()

        return total_loss.item(), l_pixel.item(), l_hgd.item()

    def train_step(self, clean_img, noisy_adv_img, alpha=0.7):
        """Backward-compatible wrapper (legacy name)."""
        return self.train_step_pair(target_clean_img=clean_img, input_img=noisy_adv_img, alpha=alpha)