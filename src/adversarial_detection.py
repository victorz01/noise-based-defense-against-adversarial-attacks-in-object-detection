import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import transforms 
from torchvision.models import detection 
from typing import Tuple
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models import get_model
from torchvision.models.detection import FasterRCNN
from PIL import Image

class AdversarialObjectDetection: 
    def __init__(self,name:str="fasterrcnn",device=None):
        self.name = name
        self.model = None
        self.patch = None 
        self.device = self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.learning_rate = 0.01 
        self.patch_scale = 0.1 
        self.confidence_threshold = 0.5

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self._load_model()

    def _load_model(self):
        if self.name == "fasterrcnn" or self.name == "fasterrcnn_resnet50":
            self.model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
        elif self.name == "fasterrcnn_mobilenet":
            self.model = detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
        else:
            raise ValueError(f"Model {self.name} no found")
        self.model.to(self.device)
        self.model.eval()
        print("Loading model")

    def load_image(self,image_path,target_size: Tuple[int,int] = (640,640)):
        image = Image.open(image_path).convert("RGB")
        image = image.resize(target_size)
        image_tensor = self.transform(image)
        return image_tensor 
    
    def initialize_patch(self,image_shape: Tuple[int,int,int]):
        initial_patch = torch.randn(image_shape, device=self.device)*0.01
        self.patch = nn.Parameter(initial_patch, requires_grad=True)
    
    def apply_patch(self,image:torch.Tensor):
        if self.patch is None:
            raise ValueError("Patch is not there")
        scaled_patch = self.patch * self.patch_scale 
        patched_image = (image+scaled_patch)
        patched_image = torch.clamp(patched_image,0,1)
        return patched_image
    
    def detect_objects(self,image:torch.Tensor):
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)
        with torch.no_grad():
            outputs = self.model(image)
        return outputs[0]
    
    def compute_targeted_loss(self,
                              class_logits,
                              target_class: int):
        if class_logits.size(0) ==0:
            return torch.tensor(0.,device=self.device,requires_grad=True)
        target = torch.full((class_logits.size(0),),target_class,dtype=torch.long,device=self.device)

        loss = F.cross_entropy(class_logits,target)

        return loss
    
    def train_step(self,image,target_class):
        patched_image = self.apply_patch(image)
        image_list = self.model.transform([patched_image])[0]
        features = self.model.backbone(image_list.tensors)
        proposals, _ = self.model.rpn(image_list, features)
        box_feats = self.model.roi_heads.box_roi_pool(features, proposals, image_list.image_sizes)
        box_feats = self.model.roi_heads.box_head(box_feats)
        class_logits, box_reg = self.model.roi_heads.box_predictor(box_feats)
        loss = self.compute_targeted_loss(class_logits, target_class)

        return loss
    

    def generate_adversarial_patch(self,
                               image_path,
                               target_class,
                               num_iterations=100,
                               eps=0.1):

        original_image = self.load_image(image_path).to(self.device)
        self.initialize_patch(original_image.shape)
        optimizer = optim.Adam([self.patch], lr=self.learning_rate)
        loss_history = []

        for i in range(num_iterations):
            optimizer.zero_grad()

            loss = self.train_step(original_image, target_class)

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                self.patch.data.clamp_(-eps, eps)

            loss_value = float(loss.detach().cpu())
            loss_history.append(loss_value)
            print(f"Iteration {i+1}: Loss = {loss_value:.3f}")
        with torch.no_grad():
            adversarial_image = self.apply_patch(original_image)
        return original_image, adversarial_image, loss_history
    
    
