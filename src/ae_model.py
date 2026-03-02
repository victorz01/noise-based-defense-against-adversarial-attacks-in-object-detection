import torch.nn as nn

class DenoisingAE(nn.Module):
    def __init__(self):
        super(DenoisingAE, self).__init__()
        # Encoder: 3 -> 64 -> 32
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        # Decoder: 32 -> 64 -> 3
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))