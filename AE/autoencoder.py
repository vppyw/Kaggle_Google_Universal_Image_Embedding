import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

class Encoder(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.cnn = models.resnet50()
        self.linear = nn.Sequential(
                    nn.BatchNorm1d(1000),
                    nn.ReLU(),
                    nn.Linear(1000, 64),
                   )

    def forward(self, x, norm=True):
        if norm:
            x = transforms.functional.resize(x, size=[224, 224])
            x = x / 255.0
            x = transforms.functional.normalize(x, 
                                            mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225])
        x = self.cnn(x)
        x = self.linear(x)
        return x

class Decoder(nn.Module):
    def __init__(self, out_dim=(224, 224)):
        super().__init__()
        self.dconv = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(7, 7)),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),
        ) 

    def forward(self, x):
        x = x.reshape(-1, 16, 2, 2)
        return self.dconv(x)

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x, latent=True):
        if latent:
            x = self.encoder(x)
            return x
        x = self.encoder(x, norm=False)
        x = self.decoder(x)
        return x
