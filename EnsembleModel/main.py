import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
# import timm
# from pytorchcv.model_provider import get_model as ptcv_get_model

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.extract0 = models.resnet50(pretrained=True)
        self.extract1 = models.vgg19(pretrained=True)
        self.extract2 = models.resnext101_32x8d(pretrained=True)
        self.adapt_avg_pool = nn.AdaptiveAvgPool1d(64)
    def norm(self, x):
        return (x - x.mean(dim=1, keepdim=True)) / (x.var(dim=1, keepdim=True) + 1e-12)
    def forward(self, x):
        x = transforms.functional.resize(x, size=[224, 224])
        x = x / 255.0
        x = transforms.functional.normalize(x,
                                            mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])

        x0 = self.norm(self.adapt_avg_pool(self.extract0(x)))
        x1 = self.norm(self.adapt_avg_pool(self.extract1(x)))
        x2 = self.norm(self.adapt_avg_pool(self.extract2(x)))
        return (x0 + x1 + x2).mean(dim=1)

model = Model()
model.eval()
saved_model = torch.jit.script(model)
saved_model.save('saved_model.pt')
