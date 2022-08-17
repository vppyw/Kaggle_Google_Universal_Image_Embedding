import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms

class MyModel(nn.Module):
  def __init__(self):
    super().__init__()
    inception_model = models.inception_v3(pretrained=True)
    inception_model.fc = nn.Linear(2048, 64)
    self.feature_extractor = inception_model

  def forward(self, x):
    x = transforms.functional.resize(x,size=[224, 224])
    x = x/255.0
    x = transforms.functional.normalize(x, 
                                            mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225])
    return self.feature_extractor(x).logits

model = MyModel()
model.eval()
saved_model = torch.jit.script(model)
saved_model.save('saved_model.pt')
