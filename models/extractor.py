import torch
import torch.nn as nn
import torchvision.models as models
    
class VGG19FeatureExtractor(nn.Module):
    def __init__(self, freeze_until=28):
        super(VGG19FeatureExtractor, self).__init__()
        
        # Load VGG19 model with the most up-to-date pretrained weights
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.fc = nn.Sequential(*list(vgg.classifier.children())[:-1])  # Remove the last classification layer
        
        # Freeze layers up to `freeze_until`
        for param in self.features[:freeze_until].parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x