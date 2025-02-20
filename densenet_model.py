import torch.nn as nn
import torchvision.models as models

class DenseNetModel(nn.Module):
    def __init__(self):
        super(DenseNetModel, self).__init__()
        self.densenet = models.densenet121(pretrained=True)
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )
    
    def forward(self, x):
        return self.densenet(x)