from torch import nn
import torch
from torch.nn import functional as F
from torchinfo import summary

class CNN(nn.Module):
    def __init__(self, shape, num_classes, act_func):
        super(CNN, self).__init__()
        self.features = nn.Sequential(

            nn.Conv2d(in_channels = shape[1], out_channels = 32, kernel_size=3, padding="same"),
            nn.BatchNorm2d(32),
            act_func(),
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size=3, padding="same"),
            nn.BatchNorm2d(32),
            act_func(),            
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.2),

            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=3, padding="same"),
            nn.BatchNorm2d(64),
            act_func(),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=3, padding="same"),
            nn.BatchNorm2d(64),
            act_func(),            
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.15),
            )

        with torch.no_grad():
            dummy_input = torch.zeros(shape)
            features_output = self.features(dummy_input)
            flattened_size = features_output.view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(flattened_size, 512),
            nn.BatchNorm1d(512),
            act_func(),
            nn.Dropout(0.3),


            nn.Linear(512, num_classes)
            )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x