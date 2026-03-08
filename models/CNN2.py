import torch
from torch import nn

class CNN_Dropout3(nn.Module):

    def __init__(self, shape, num_classes, act_func):
        super().__init__()

        # Feature extractor
        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            act_func(),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            act_func(),
            nn.MaxPool2d(2),
            nn.Dropout(0.1)
        )

        with torch.no_grad():
            dummy_input = torch.zeros(shape)
            features_output = self.features(dummy_input)
            flattened_size = features_output.view(1, -1).size(1)

        # classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 128),
            act_func(),
            nn.Linear(128, 64),
            act_func(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x