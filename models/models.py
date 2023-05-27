import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CNNColor(nn.Module):
    def __init__(self, input_size=128):
        super(CNNColor, self).__init__()

        # Apliquem ResNet18 per extreure les caracteriestiques de nivell mig
        resnet = models.resnet18(num_classes=365)
        resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze(1))
        self.ResNet18 = nn.Sequential(*list(resnet.children())[0:6])

        self.conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.ReLU()
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1)
        self.upsample3 = nn.Upsample(scale_factor=2)

    def forward(self, input):
        #  Extraiem les caracteristiques de les gray features
        gray_features = self.ResNet18(input)

        # Apliquem el Upsample per conseguir color
        # Pas 1
        x = self.conv1(gray_features)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.upsample1(x)
        # Pas 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        # Pas 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        # Pas 4
        x = self.upsample2(x)
        # Pas 5
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        # Pas 6
        x = self.conv5(x)
        # Pas 7
        output = self.upsample3(x)

        return output