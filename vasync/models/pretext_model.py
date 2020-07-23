import torch
import torch.nn as nn
import torchvision.models as models


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
    
        self.conv_layers = nn.Sequential(nn.Conv2d(3, 32, 7),
                                         nn.BatchNorm2d(32),
                                         nn.MaxPool2d(4, 4),
                                         nn.Conv2d(32, 64, 5),
                                         nn.BatchNorm2d(64),
                                         nn.MaxPool2d(4, 4),
                                         nn.Conv2d(64, 64, 3),
                                         nn.BatchNorm2d(64),
                                         nn.MaxPool2d(2, 2),) # B x 64 x 5 x 5)
        self.linear = nn.Linear(64 * 5 * 5, 128)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class ClassifierPretextModel(nn.Module):
    def __init__(self, pretext_base_model, num_cls):
        super(ClassifierPretextModel, self).__init__()
        if pretext_base_model == "resnet50":
            base_model = models.resnet50(pretrained=False)
            self.base_model = nn.Sequential(*list(base_model.children())[:-1])
            self.classifier = nn.Linear(2048, num_cls)
        elif pretext_base_model == "resnet18":
            base_model = models.resnet18(pretrained=False)
            self.base_model = nn.Sequential(*list(base_model.children())[:-1])
            self.classifier = nn.Linear(512, num_cls)
        elif pretext_base_model == "simple_model":
            self.base_model = SimpleModel()
            self.classifier = nn.Linear(128, num_cls)

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x