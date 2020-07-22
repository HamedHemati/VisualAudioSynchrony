import torch
import torch.nn as nn
import torchvision.models as models


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

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x