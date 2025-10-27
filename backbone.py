import torch
import torch.nn as nn
import torch.nn.functional as F

from cvtools.models.pytorch import PyTorchModel, L2Norm

class ArcLayer(nn.Module):

    def __init__(self, embedding_size, n_classes):
        super().__init__()

        self.weights = nn.Parameter(torch.randn(embedding_size, n_classes))

        nn.init.kaiming_normal_(self.weights)


    def forward(self, x):
        weights = F.normalize(self.weights, p=2, dim=0)

        return torch.mm(x, weights)


class BaseCNNModel(PyTorchModel):
    
    def __init__(self, embedding_dim, n_classes, classifier="linear"):
        super().__init__()

        self.outputs = {}
        self.features: nn.Sequential

        if classifier == "linear":
            self.classifier = nn.Linear(embedding_dim, n_classes)
        elif classifier == "arcface":
            self.classifier = ArcLayer(embedding_dim, n_classes)


    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x


    def _init_weights(self, submodule=None):

        if submodule is None:
            submodule = self

        for name, module in submodule.named_modules():

            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


    def _register_hooks(self, layers):
        
        def get_hook(name):
            def hook(module, input, output):
                self.outputs[name] = output
            return hook
        
        for layer in layers:
            getattr(self, layer).register_forward_hook(get_hook(layer))


    def freeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = False


    def unfreeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = True

class ConvGist(BaseCNNModel):

    def __init__(self, in_channels, n_classes, classifier="linear", output_layers=[]):
        super().__init__(embedding_dim=512, n_classes=n_classes, classifier=classifier)

        self.features = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, 128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        if classifier == "arcface":
            self.features.append(L2Norm())

        self._init_weights()
        self._register_hooks(output_layers)