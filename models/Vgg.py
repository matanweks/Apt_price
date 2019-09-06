import torch.nn as nn
import torch
# from .utils import load_state_dict_from_url

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):
    def __init__(self, cfg, batch_norm=False, num_classes=10, init_weights=True, coreset_size=[4096, 4096]):  # 4096
        super(VGG, self).__init__()
        self.mode = 'train'
        self.cfg = cfg
        self.threshold = 0.01
        self.features = self._make_layer(cfg, batch_norm=batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, coreset_size[0]),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(coreset_size[0], coreset_size[1]),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(coreset_size[1], num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # x = self.CNNBlock(in_channels=x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _make_layer(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                mask1 = torch.ones(conv2d.weight.size())

                if self.mode == 'prune':
                    # mask1 = self.__prune__(conv2d, mask1, threshold=self.threshold)
                    conv2d.weight = nn.Parameter(conv2d.weight * mask1)

                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def __prune__(self, threshold, prune_classifier=0):

        # for i, param in model.parameters():
        weight_flag = 1
        j = 0
        cfg_flag = 0
        for i, param in enumerate(self.features.parameters()):
            if param is not None:
                if param.requires_grad:
                    # print(torch.median(param))
                    # print(torch.kthvalue(param, 90))
                    mask = torch.ones(param.size())
                    mask = torch.mul(torch.gt(torch.abs(param), threshold[i]).float(), mask.to(device))

                    if weight_flag:
                        self.features[j].weight = torch.nn.Parameter(param * mask)
                        weight_flag = 0
                    else:
                        self.features[j].bias = torch.nn.Parameter(param * mask)
                        weight_flag = 1
                        if self.cfg[cfg_flag+1] == 'M':
                            cfg_flag += 2
                            j = j + 3
                        else:
                            j += 2
                            cfg_flag += 1
        j = 0
        features_threshold = i+1
        if prune_classifier:
            for i, param in enumerate(self.classifier.parameters()):
                if param is not None:
                    if param.requires_grad:
                        mask = torch.ones(param.size())
                        mask = torch.mul(torch.gt(torch.abs(param), threshold[features_threshold+i]).float(), mask.to(device))
                        if weight_flag:
                            self.classifier[j].weight = torch.nn.Parameter(param * mask)
                            weight_flag = 0
                        else:
                            self.classifier[j].bias = torch.nn.Parameter(param * mask)
                            weight_flag = 1
                            j = j + 3


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, coreset_size=[4096, 4096], **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    # model = VGG(CNNBlock(in_channels=3, cfg=cfgs[cfg], batch_norm=batch_norm), **kwargs)
    model = VGG(cfgs[cfg], batch_norm=batch_norm, coreset_size=coreset_size, **kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls[arch],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    return model


def vgg11(pretrained=False, progress=True, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', 'A', False, pretrained, progress, **kwargs)


def vgg11_bn(pretrained=False, progress=True, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11_bn', 'A', True, pretrained, progress, **kwargs)


def vgg13(pretrained=False, progress=True, **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13', 'B', False, pretrained, progress, **kwargs)


def vgg13_bn(pretrained=False, progress=True, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13_bn', 'B', True, pretrained, progress, **kwargs)


def vgg16(pretrained=False, progress=True, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)


def vgg16_bn(pretrained=False, progress=True, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', True, pretrained, progress, **kwargs)


def vgg19(pretrained=False, progress=True, coreset_size=[4096, 4096],  **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19', 'E', False, pretrained, progress, coreset_size=coreset_size, **kwargs)


def vgg19_bn(pretrained=False, progress=True, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19_bn', 'E', True, pretrained, progress, **kwargs)
