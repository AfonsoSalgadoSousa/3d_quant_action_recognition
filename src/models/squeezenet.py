'''SqueezeNet in PyTorch.
See the paper "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size" for more details.
'''

import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.quantization import DeQuantStub, QuantStub, fuse_modules

__all__ = ['SqueezeNet', 'squeezenet1_0', 'squeezenet1_1']


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes,
                 use_bypass=False, quantize=False):
        super(Fire, self).__init__()
        self.use_bypass = use_bypass
        self.quantize = quantize
        # self.q_add = nn.quantized.FloatFunctional()
        self.inplanes = inplanes
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU(inplace=False)
        self.squeeze = nn.Conv3d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_bn = nn.BatchNorm3d(squeeze_planes)
        self.expand1x1 = nn.Conv3d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_bn = nn.BatchNorm3d(expand1x1_planes)
        self.expand3x3 = nn.Conv3d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_bn = nn.BatchNorm3d(expand3x3_planes)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.squeeze_bn(out)
        out = self.relu(out)

        out1 = self.expand1x1(out)
        out1 = self.expand1x1_bn(out1)

        out2 = self.expand3x3(out)
        out2 = self.expand3x3_bn(out2)

        out = torch.cat([out1, out2], 1)
        """
        if self.use_bypass:
            if self.quantize:
                out = self.q_add.add(out, x)
            else:
                out += x
        out = self.relu(out)
        """
        return out

    def fuse_model(self):
        fuse_modules(self, [['squeeze', 'squeeze_bn', 'relu'],
                            ['expand1x1', 'expand1x1_bn'],
                            ['expand3x3', 'expand3x3_bn']], inplace=True)


class SqueezeNet(nn.Module):

    def __init__(self,
                 sample_size,
                 sample_duration,
                 version=1.1,
                 num_classes=600,
                 quantize=False):
        super(SqueezeNet, self).__init__()
        if version not in [1.0, 1.1]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 expected".format(version=version))
        self.num_classes = num_classes
        self.quantize = quantize
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        if version == 1.0:
            self.features = nn.Sequential(
                nn.Conv3d(3, 96, kernel_size=7, stride=(
                    1, 2, 2), padding=(3, 3, 3)),
                nn.BatchNorm3d(96),
                # nn.ReLU(inplace=True),
                nn.ReLU(inplace=False),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64, use_bypass=True),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128, use_bypass=True),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192, use_bypass=True),
                Fire(384, 64, 256, 256),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                Fire(512, 64, 256, 256, use_bypass=True),
            )
        if version == 1.1:
            self.features = nn.Sequential(
                nn.Conv3d(3, 64, kernel_size=3, stride=(
                    1, 2, 2), padding=(1, 1, 1)),
                nn.BatchNorm3d(64),
                # nn.ReLU(inplace=True),
                nn.ReLU(inplace=False),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64, use_bypass=True),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128, use_bypass=True),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192, use_bypass=True),
                nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256, use_bypass=True),
            )
        if self.quantize:
            self.quant = QuantStub()
            self.dequant = DeQuantStub()
        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv3d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            # nn.ReLU(inplace=True),
            nn.ReLU(inplace=False),
            nn.AvgPool3d((last_duration, last_size, last_size), stride=1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        if self.quantize:
            # run Pooling operations and dropout in FP32
            x = self.quant(x)
            # x = self.features(x)
            for m in self.features:
                if isinstance(m, nn.MaxPool3d):
                    x = self.dequant(x)
                    x = m(x)
                    x = self.quant(x)
                else:
                    x = m(x)
            """
            for c in self.classifier:
                if isinstance(c, nn.AvgPool3d) or isinstance(c, nn.Dropout):
                    x = self.dequant(x)
                    x = c(x)
                    x = self.quant(x)
                else:
                    x = c(x)
            """
            
            x = self.dequant(x)
            x = self.classifier[0](x)
            x = self.quant(x)
            x = self.classifier[1](x)
            x = self.classifier[2](x)
            x = self.dequant(x)
            x = self.classifier[-1](x)

            x = x.view(x.size(0), -1)
            # x = self.dequant(x)
            return x
        else:
            x = self.features(x)
            x = self.classifier(x)
            return x.view(x.size(0), -1)

    def fuse_model(self):
        # features 0,1,2 are respectively first conv bn and relu
        fuse_modules(self, [['features.0', 'features.1', 'features.2']], inplace=True)
        for m in self.modules():
            if type(m) == Fire:
                m.fuse_model()

def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()

    elif ft_portion == "last_layer":
        ft_module_names = []
        ft_module_names.append('classifier')

        parameters = []
        for k, v in model.named_parameters():
            for ft_module in ft_module_names:
                if ft_module in k:
                    parameters.append({'params': v})
                    break
            else:
                parameters.append({'params': v, 'lr': 0.0})
        return parameters

    else:
        raise ValueError(
            "Unsupported ft_portion: 'complete' or 'last_layer' expected")


def get_model(**kwargs):
    """
    Returns the model.
    """
    model = SqueezeNet(**kwargs)
    return model


if __name__ == '__main__':
    model = SqueezeNet(version=1.1, sample_size=112,
                       sample_duration=16, num_classes=600)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)
    print(model)

    input_var = Variable(torch.randn(8, 3, 16, 112, 112))
    output = model(input_var)
    print(output.shape)
