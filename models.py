import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.transforms import transforms
from pretrainedmodels import resnet18


class SubBlock(nn.Module):
    def __init__(self, name, filter1, filter2):
        super(SubBlock, self).__init__()
        # prefix = f'{name}-'
        prefix = ''
        self.add_module(prefix + 'bn1', nn.BatchNorm2d(filter2))
        self.add_module(prefix + 'conv2d_1', nn.Conv2d(filter2, filter1, (1, 1)))
        self.add_module(prefix + 'bn2', nn.BatchNorm2d(filter1))
        self.add_module(prefix + 'conv2d_2', nn.Conv2d(filter1, filter1, (3, 3), padding=(1, 1)))
        self.add_module(prefix + 'bn3', nn.BatchNorm2d(filter1))
        self.add_module(prefix + 'conv2d_3', nn.Conv2d(filter1, filter2, (1, 1)))
        self.add_module(prefix + 'relu3', nn.ReLU())

    def forward(self, x):
        x = self.bn1(x)
        y = x
        y = self.conv2d_1(y)
        y = nn.ReLU()(y)  # Reduce the number of features to 'filter'
        y = self.bn2(y)
        y = self.conv2d_2(y)
        y = nn.ReLU()(y)  # Extend the feature field
        y = self.bn3(y)
        y = self.conv2d_3(y)  # no activation # Restore the number of original features
        y += x  # Add the bypass connection
        y = self.relu3(y)
        return y


class SiameseNet(nn.Module):
    def __init__(self, channel=1, branch_features=512):
        super(SiameseNet, self).__init__()
        self.channel = channel
        self.branch_features = branch_features
        self.conv2d_1 = nn.Conv2d(self.channel, 64, (9, 9), stride=2, padding=(4, 4))
        self.maxpool1 = nn.MaxPool2d((2, 2), stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2d_2 = nn.Conv2d(64, 64, (3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv2d_3 = nn.Conv2d(64, 64, (3, 3), padding=(1, 1))
        self.layer_1 = self._make_layer('layer_1', 64, 64, 128)
        self.layer_2 = self._make_layer('layer_2', 128, 64, 256)
        self.layer_3 = self._make_layer('layer_3', 256, 96, 384)
        self.layer_4 = self._make_layer('layer_4', 384, 128, 512)
        self.globmaxpool2d = nn.MaxPool2d(kernel_size=(6, 6))

        self.conv2d_head_1 = nn.Conv2d(1, 32, (4, 1), stride=1)
        self.conv2d_head_2 = nn.Conv2d(1, 1, (1, 32), stride=1)
        self.fc_head = nn.Linear(self.branch_features, 1, bias=True)

    def _make_layer(self, name, filter0, filter1, filter2):
        # prefix = f'{name}-'
        prefix = ''
        layers = OrderedDict([
            (prefix + 'maxpool', nn.MaxPool2d((2, 2), stride=2)),
            (prefix + '1-bn1', nn.BatchNorm2d(filter0)),
            (prefix + '2-conv2d_1', nn.Conv2d(filter0, filter2, (1, 1))),
            (prefix + '2-relu1', nn.ReLU())
        ])
        # layers = []
        # layers.append(nn.MaxPool2d((2, 2), stride=2))
        # layers.append(nn.BatchNorm2d(filter0))
        # layers.append(nn.Conv2d(filter0, filter2, (1, 1)))
        # layers.append(nn.ReLU())
        for i in range(3, 7):
            # layers.append(SubBlock(filter1, filter2))
            # layers[f'{name}-{i}'] = SubBlock(f'{name}-{i}', filter1, filter2)
            layers[f'{i}'] = SubBlock(f'{name}-{i}', filter1, filter2)

        return nn.Sequential(layers)

    def forward(self, data, mode='train'):
        ##############
        # BRANCH MODEL
        ##############
        if mode == 'branch':
            data = [data]

        res = []

        if mode == 'head':
            res = data
        else:
            for i in range(len(data)):
                x = self.conv2d_1(data[i])
                x = nn.ReLU()(x)

                x = self.maxpool1(x)  # 96x96x64
                x = self.bn1(x)
                x = self.conv2d_2(x)
                x = nn.ReLU()(x)
                x = self.bn2(x)
                x = self.conv2d_3(x)
                x = nn.ReLU()(x)

                x = self.layer_1(x)
                x = self.layer_2(x)
                x = self.layer_3(x)
                x = self.layer_4(x)

                x = self.globmaxpool2d(x)  # 512
                x = x.view((data[0].shape[0], self.branch_features))
                res.append(x)

        if mode == 'branch':
            return res[0]

        ############
        # HEAD MODEL
        ############

        x1 = transforms.Lambda(lambda x: x[0] * x[1])(res)
        x2 = transforms.Lambda(lambda x: x[0] + x[1])(res)
        x3 = transforms.Lambda(lambda x: torch.abs(x[0] - x[1]))(res)
        x4 = transforms.Lambda(lambda x: torch.pow(x, 2))(x3)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = x.view((res[0].shape[0], 1, 4, res[0].shape[1]))

        # Per feature NN with shared weight is implemented using CONV2D with appropriate stride.
        x = self.conv2d_head_1(x)
        x = nn.ReLU()(x)
        x = x.transpose(1, 2).transpose(2, 3)
        x = self.conv2d_head_2(x)
        x = x.view(x.size()[0], -1)

        # Weighted sum implemented as a Dense layer.
        x = self.fc_head(x)
        x = nn.Sigmoid()(x)

        return x


class SiameseNetVer2(nn.Module):
    def __init__(self, channel, input_size):
        super(SiameseNetVer2, self).__init__()
        self.channel = channel
        self.input_size = input_size
        self.features_size = 640 if input_size == 768 else 512

        self.conv2d_1 = nn.Conv2d(self.channel, 64, (9, 9), stride=2, padding=(4, 4))
        self.maxpool1 = nn.MaxPool2d((2, 2), stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2d_2 = nn.Conv2d(64, 64, (3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv2d_3 = nn.Conv2d(64, 64, (3, 3), padding=(1, 1))
        self.layer_1 = self._make_layer('layer_1', 64, 64, 128)
        self.layer_2 = self._make_layer('layer_2', 128, 64, 256)
        self.layer_3 = self._make_layer('layer_3', 256, 96, 384)
        self.layer_4 = self._make_layer('layer_4', 384, 128, 512)
        if self.input_size == 768:
            self.layer_5 = self._make_layer('layer_5', 512, 160, 640)
        self.globmaxpool2d = nn.MaxPool2d(kernel_size=(6, 6))

        self.conv2d_head_1 = nn.Conv2d(1, 32, (4, 1), stride=1)
        self.conv2d_head_2 = nn.Conv2d(1, 1, (1, 32), stride=1)
        self.fc_head = nn.Linear(self.features_size, 1, bias=True)

    def _make_layer(self, name, filter0, filter1, filter2):
        # prefix = f'{name}-'
        prefix = ''
        layers = OrderedDict([
            (prefix + 'maxpool', nn.MaxPool2d((2, 2), stride=2)),
            (prefix + '1-bn1', nn.BatchNorm2d(filter0)),
            (prefix + '2-conv2d_1', nn.Conv2d(filter0, filter2, (1, 1))),
            (prefix + '2-relu1', nn.ReLU())
        ])
        for i in range(3, 7):
            layers[f'{i}'] = SubBlock(f'{name}-{i}', filter1, filter2)

        return nn.Sequential(layers)

    def forward(self, data, mode='train'):
        ##############
        # BRANCH MODEL
        ##############
        if mode == 'branch':
            data = [data]

        res = []

        if mode == 'head':
            res = data
        else:
            for i in range(len(data)):
                x = self.conv2d_1(data[i])
                x = nn.ReLU()(x)

                x = self.maxpool1(x)  # 96x96x64
                x = self.bn1(x)
                x = self.conv2d_2(x)
                x = nn.ReLU()(x)
                x = self.bn2(x)
                x = self.conv2d_3(x)
                x = nn.ReLU()(x)

                x = self.layer_1(x)
                x = self.layer_2(x)
                x = self.layer_3(x)
                x = self.layer_4(x)
                if self.input_size == 768:
                    x = self.layer_5(x)

                x = self.globmaxpool2d(x)  # 512
                x = x.view((data[0].shape[0], self.features_size))
                res.append(x)

        if mode == 'branch':
            return res[0]

        ############
        # HEAD MODEL
        ############

        x1 = transforms.Lambda(lambda x: x[0] * x[1])(res)
        x2 = transforms.Lambda(lambda x: x[0] + x[1])(res)
        x3 = transforms.Lambda(lambda x: torch.abs(x[0] - x[1]))(res)
        x4 = transforms.Lambda(lambda x: torch.pow(x, 2))(x3)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = x.view((res[0].shape[0], 1, 4, res[0].shape[1]))

        # Per feature NN with shared weight is implemented using CONV2D with appropriate stride.
        x = self.conv2d_head_1(x)
        x = nn.ReLU()(x)
        x = x.transpose(1, 2).transpose(2, 3)
        x = self.conv2d_head_2(x)
        x = x.view(x.size()[0], -1)

        # Weighted sum implemented as a Dense layer.
        x = self.fc_head(x)
        x = nn.Sigmoid()(x)

        return x


class SiameseResNet18(nn.Module):
    def __init__(self, features):
        super(SiameseResNet18, self).__init__()
        self.branch_features = features
        self.branch = self._get_branch_net(channels=3, num_classes=5004)
        self.conv2d_head_1 = nn.Conv2d(1, 32, (4, 1), stride=1)
        self.conv2d_head_2 = nn.Conv2d(1, 1, (1, 32), stride=1)
        self.fc_head = nn.Linear(self.branch_features, 1, bias=True)

    def _get_branch_net(self, channels, num_classes):
        model = resnet18(pretrained="imagenet")
        model.global_pool = nn.AdaptiveAvgPool2d(1)
        model.conv1_7x7_s2 = nn.Conv2d(channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        model.last_linear = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )
        # print('Model architecture:')
        # print(model)
        # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # print(f'\n\n\n\nModel trainable parameters {total_params}')
        return model

    def forward(self, data, mode='train'):
        ##############
        # BRANCH MODEL
        ##############
        if mode == 'branch':
            data = [data]

        res = []

        if mode == 'head':
            res = data
        else:
            for i in range(len(data)):
                x = self.branch(data[i])
                x = x.view((data[0].shape[0], self.features_size))
                res.append(x)

        if mode == 'branch':
            return res[0]

        ############
        # HEAD MODEL
        ############

        x1 = transforms.Lambda(lambda x: x[0] * x[1])(res)
        x2 = transforms.Lambda(lambda x: x[0] + x[1])(res)
        x3 = transforms.Lambda(lambda x: torch.abs(x[0] - x[1]))(res)
        x4 = transforms.Lambda(lambda x: torch.pow(x, 2))(x3)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = x.view((res[0].shape[0], 1, 4, res[0].shape[1]))

        # Per feature NN with shared weight is implemented using CONV2D with appropriate stride.
        x = self.conv2d_head_1(x)
        x = nn.ReLU()(x)
        x = x.transpose(1, 2).transpose(2, 3)
        x = self.conv2d_head_2(x)
        x = x.view(x.size()[0], -1)

        # Weighted sum implemented as a Dense layer.
        x = self.fc_head(x)
        x = nn.Sigmoid()(x)

        return x
