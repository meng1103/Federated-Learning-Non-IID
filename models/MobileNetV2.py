import torch
import torch.nn as nn

import torch.nn.functional as F



class LinearBottleNeck(nn.Module):
    def __init__(self, in_c, out_c, s, t):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_c, in_c * t, 1),
            nn.BatchNorm2d(in_c * t),
            nn.ReLU6(inplace=True),

            # Depthwise
            nn.Conv2d(in_c * t, in_c * t, 3, stride=s, padding=1, groups=in_c * t),
            nn.BatchNorm2d(in_c * t),
            nn.ReLU6(inplace=True),

            # Pointwise
            nn.Conv2d(in_c * t, in_c * t, 1, stride=1, padding=0, groups=1),
            nn.BatchNorm2d(in_c * t),

            nn.Conv2d(in_c * t, out_c, 1),
            nn.BatchNorm2d(out_c)
        )

        self.stride = s
        self.in_channels = in_c
        self.out_channels = out_c

    def forward(self, x):
        # print('before shortcut x shape is', x.shape)
        residual = self.residual(x)
        # ('before shortcut residual shape is', residual.shape)
        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x
        # print('BottleNeck output shape is:', residual.shape)
        return residual


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=19):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.BatchNorm2d(32),

            nn.ReLU6(inplace=True),
        )


        # Bottleneck repeat, inputs_channel, outputs_channel , s, t.
        self.stage1 = LinearBottleNeck(32, 16, 1, 1)
        self.stage2 = self.make_stage(2, 16, 24, 2, 6)
        self.stage3 = self.make_stage(3, 24, 32, 2, 6)
        self.stage4 = self.make_stage(4, 32, 64, 2, 6)
        self.stage5 = self.make_stage(3, 64, 96, 1, 6)
        self.stage6 = self.make_stage(3, 96, 160, 2, 6)
        self.stage7 = LinearBottleNeck(160, 320, 1, 6)

        self.conv1 = nn.Sequential(
            nn.Conv2d(320, 1280, 1),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )

        self.conv2 = nn.Conv2d(1280, num_classes, 1)


    def forward(self, x):

        x = self.pre(x)

        x = self.stage1(x)

        x = self.stage2(x)

        x = self.stage3(x)

        x = self.stage4(x)

        x = self.stage5(x)

        x = self.stage6(x)

        x = self.stage7(x)

        x = self.conv1(x)

        x = F.adaptive_avg_pool2d(x, 1)

        x = self.conv2(x)

        x = x.view(x.size(0), -1)

        return x

    def make_stage(self, repeat, in_c, out_c, s, t):
        layers = []
        layers.append(LinearBottleNeck(in_c, out_c, s, t))

        while repeat - 1:
            layers.append(LinearBottleNeck(out_c, out_c, 1, t))
            repeat -= 1

        return nn.Sequential(*layers)

class MOONMobileNetV2(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )


        # Bottleneck repeat, inputs_channel, outputs_channel , s, t.
        self.stage1 = LinearBottleNeck(32, 16, 1, 1)
        self.stage2 = self.make_stage(2, 16, 24, 2, 6)
        self.stage3 = self.make_stage(3, 24, 32, 2, 6)
        self.stage4 = self.make_stage(4, 32, 64, 2, 6)
        self.stage5 = self.make_stage(3, 64, 96, 1, 6)
        self.stage6 = self.make_stage(3, 96, 160, 2, 6)
        self.stage7 = LinearBottleNeck(160, 320, 1, 6)

        self.conv1 = nn.Sequential(
            nn.Conv2d(320, 1280, 1),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )

        # self.conv2 = nn.Conv2d(1280, num_classes, 1)

        self.l1 = nn.Linear(1280, 1280)
        self.l2 = nn.Linear(1280, 256)
        self.l3 = nn.Linear(256, 100)


    def forward(self, x):

        x = self.pre(x)

        x = self.stage1(x)

        x = self.stage2(x)

        x = self.stage3(x)

        x = self.stage4(x)

        x = self.stage5(x)

        x = self.stage6(x)

        x = self.stage7(x)

        x = self.conv1(x)

        x = F.adaptive_avg_pool2d(x, 1)

        # x = x.view(x.size(0), -1)
        x = x.view(-1, 1280)
        # print('x.shape', x.shape)
        h = x.squeeze()

        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)

        y = self.l3(x)

        if len(h.shape) == 1:
            h = torch.unsqueeze(h, dim=0)
        if len(x.shape) == 1:
            x = torch.unsqueeze(x, dim=0)
        if len(y.shape) == 1:
            y = torch.unsqueeze(y, dim=0)
        return h, x, y



    def make_stage(self, repeat, in_c, out_c, s, t):
        layers = []
        layers.append(LinearBottleNeck(in_c, out_c, s, t))

        while repeat - 1:
            layers.append(LinearBottleNeck(out_c, out_c, 1, t))
            repeat -= 1

        return nn.Sequential(*layers)



