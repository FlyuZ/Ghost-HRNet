import torch
from torch import nn
import math
import timm
from torch.functional import Tensor
import torchinfo
BN_MOMENTUM = 0.1

class Conv(nn.Module):
    def __init__(self, inp, oup, kernel_size=3,  stride=1, ratio=2, dw_size=3, relued=True):
        super(Conv, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relued else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relued else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_ch, out_ch, downsampling=None):
        super(BasicBlock, self).__init__()
        self.conv = nn.Sequential(
                Conv(in_ch, out_ch),
                Conv(in_ch, out_ch, relued=False))
        self.relu = nn.ReLU()
    def forward(self, x):
        identity = x
        x = self.conv(x)
        x += identity
        return self.relu(x)


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_ch, out_ch, downsampling=None):
        super(Bottleneck, self).__init__()
        self.conv = nn.Sequential(
                Conv(in_ch, out_ch, kernel_size=1),
                Conv(out_ch, out_ch),
                Conv(out_ch, out_ch * self.expansion, kernel_size=1, relued=False))
        self.relu = nn.ReLU()
        self.downsampling = downsampling

    def forward(self, x):
        identity = x
        x = self.conv(x)
        if self.downsampling:
            identity = self.downsampling(identity)
        x += identity
        return self.relu(x)


class StageModule(nn.Module):
    def __init__(self, stage, output_branches, c):
        super(StageModule, self).__init__()
        self.stage = stage
        self.output_branches = output_branches

        self.branches = nn.ModuleList()
        for i in range(self.stage):
            w = c * (2 ** i)
            # 这里对应NUM_BLOCKS
            branch = nn.Sequential(
                BasicBlock(w, w),
                BasicBlock(w, w),
                BasicBlock(w, w),
                BasicBlock(w, w),
            )
            self.branches.append(branch)

        self.fuse_layers = nn.ModuleList()
        # for each output_branches (i.e. each branch in all cases but the very last one)
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.stage):  # for each branch
                if i == j:
                    self.fuse_layers[-1].append(nn.Sequential())  # Used in place of "None" because it is callable
                elif i < j:
                    self.fuse_layers[-1].append(nn.Sequential(
                        nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=(1, 1), stride=(1, 1), bias=False),
                        nn.BatchNorm2d(c * (2 ** i), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Upsample(scale_factor=(2.0 ** (j - i)), mode='nearest'),
                    ))
                elif i > j:
                    ops = []
                    for k in range(i - j - 1):
                        ops.append(nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** j), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                      bias=False),
                            nn.BatchNorm2d(c * (2 ** j), eps=1e-05, momentum=0.1, affine=True,
                                           track_running_stats=True),
                            nn.ReLU(inplace=True),
                        ))
                    ops.append(nn.Sequential(
                        nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                  bias=False),
                        nn.BatchNorm2d(c * (2 ** i), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    ))
                    self.fuse_layers[-1].append(nn.Sequential(*ops))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        assert len(self.branches) == len(x)

        x = [branch(b) for branch, b in zip(self.branches, x)]

        x_fused = []
        for i in range(len(self.fuse_layers)):
            for j in range(0, len(self.branches)):
                if j == 0:
                    x_fused.append(self.fuse_layers[i][0](x[0]))
                else:
                    x_fused[i] = x_fused[i] + self.fuse_layers[i][j](x[j])

        for i in range(len(x_fused)):
            x_fused[i] = self.relu(x_fused[i])

        return x_fused

class GhostHRNet(nn.Module):
    def __init__(self, channels = 32, class_num=200):
        super(GhostHRNet, self).__init__()

        self.init_conv = nn.Sequential(
                    Conv(3, 64, kernel_size=3, stride=2),
                    Conv(64, 64, kernel_size=3, stride=2))
        downsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )
        # stage1
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, downsampling=downsample),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
        )
        # Fusion layer 1 (transition1)      - Creation of the first two branches (one full and one half resolution)
        self.transition1 = nn.ModuleList([
            nn.Sequential(Conv(256, channels, stride=1)),
             nn.Sequential(nn.Sequential(  # Double Sequential to fit with official pretrained weights
                Conv(256, channels * (2 ** 1), stride=2)
            )),
        ])
        # Stage 2 (stage2)      - Second module with 1 group of bottleneck (resnet) modules. This has 2 branches
        self.stage2 = nn.Sequential(
            StageModule(stage=2, output_branches=2, c=channels),
        )
        # Fusion layer 2 (transition2)      - Creation of the third branch (1/4 resolution)
        self.transition2 = nn.ModuleList([
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(nn.Sequential(  # Double Sequential to fit with official pretrained weights
                Conv(channels* (2 ** 1), channels * (2 ** 2), stride=2)
            )),  # ToDo Why the new branch derives from the "upper" branch only?
        ])
        # Stage 3 (stage3)      - Third module with 4 groups of bottleneck (resnet) modules. This has 3 branches
        self.stage3 = nn.Sequential(
            StageModule(stage=3, output_branches=3, c=channels),
            StageModule(stage=3, output_branches=3, c=channels),
            StageModule(stage=3, output_branches=3, c=channels),
            StageModule(stage=3, output_branches=3, c=channels),
        )
        # Fusion layer 3 (transition3)      - Creation of the fourth branch (1/8 resolution)
        self.transition3 = nn.ModuleList([
            nn.Sequential(),
            nn.Sequential(), 
            nn.Sequential(), 
            nn.Sequential(nn.Sequential(  
                Conv(channels* (2 ** 2), channels * (2 ** 3), stride=2)
            )),
        ])
        # Stage 4 (stage4)      - Fourth module with 3 groups of bottleneck (resnet) modules. This has 4 branches
        self.stage4 = nn.Sequential(
            StageModule(stage=4, output_branches=4, c=channels),
            StageModule(stage=4, output_branches=4, c=channels),
            StageModule(stage=4, output_branches=4, c=channels),
        )
        self.incre_modules = nn.ModuleList([
            nn.Sequential(Bottleneck(channels, 32, downsampling=self.newdownsample(channels))),
            nn.Sequential(Bottleneck(channels* 2,64, downsampling=self.newdownsample(channels* 2))), 
            nn.Sequential(Bottleneck(channels* (2 ** 2),128, downsampling=self.newdownsample(channels* (2 ** 2)))), 
            nn.Sequential(Bottleneck(channels* (2 ** 3),256, downsampling=self.newdownsample(channels* (2 ** 3)))),
            ])
        self.downsamp_modules = nn.ModuleList([
            nn.Sequential(Conv(32*4,64*4,stride=2)),
            nn.Sequential(Conv(64*4,128*4,stride=2)), 
            nn.Sequential(Conv(128*4,256*4,stride=2)), 
            ])
        self.final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=256 * 4,
                out_channels=2048,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.BatchNorm2d(2048, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(2048, class_num)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def newdownsample(self,channels):
        return nn.Sequential(
                nn.Conv2d(channels, channels*4, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(channels*4, momentum=BN_MOMENTUM),
            )


    def forward(self, x):
        x = self.init_conv(x)
        x = self.layer1(x)

        x = [trans(x) for trans in self.transition1]  # Since now, x is a list (# == nof branches)

        x = self.stage2(x)
        # x = [trans(x[-1]) for trans in self.transition2]    # New branch derives from the "upper" branch only
        x = [
            self.transition2[0](x[0]),
            self.transition2[1](x[1]),
            self.transition2[2](x[-1])
        ]  # New branch derives from the "upper" branch only

        x = self.stage3(x)
        # x = [trans(x) for trans in self.transition3]    # New branch derives from the "upper" branch only
        x = [
            self.transition3[0](x[0]),
            self.transition3[1](x[1]),
            self.transition3[2](x[2]),
            self.transition3[3](x[-1])
        ]  # New branch derives from the "upper" branch only

        x = self.stage4(x)

        y = self.incre_modules[0](x[0])
        for i in range(len(self.downsamp_modules)):
            y = self.incre_modules[i+1](x[i+1])+ self.downsamp_modules[i](y)

        y = self.final_layer(y)
        if torch._C._get_tracing_state():
            y = y.flatten(start_dim=2).mean(dim=2)
        else:
            import torch.nn.functional as F
            y = F.avg_pool2d(y, kernel_size=y.size()
                                 [2:]).view(y.size(0), -1)

        y = self.classifier(y)
        return y

if __name__ == '__main__':
    # model = timm.create_model('hrnet_w32')
    model = GhostHRNet()
    # input = torch.rand(1, 3,256, 256)
    # output = model(input)
    # print(output)
    torchinfo.summary(model,input_size=(1, 3, 256, 256))


# https://github.com/samcw/ResNet18-GhostNet/blob/master/resnet18.py
# https://github.com/KopiSoftware/Ghost_ResNet56/blob/master/resnet.py
# https://github.com/HRNet/HRNet-Image-Classification/edit/master/lib/models/cls_hrnet.py
# https://github.com/iamhankai/ghostnet.pytorch/blob/master/ghost_net.py
# https://github.com/stefanopini/simple-HRNet/blob/master/models/hrnet.py