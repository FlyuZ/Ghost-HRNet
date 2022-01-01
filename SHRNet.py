import torch
from torch import nn

BN_MOMENTUM = 0.1

class ConvBR(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size=3,  stride=1, padding=1, bias=False, relued=True):
        super(ConvBR, self).__init__()
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(outplanes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.relued = relued
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relued:
            x = self.relu(x)
        return x

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_ch, out_ch, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv = nn.Sequential(
                ConvBR(in_ch, out_ch, stride=1),
                ConvBR(in_ch, out_ch, relued=False))
        self.relu = nn.ReLU()
        self.downsample = downsample
    def forward(self, x):
        residual = x
        out = self.conv(x)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return x


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_ch, out_ch, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv = nn.Sequential(
                ConvBR(in_ch,  out_ch, kernel_size=1, padding=0),
                ConvBR(out_ch, out_ch, kernel_size=3),
                ConvBR(out_ch, out_ch * self.expansion, kernel_size=1, padding=0, relued=False))
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv(x)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class StageModule(nn.Module):
    def __init__(self, stage, output_branches, c):
        super(StageModule, self).__init__()
        self.stage = stage
        self.output_branches = output_branches

        self.branches = nn.ModuleList()
        for i in range(self.stage):
            w = c * (2 ** i)
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
                            ConvBR(c * (2 ** j), c * (2 ** j), kernel_size=3, stride=2, padding=1, bias=False,relued=True)
                        ))
                    ops.append(nn.Sequential(
                        ConvBR(c * (2 ** j), c * (2 ** i), kernel_size=3, stride=2, padding=1, bias=False,relued=False)
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


class SHRNet(nn.Module):
    def __init__(self, c=18):
        super(SHRNet, self).__init__()
        # Input (stem net)
        self.init_conv = nn.Sequential(
            ConvBR(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            ConvBR(64, 64, kernel_size=3, stride=2, padding=1, bias=False))
        downsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
        )
        self.layer1_1 = Bottleneck(64, 64, downsample=downsample)
        self.layer1_2 = nn.Sequential(
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
        )

        # Fusion layer 1 (transition1)      - Creation of the first two branches (one full and one half resolution)
        self.transition1 = nn.ModuleList([
            nn.Sequential(
                ConvBR(256, c, kernel_size=3, stride=1, padding=1, bias=False)
            ),
            nn.Sequential(nn.Sequential(  # Double Sequential to fit with official pretrained weights
                ConvBR(256, c * (2 ** 1), kernel_size=3, stride=2, padding=1, bias=False)
            )),
        ])

        # Stage 2 (stage2)      - Second module with 1 group of bottleneck (resnet) modules. This has 2 branches
        self.stage2 = nn.Sequential(
            StageModule(stage=2, output_branches=2, c=c),
        )

        # Fusion layer 2 (transition2)      - Creation of the third branch (1/4 resolution)
        self.transition2 = nn.ModuleList([
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(nn.Sequential(  # Double Sequential to fit with official pretrained weights
                ConvBR(c * (2 ** 1), c * (2 ** 2), kernel_size=3, stride=2, padding=1, bias=False)
            )),  # ToDo Why the new branch derives from the "upper" branch only?
        ])

        # Stage 3 (stage3)      - Third module with 4 groups of bottleneck (resnet) modules. This has 3 branches
        self.stage3 = nn.Sequential(
            StageModule(stage=3, output_branches=3, c=c),
            StageModule(stage=3, output_branches=3, c=c),
            StageModule(stage=3, output_branches=3, c=c),
            StageModule(stage=3, output_branches=3, c=c),
        )

        # Fusion layer 3 (transition3)      - Creation of the fourth branch (1/8 resolution)
        self.transition3 = nn.ModuleList([
            nn.Sequential(),  
            nn.Sequential(),  
            nn.Sequential(),  
            nn.Sequential(nn.Sequential( 
                ConvBR(c * (2 ** 2), c * (2 ** 3), kernel_size=3, stride=2, padding=1, bias=False)
            )),  
        ])

        # Stage 4 (stage4)      - Fourth module with 3 groups of bottleneck (resnet) modules. This has 4 branches
        self.stage4 = nn.Sequential(
            StageModule(stage=4, output_branches=4, c=c),
            StageModule(stage=4, output_branches=4, c=c),
            StageModule(stage=4, output_branches=4, c=c),
        )

        self.incre_modules = nn.ModuleList([
            nn.Sequential(Bottleneck(c, 32, downsample=self.newdownsample(c, 32))),
            nn.Sequential(Bottleneck(c* 2, 64, downsample=self.newdownsample(c* 2, 64))), 
            nn.Sequential(Bottleneck(c* (2 ** 2), 128, downsample=self.newdownsample(c* (2 ** 2), 128))), 
            nn.Sequential(Bottleneck(c* (2 ** 3), 256, downsample=self.newdownsample(c* (2 ** 3), 256))),
            ])
        self.downsamp_modules = nn.ModuleList([
            nn.Sequential(ConvBR(32*4,64*4,stride=2)),
            nn.Sequential(ConvBR(64*4,128*4,stride=2)), 
            nn.Sequential(ConvBR(128*4,256*4,stride=2)), 
            ])
        self.final_layer = nn.Sequential(ConvBR(256 * 4, 2048, kernel_size=1,stride=1,padding=0))
        self.classifier = nn.Linear(2048, 1000)

    def newdownsample(self,channels, outplanes):
        return nn.Sequential(
                nn.Conv2d(channels, outplanes*4, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(outplanes*4, momentum=BN_MOMENTUM),
            )        

    def forward(self, x):
        x = self.init_conv(x)

        x = self.layer1_1(x)
        x = self.layer1_2(x)
        x = [trans(x) for trans in self.transition1]  # Since now, x is a list (# == nof branches)

        x = self.stage2(x)
        x = [
            self.transition2[0](x[0]),
            self.transition2[1](x[1]),
            self.transition2[2](x[-1])
        ]  # New branch derives from the "upper" branch only

        x = self.stage3(x)
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
    model = SHRNet()
    # input = torch.rand(1, 3, 224, 224)
    # output = model(input)
    # print(output)
    import torchinfo
    torchinfo.summary(model,input_size=(1, 3, 224, 224))