import torch
import torch.nn as nn
import torch.nn.functional as torchfunc


class HGBottleNeck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(HGBottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Hourglass(nn.Module):
    def __init__(self, block, nblocks, planes, depth=4):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.hg = self._make_hourglass(block, nblocks, planes, depth)

    def _make_hourglass(self, block, nblocks, planes, depth) -> nn.ModuleList:
        hg = []
        for i in range(depth):
            res = []

            for j in range(3):
                res.append(self._make_residual(block, nblocks, planes))
            if i == 0:
                res.append(self._make_residual(block, nblocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _make_residual(self, block, nblocks, planes) -> nn.Sequential:
        layers = []
        for i in range(0, nblocks):
            layers.append(block(planes * block.expansion, planes))
        return nn.Sequential(*layers)

    def _hourglass_foward(self, n, x) -> torch.Tensor:
        up1 = self.hg[n - 1][0](x)
        low1 = torchfunc.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n - 1][1](low1)

        if n > 1:
            low2 = self._hourglass_foward(n - 1, low1)
        else:
            low2 = self.hg[n - 1][3](low1)
        low3 = self.hg[n - 1][2](low2)
        up2 = torchfunc.interpolate(low3, scale_factor=2)

        # ============ DEAL WITH:  H != W and (H, W) != 2^n  >>>>>>>>>>>>>>>>>>>>
        # in case of up1 H != up2 H
        if up1.shape[2] != up2.shape[2]:
            up2 = torchfunc.pad(up2, pad=[0, 0, 0, 1])  # only ONE pixel in difference
        # in case of up1 W != up2 W
        if up1.shape[3] != up2.shape[3]:
            up2 = torchfunc.pad(up2, pad=[0, 1, 0, 0])  # only ONE pixel in difference

        out = up1 + up2
        return out

    def forward(self, x):
        return self._hourglass_foward(self.depth, x)


class StackedHourglass(nn.Module):
    def __init__(self, nstacks=2, nblocks=1, nclasses=64, block=HGBottleNeck):
        super(StackedHourglass, self).__init__()
        self.nclasses = nclasses
        self.nstacks = nstacks
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        self.layer1 = self._make_layer(block, planes=64, nblocks=nblocks, stride=1)
        self.layer2 = self._make_layer(block, planes=128, nblocks=nblocks, stride=1)
        self.layer3 = self._make_layer(block, planes=128, nblocks=nblocks, stride=1)

        self.num_feats = 128
        ch = self.num_feats * block.expansion

        hgs, res, fc, _fc, score, _score = [], [], [], [], [], []
        for i in range(nstacks):  # stacking the hourglass
            hgs.append(Hourglass(block, nblocks, self.num_feats, depth=4))
            res.append(self._make_layer(block, self.num_feats, nblocks=nblocks))
            fc.append(self._make_fc(ch, ch))
            score.append(nn.Conv2d(ch, nclasses, kernel_size=1, bias=True))

            if i < nstacks - 1:
                _fc.append(self._make_fc(ch, ch))
                _score.append(nn.Conv2d(nclasses, ch, kernel_size=1, bias=True))

        self.hgs = nn.ModuleList(hgs)  # hgs: hourglass stack
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self._fc = nn.ModuleList(_fc)
        self.score = nn.ModuleList(score)
        self._score = nn.ModuleList(_score)

    def _make_layer(self, block, planes, nblocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, nblocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(conv, bn, self.relu,)

    def forward(self, x):
        ls_out = []
        ls_encoding = []  # heatmaps encoding
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)
        ls_encoding.append(x)

        for i in range(self.nstacks):
            y = self.hgs[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            ls_out.append(score)
            if i < self.nstacks - 1:
                _fc = self._fc[i](y)
                _score = self._score[i](score)
                x = x + _fc + _score
                ls_encoding.append(x)
            else:
                ls_encoding.append(y)
        return ls_out, ls_encoding


def main():
    batch = torch.rand((4, 3, 480, 640))  # TENSOR (B, C, H, W)
    stack_hourglass = StackedHourglass(nstacks=1, nblocks=1, nclasses=64)
    ls_pred, ls_encodings = stack_hourglass(batch)
    print(f"HG Prediction Shape: {ls_pred[-1].shape}; HG Feature Shape: {ls_encoding[-1].shape}")


if __name__ == "__main__":
    main()
