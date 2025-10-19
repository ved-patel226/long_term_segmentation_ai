import torch
import torch.nn as nn


def dice_loss(pred, target, smooth=1):
    pred = torch.sigmoid(pred)

    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

    dice = (2.0 * intersection + smooth) / (union + smooth)

    return 1 - dice.mean()


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, filters=[64, 128, 256, 512]):
        super().__init__()
        self.filters = filters
        self.enc1 = self._depth_wise_conv_block(in_channels, filters[0], dropout=0.3)
        self.enc2 = self._depth_wise_conv_block(filters[0], filters[1], dropout=0.2)
        self.enc3 = self._depth_wise_conv_block(filters[1], filters[2])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = self._depth_wise_conv_block(
            filters[2], filters[3], dropout=0.3
        )
        self.up3 = nn.ConvTranspose2d(filters[3], filters[2], 2, stride=2)
        self.dec3 = self._conv_block(filters[3], filters[2])
        self.up2 = nn.ConvTranspose2d(filters[2], filters[1], 2, stride=2)
        self.dec2 = self._conv_block(filters[2], filters[1])
        self.up1 = nn.ConvTranspose2d(
            filters[1],
            filters[0],
            2,
            stride=2,
        )
        self.dec1 = self._conv_block(filters[1], filters[0])
        self.out_conv = nn.Conv2d(filters[0], out_channels, 1)

    def _conv_block(self, in_ch, out_ch, dropout=0.3):
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))

        layers.extend(
            [
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True),
            ]
        )

        return nn.Sequential(*layers)

    def _depth_wise_conv_segment(self, in_ch, out_ch, dropout=0.0):
        layers = [
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch),
            nn.BatchNorm2d(in_ch),
            nn.LeakyReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))

        layers.extend(
            [
                nn.Conv2d(in_ch, out_ch, 1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(inplace=True),
            ]
        )
        return nn.Sequential(*layers)

    def _depth_wise_conv_block(self, in_ch, out_ch, dropout=0):
        return nn.Sequential(
            self._depth_wise_conv_segment(in_ch, out_ch, dropout),
            self._depth_wise_conv_segment(out_ch, out_ch, dropout),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)


class CBAM_UNet(UNet):
    def __init__(self, in_channels=3, out_channels=1, filters=[64, 128, 256, 512]):
        super().__init__(in_channels, out_channels, filters)
        # override decoders to add CBAM
        self.cbam_dec3 = CBAM(filters[2])
        self.cbam_dec2 = CBAM(filters[1])
        self.cbam_dec1 = CBAM(filters[0])

    def forward(self, x):
        # reuse UNet forward but add CBAM
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d3 = self.cbam_dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d2 = self.cbam_dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        d1 = self.cbam_dec1(d1)

        return self.out_conv(d1)
