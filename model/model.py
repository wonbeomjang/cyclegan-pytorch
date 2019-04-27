import torch.nn as nn

def weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)

    elif class_name.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class ResidualBlock(nn.Module):
    def __init__(self, in_channel):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1),
            nn.InstanceNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.block(x)
        return out + x


class ResidualGenerator(nn.Module):
    def __init__(self, in_shape, num_residual_block):
        super(ResidualGenerator, self).__init__()
        in_channels = in_shape[0]
        out_channels = 64

        self.model = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7),
                      nn.InstanceNorm2d(out_channels),
                      nn.ReLU(inplace=True)]
        in_channels = out_channels

        for _ in range(2):
            out_channels *= 2
            self.model += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2),
                           nn.InstanceNorm2d(out_channels),
                           nn.ReLU(inplace=True)]
            in_channels = out_channels

        for _ in range(num_residual_block):
            self.model += [ResidualBlock(in_channels)]

        for _ in range(2):
            out_channels //= 2
            self.model += [nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2),
                           nn.InstanceNorm2d(out_channels),
                           nn.ReLU(inplace=True)]
            in_channels = out_channels

        self.model = [nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=7),
                      nn.InstanceNorm2d(out_channels),
                      nn.ReLU(inplace=True)]

        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, in_shape):
        super(Discriminator, self).__init__()
        in_channels = in_shape[0]
        out_channels=64

        self.model = []
        for _ in range(4):
            self.model += self.discriminator_block(in_channels=in_channels, out_channels=out_channels)
            in_channels = out_channels
            out_channels *= 2
        in_channels = out_channels

        self.model += [nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=4),
                       nn.Sigmoid()]

    def discriminator_block(self, in_channels, out_channels, nomalization=True):
        block = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2),
                 nn.InstanceNorm2d(out_channels),
                 nn.LeakyReLU(inplace=True)]

        return block
