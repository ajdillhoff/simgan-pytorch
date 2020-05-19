from torch import nn


class ResnetBlock(nn.Module):
    def __init__(self, in_features, num_features,
                 kernel_size, stride, padding):
        super(ResnetBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_features, num_features,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_features, num_features,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        out += identity
        out = self.relu(out)

        return out


class Refiner(nn.Module):
    def __init__(self, num_blocks, in_features=1, num_features=64):
        super(Refiner, self).__init__()

        self.conv1 = nn.Conv2d(in_features, num_features, kernel_size=7,
                               stride=1, padding=3)
        self.relu = nn.ReLU()

        blocks = [ResnetBlock(num_features, num_features, 7, 1, 3)
                  for i in range(num_blocks)]
        self.blocks = nn.Sequential(*blocks)

        self.conv2 = nn.Conv2d(num_features, in_features, kernel_size=1,
                               stride=1, padding=0)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)

        out = self.blocks(out)

        out = self.conv2(out)
        out = self.tanh(out)

        return out


class Discriminator(nn.Module):
    def __init__(self, in_features=1):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_features, 96, kernel_size=7, stride=4)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(96, 64, kernel_size=5, stride=2)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=1, stride=1)
        self.conv5 = nn.Conv2d(32, 2, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.max_pool(out)

        out = self.conv3(out)
        out = self.relu(out)

        out = self.conv4(out)
        out = self.relu(out)

        out = self.conv5(out)
        out = self.relu(out)

        return out.view(out.size(0), -1, 2)
