# encoding: utf-8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks import dense_block, transition


class DenseNet3(nn.Module):
    def __init__(self, output_shape, n = 12, k = 12):

        super(DenseNet3, self).__init__()
        self.output_shape = output_shape

        # regiter first conv layer and last fc layer
        self.conv = nn.Conv2d(3, 16, kernel_size = 3, padding = 1, bias = False)
        self.fc = nn.Linear(16 + 3*n*k, output_shape, bias = True)
        self.bn = nn.BatchNorm2d(16 + 3*n*k)
        
        # register 3 dense blocks
        self.dense_block1 = dense_block(16, n = n, k = k)
        self.dense_block2 = dense_block(16 + n*k, n = n, k = k)
        self.dense_block3 = dense_block(16 + 2*n*k, n = n, k = k)

        # register transition blocks
        self.transition1 = transition(16 + n*k)
        self.transition2 = transition(16 + 2*n*k)

        # weights init
        self.weights_init()


    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode = 'fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()


    def forward(self, inputs):
        # expect input size 3x32x32
        assert(inputs.shape[1:] == torch.Size([3,32,32]))

        # 1 conv layer 
        x = self.conv(inputs)

        # expect x size 16x32x32 here
        assert(x.shape[1:] == torch.Size([16,32,32]))


        # first dense block and transition
        x = self.dense_block1(x)
        x = self.transition1(x)

        # expect x size 160x16x16 here
        assert(x.shape[1:] == torch.Size([160,16,16]))


        # second dense block and transition
        x = self.dense_block2(x)
        x = self.transition2(x)

        # expect x size 304x8x8 here
        assert(x.shape[1:] == torch.Size([304,8,8]))


        # third dense block and no transition
        x = self.dense_block3(x)

        # expect x size 448x8x8 here
        assert(x.shape[1:] == torch.Size([448,8,8]))


        # there should be a BN-Relu layer
        x = self.bn(x)
        x = F.relu(x, inplace = True)


        # global pooling and fc
        x = F.avg_pool2d(x, x.shape[-1])
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    from torchvision.datasets import CIFAR10
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = CIFAR10("~/dataset/cifar10", transform = transform)
    x = trainset[0][0].unsqueeze(0)
    densenet = DenseNet3(10)
    y = densenet(x)

    print(x.shape)
    print(y.shape)
