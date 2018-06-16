import torch
import torch.nn as nn
import torch.nn.functional as F


class dense_layer(nn.Module):
    def __init__(self, input_channel, k, kernel_size = 3):
        
        super(dense_layer, self).__init__()

        # get padding size
        assert kernel_size % 2 == 1, "kernel_size should be odd"
        p = (kernel_size - 1) / 2

        # register submodules
        self.conv = nn.Conv2d(input_channel, k, kernel_size = kernel_size, padding = p, bias = False)
        self.bn = nn.BatchNorm2d(input_channel)


    def forward(self, inputs):
        # two conv layers path
        x = self.bn(inputs)
        x = F.relu(x, inplace = True)
        x = self.conv(x)

        # cat two paths
        assert x.shape[2:] == inputs.shape[2:], "cat failed in dense_layer"
        x = torch.cat((x, inputs), dim = 1)

        return x


class dense_block(nn.Module):
    def __init__(self, input_channel, n, kernel_size = 3, k = 12):
        super(dense_block, self).__init__()

        # get padding size
        assert kernel_size % 2 == 1, "kernel_size should be odd"
        p = (kernel_size - 1) / 2

        # register dense layers
        self.dense_layers = nn.ModuleList([dense_layer(input_channel + i*k, k) for i in range(n)])


    def forward(self, inputs):
        # apply the dense layers 
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)

        return x


class transition(nn.Module):
    def __init__(self, input_channel, output_channel = None):
        super(transition, self).__init__()
        
        # get output_channel
        output_channel = input_channel if output_channel is None else output_channel

        # register conv layer
        self.bn = nn.BatchNorm2d(input_channel)
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size = 1, bias = False)


    def forward(self, inputs):
        # bn-relu-conv 1x1, average pooling 2x2

        x = self.bn(inputs)
        x = F.relu(x, inplace = True)
        x = self.conv(x)

        return F.avg_pool2d(x, 2, 2)



if __name__ == "__main__":
    from torchvision.datasets import CIFAR10
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = CIFAR10("~/dataset/cifar10", transform = transform)
    x = trainset[0][0].unsqueeze(0)     # shape 1x3x32x32
    
    y = dense_block(3, n = 12)(x)
    y = transition(y.shape[1])(y)
    print(y.shape)