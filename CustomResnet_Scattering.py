import torch
import torch.nn as nn
import math

DTYPE = torch.cuda.FloatTensor

class Flatten(nn.Module):

    """
    This module converts 4D tensors to 2D tensors.
    The tensor can be used in Linear Layer after this module.
    """
    def forward(self,x):
        value = x.shape[0]
        return x.view(value,-1)

class ResidualLayer(nn.Module):
    """
    This module performs the residual mapping.
    Attributes:
        downsampling (boolean): Specifies if dimension downsampling occurred.
        res_input (Tensor): The residual block input (identity).
        projection (boolean): When the dimensions of the residual block identity
        (res_input) and the input (x) do not match, linear projection
        used to make equal them.
    """

    def __init__(self, res_input,projection, downsampling=False):
        super().__init__()
        self.downsampling = downsampling
        self.res_input = res_input
        self.projection = projection

    def forward(self, x):
        """
        This function does the  element-wise sum between the residual block
        identity and the input of this module. If different dimension values
        occurs, perform a 1x1 convolution to match dimensions and number of
        channels.

        Returns
            Tensor: sum between residual block identity and input.
        """
        if self.projection:
            stride = 2 if self.downsampling else 1
            shortcut_conv = nn.Sequential(
                nn.Conv2d(
                    self.res_input.shape[1],
                    x.shape[1],
                    kernel_size=1,
                    stride=stride,
                    padding=0), nn.BatchNorm2d(x.shape[1])).type(DTYPE)
            output = shortcut_conv(self.res_input) + x
        else:
            output = self.res_input + x
        return output


class ResidualBlock(nn.Module):
    """
    This module performs full residual block.
    Initialize the model with the first part of the block, then call the residual
    layer and final relu in the forward method.
    Attributes:
        input_channels (int): Number of channels in the input.
        output_channels (int): Number of channels to the output.
        downsampling (boolean): Specifies if dimension downsampling occurred.
        bottleneck (boolean): Layer with fewer neurons than the layer below
        or above it. The bottleneck architecture is used in very deep networks
        due to computational considerations.
        bottleneck_factor (int): Specifies the value in 3x3 convolution.
    """

    def __init__(self,
                 input_channels,
                 output_channels,
                 downsampling=False,
                 bottleneck=False,
                 bottleneck_factor=4):
        super().__init__()

        self.downsampling = downsampling
        self.projection = input_channels != output_channels
        stride = 2 if downsampling else 1
        bottleneck_channels = int(input_channels / bottleneck_factor)
        if bottleneck:
            self.model = nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    bottleneck_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0), nn.BatchNorm2d(bottleneck_channels),
                nn.ReLU(),
                nn.Conv2d(
                    bottleneck_channels,
                    bottleneck_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1), nn.BatchNorm2d(bottleneck_channels),
                nn.ReLU(),
                nn.Conv2d(
                    bottleneck_channels,
                    output_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0), nn.BatchNorm2d(output_channels))
        else:
            self.model = nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1), nn.BatchNorm2d(output_channels), nn.ReLU(),
                nn.Conv2d(
                    output_channels,
                    output_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1), nn.BatchNorm2d(output_channels))

    def forward(self, x):
        """
        Run the initial part of the residual block, then instantiate and run
        the residual mapping layer and the final relu.
        Returns:
            Tensor: final output of the residual block.
        """
        output = self.model(x)
        residual_layer = ResidualLayer(x, self.projection,
                                       self.downsampling).type(DTYPE)
        output = residual_layer(output)
        relu = nn.ReLU()
        return relu(output)


class ResNet(nn.Module):
    """
    This Module contains ResNet model.
    Attributes:
        num_classes (int): Number of classes in the dataset.
    """

    def __init__(self, num_classes):
        super().__init__()
        self.name = 'ResNet'
        self.init_conv = nn.Sequential(
            nn.BatchNorm2d(51, eps=1e-5, affine=False),
            nn.Conv2d(51, 32,
                  kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        self.model = nn.Sequential(
            ResidualBlock(32, 64),
            ResidualBlock(64, 64), ResidualBlock(64, 64),ResidualBlock(64, 64),
            ResidualBlock(64, 64),ResidualBlock(64, 64),ResidualBlock(64, 64),

            ResidualBlock(64, 128), ResidualBlock(128, 128),ResidualBlock(128, 128),
            ResidualBlock(128, 128),ResidualBlock(128, 128),ResidualBlock(128, 128),
            ResidualBlock(128, 128), nn.AdaptiveAvgPool2d(2),

            Flatten(), nn.Linear(512, num_classes))
        
    def forward(self, x):
        """
        Performs the Resnet model.
        Returns:
            Model: final ResNet model.
        """
        x = x.view(x.size(0), 51, 16, 16)
        x = self.init_conv(x)
        return self.model(x)

