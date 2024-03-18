import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        """
        CNNBlock class represents a convolutional block in a neural network.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int, optional): Stride value for the convolutional layer. Defaults to 2.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        """
        Forward pass of the CNNBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the convolutional block.
        """
        return self.conv(x)
    

class Descriminator(nn.Module):
    """
    The Descriminator class represents the discriminator network in the Pix2Pix model.

    Args:
        in_channels (int): Number of input channels. Default is 3.
        features (list): List of integers representing the number of features in each layer. Default is [64, 128, 256, 512].

    Attributes:
        initial (nn.Sequential): Initial layers of the discriminator network.
        model (nn.Sequential): Main model of the discriminator network.

    Methods:
        forward(x, y): Forward pass of the discriminator network.

    """

    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels*2, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )
        layers = []
        in_channels = features[0]
        for feature in features [1:]:
            layers.append(
                CNNBlock(in_channels,feature, stride=1 if feature == features[-1] else 2 ),
            
            )
            in_channels = feature
        
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"))
            
        self.model = nn.Sequential(*layers)
        
    def forward(self, x, y):
        """
        Forward pass of the discriminator network.

        Args:
            x (torch.Tensor): Input tensor representing the real image.
            y (torch.Tensor): Input tensor representing the generated image.

        Returns:
            torch.Tensor: Output tensor representing the discriminator's prediction.

        """
        x = torch.cat([x, y], dim = 1)
        x = self.initial(x)
        return self.model(x)
    
    
def test():
    x = torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 3, 256, 256))
    model = Descriminator()
    preds = model(x, y)
    print(preds.shape)
    
    
if __name__ == "__main__":
    test()
        