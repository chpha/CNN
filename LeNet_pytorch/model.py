import torch
from torch import nn
from torchsummary import summary


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1d = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.sigmoid1 = nn.Sigmoid()
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2d = nn.Conv2d(6, 16, 5, 1, 0)
        self.sigmoid2 = nn.Sigmoid()
        self.avgpool2 = nn.AvgPool2d(2, 2, 0) 
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(5*5*16, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.sigmoid1(self.conv1d(x))
        x = self.avgpool1(x)
        x = self.sigmoid2(self.conv2d(x))
        x = self.avgpool2(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x
    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = LeNet().to(device)
    summary(model, (1, 28, 28))


