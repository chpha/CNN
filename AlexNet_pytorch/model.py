import torch 
from torch import nn
from torchsummary import summary
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.ReLU = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 96, 11, 4)
        self.maxpool1 = nn.MaxPool2d(3, 2)
        self.conv2 = nn.Conv2d(96, 256, 5, 1, 2)
        self.maxpool2 = nn.MaxPool2d(3, 2)
        self.conv3 = nn.Conv2d(256, 384, 3, 1, 1)
        self.conv4 = nn.Conv2d(384, 384, 3, 1, 1)
        self.conv5 = nn.Conv2d(384, 256, 3, 1, 1)
        self.maxpool3 = nn.MaxPool2d(3, 2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(6*6*256, 4096)
        self.linear2 = nn.Linear(4096, 4096)
        self.linear3 = nn.Linear(4096, 10)

    def forward(self, x):
        x = self.ReLU(self.conv1(x))
        x = self.maxpool1(x)
        x = self.ReLU(self.conv2(x))
        x = self.maxpool2(x)
        x = self.ReLU(self.conv3(x))
        x = self.ReLU(self.conv4(x))
        x = self.ReLU(self.conv5(x))
        x = self.maxpool3(x)

        x = self.flatten(x)
        x = self.ReLU(self.linear1(x))
        x = F.dropout(x, 0.5)
        x = self.ReLU(self.linear2(x))
        x = F.dropout(x, 0.5)
        x = self.linear3(x)
        return x
    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlexNet().to(device)
    summary(model, (1, 227, 227))


