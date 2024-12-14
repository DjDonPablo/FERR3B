import torch.nn as nn
import torch.nn.functional as F


class FERModel(nn.Module):
    def __init__(self, nb_classes: int):
        super(FERModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(256 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, nb_classes)

        self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.batch_norm3 = nn.BatchNorm2d(256)

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.4)

    def forward(self, x):
        x = F.gelu(self.conv1(x))

        x = self.conv2(x)
        x = F.gelu(F.max_pool2d(x, 2))
        x = self.batch_norm1(x)

        x = self.conv3(x)
        x = F.gelu(F.max_pool2d(x, 2))
        x = self.batch_norm2(x)

        x = self.dropout1(x)

        x = F.gelu(self.conv4(x))
        x = self.batch_norm3(x)

        x = x.view(x.size(0), -1)
        x = F.gelu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)

        return x
