import torch.nn as nn

class AgeModel(nn.Module):
    def __init__(self):
        super(AgeModel, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # conv1
            nn.ReLU(),
            nn.MaxPool2d(2),  # 100 -> 50

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # conv2
            nn.ReLU(),
            nn.MaxPool2d(2),  # 50 -> 25

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # conv3
            nn.ReLU(),
            nn.MaxPool2d(2)  # 25 -> 12
        )

        self.fc = nn.Sequential(
            nn.Flatten(),                       # 128 * 12 * 12 = 18432
            nn.Linear(128 * 12 * 12, 128),      # match checkpoint
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x.squeeze()
