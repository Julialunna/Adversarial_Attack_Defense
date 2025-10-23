# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from flautim.pytorch.Model import Model


# class Model(Model):
#     def __init__(self, context, num_classes=10, **kwargs):
#         super(Model, self).__init__(context, name="LeNet5", **kwargs)

#         # entrada esperada -> imagens preta e branca 1x28x28
#         #caso tenha cores (RGB)
#         #self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1)
#         self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
#         self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)

#         # Após convs/pools, o tamanho típico de saída é 16x4x4 (para entrada 28x28)
#         self.fc1 = nn.Linear(16 * 4 * 4, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, num_classes)

#         #testar se fica melhor com o dropout
#         self.dropout = nn.Dropout(p=0.1)

#     @staticmethod
#     def build_lenet5(num_classes=10):
#         """
#         Retorna uma instância da LeNet-5 pura (sem wrapper do Model).
#         """
#         class LeNet5(nn.Module):
#             def __init__(self):
#                 super().__init__()
#                 self.conv1 = nn.Conv2d(1, 6, 5)
#                 self.pool = nn.AvgPool2d(2)
#                 self.conv2 = nn.Conv2d(6, 16, 5)
#                 self.fc1 = nn.Linear(16 * 4 * 4, 120)
#                 self.fc2 = nn.Linear(120, 84)
#                 self.fc3 = nn.Linear(84, num_classes)

#             def forward(self, x):
#                 x = F.relu(self.conv1(x))
#                 x = self.pool(x)
#                 x = F.relu(self.conv2(x))
#                 x = self.pool(x)
#                 x = x.view(x.size(0), -1)
#                 x = F.relu(self.fc1(x))
#                 x = F.relu(self.fc2(x))
#                 x = self.fc3(x)
#                 return x

#         return LeNet5()

#     def forward(self, x):
#         # Fluxo da LeNet-5
#         x = F.relu(self.conv1(x))
#         x = self.pool(x)
#         x = F.relu(self.conv2(x))
#         x = self.pool(x)
#         x = x.view(x.size(0), -1)  # Flatten
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

from flautim.pytorch.Model import Model
import torch
import torch.nn as nn
import torch.nn.functional as F



class MNISTModel(Model):
    def __init__(self, context, num_classes: int, **kwargs) -> None:
        super(MNISTModel, self).__init__(context, name = "MNIST", version = 1, id = 1, **kwargs)
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x