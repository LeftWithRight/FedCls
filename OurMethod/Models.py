import torch
import torch.nn as nn
import torch.nn.functional as F


# class CoordinateAttention(nn.Module):
#     def __init__(self, in_channels, height, width):   # def __init__(self, in_channels, height, width):
#         super(CoordinateAttention, self).__init__()
#         self.fc1 = nn.Linear(height * width, 64)
#         self.fc2 = nn.Linear(64, height * width)
#
#     def forward(self, x):
#         batch_size, channels, height, width = x.size()
#
#         # Reshape input to (batch_size, height*width)
#         x = x.view(batch_size, channels, height * width)
#
#         # Calculate attention weights
#         attention_weights = self.fc2(F.relu(self.fc1(x)))
#
#         # Reshape attention weights to match input size
#         attention_weights = attention_weights.view(batch_size, height, width)
#
#         # Apply attention weights to input
#         x = x * attention_weights.unsqueeze(1)
#
#         return x


# 定义神经网络
class Mnist_2NN(nn.Module):
    def __init__(self, dropout_prob):
        super().__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.dropout2 = nn.Dropout(p=dropout_prob)

    def forward(self, inputs):
        tensor = F.relu(self.fc1(inputs))
        tensor = self.dropout1(tensor)
        tensor = F.relu(self.fc2(tensor))
        tensor = self.dropout2(tensor)
        tensor = self.fc3(tensor)
        return tensor

class CoordinateAttention(nn.Module):
    def __init__(self, in_channels, height, width):
        super(CoordinateAttention, self).__init__()
        self.fc1 = nn.Linear(height * width, 64)
        self.fc2 = nn.Linear(64, height * width)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Reshape input to (batch_size, height * width)
        x_reshaped = x.view(batch_size, channels, height * width)

        # Calculate attention weights
        attention_weights = self.fc2(F.relu(self.fc1(x_reshaped)))

        # Reshape attention weights to match input size
        attention_weights = attention_weights.view(batch_size, channels, height, width)

        # Apply attention weights to input
        x_att = x * attention_weights

        return x_att


class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(7*7*64, 512)
        self.fc2 = nn.Linear(512, 10)

        # Add Coordinate Attention modules
        self.coord_att1 = CoordinateAttention(in_channels=32, height=28, width=28)
        self.coord_att2 = CoordinateAttention(in_channels=64, height=14, width=14)

    def forward(self, inputs):
        tensor = inputs.view(-1, 1, 28, 28)
        tensor = F.relu(self.conv1(tensor))
        tensor = self.coord_att1(tensor)
        tensor = self.pool1(tensor)
        tensor = F.relu(self.conv2(tensor))
        tensor = self.coord_att2(tensor)
        tensor = self.pool2(tensor)
        tensor = tensor.view(-1, 7*7*64)
        tensor = F.relu(self.fc1(tensor))
        tensor = self.fc2(tensor)
        return tensor





class Mnist_CNN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(7*7*64, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, inputs):
        tensor = inputs.view(-1, 1, 28, 28)
        tensor = F.relu(self.conv1(tensor))
        tensor = self.pool1(tensor)
        tensor = F.relu(self.conv2(tensor))
        tensor = self.pool2(tensor)
        tensor = tensor.view(-1, 7*7*64)
        tensor = F.relu(self.fc1(tensor))
        tensor = self.fc2(tensor)
        return tensor

