# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 10:09:20 2020

@author: user
"""
import torch
import torch.nn as nn
import torchvision.models as models


def model_init(data_name, vocab=None):
    if(data_name == 'mnist'):
        model = Net_mnist()
    elif(data_name == 'cifar10'):
        model = Net_cifar10()
    elif(data_name == 'cifar100'):
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 100)
    elif(data_name == 'imdb'):
        embedding_dim = 128
        hidden_dim = 128
        vocab_size = len(vocab)
        num_class = 2
        model = AttentionModel(vocab_size, embedding_dim, hidden_dim, num_class)
    return model

class Net_mnist(nn.Module):
    def __init__(self):
        super(Net_mnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Net_cifar10(nn.Module):
    def __init__(self):
        super(Net_cifar10, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # 输出尺寸: (batch_size, 32, 16, 16)
        x = self.pool(torch.relu(self.conv2(x)))  # 输出尺寸: (batch_size, 64, 8, 8)
        x = x.view(-1, 64 * 8 * 8)  # 展平
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class AttentionModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_class):
        super(AttentionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, bidirectional=True, batch_first=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, num_class)

    def forward(self, text):
        embedded = self.embedding(text)
        lstm_out, _ = self.lstm(embedded)
        # 计算注意力权重
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        # 计算上下文向量
        context = torch.sum(attn_weights * lstm_out, dim=1)
        output = self.fc(context)
        return output