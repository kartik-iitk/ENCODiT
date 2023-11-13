import numpy as np

import torch
import torch.nn as nn


class EncBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(EncBlock, self).__init__()
        self.layers = nn.Sequential()
        self.layers.add_module('Linear', nn.Linear(in_features=in_features, out_features=out_features))

    def forward(self, x):
        out = self.layers(x)
        return torch.cat([x, out], dim=1)

class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, feature_dim)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        batch_size, step_dim, feature_dim = x.size()
        eij = torch.mm(x.reshape(-1, feature_dim), self.weight).view(-1, step_dim)
        if self.bias:
            eij = eij + self.b
        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / torch.sum(a, 1, keepdim=True) + 1e-10
        weighted_input = x * a.view(batch_size, step_dim, feature_dim)
        return torch.sum(weighted_input, 1)

class RNN(nn.Module):

    def __init__(self, n_classes, in_feat=720):
        super(RNN, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),nn.Tanh())
        self.layer2 = nn.AvgPool2d(kernel_size=2)
        self.layer11 = nn.Sequential(nn.GRU(input_size=1, hidden_size= 32, num_layers=4, batch_first=True))
        self.layer3 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),nn.Tanh())
        self.layer4 = nn.AvgPool2d(kernel_size=2)
        self.layer5 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding='same'),nn.Tanh())
        self.gru = nn.GRU(input_size=12, hidden_size= 60, num_layers=4, bidirectional=True, batch_first=True)
        self.gru1 = nn.GRU(input_size=120, hidden_size= 60, num_layers=4, bidirectional=True, batch_first=True)
        self.gru_attention = Attention(120, 254)
        self.bn = nn.BatchNorm1d(120, momentum=0.5)
        self.linear = nn.Linear(120, 16 ** 2)  # 643:80 - 483:60 - 323:40
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(16 ** 2, 16)
        self.out = nn.Linear(16, 1)
        self.layer6 = nn.Sequential(nn.Flatten(),nn.Linear(in_features=30480, out_features=84), nn.Tanh())

        # new enc block
        self.enc = EncBlock(in_features=84, out_features=84)
        self.all_layers = [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5,
                           self.layer6]  # , self.layer7]
        self.lstm = nn.LSTM(input_size=12, hidden_size= 60, num_layers= 6, bidirectional= True, batch_first= True)
        self.lstm1 = nn.LSTM(input_size=120, hidden_size= 60, num_layers= 6, bidirectional= True, batch_first= True)

    def forward(self, x):
        #x = self.layer2(x)

        #x = self.layer11(x)# AvgPool

        # x = self.layer1(x)           #Hybrid GRY-CNN
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        # x = self.layer5(x)


        x, _ = self.gru(x)
        #x = self.gru_attention(x)        #GRU / GRU+Attention
        x, _ = self.gru1(x)
        #x = self.gru_attention(x)
        #x = self.bn(x)

        #x, _ = self.lstm(x)             #LSTM / LSTM+Attention
        # x = self.gru_attention(x)
        #x, _ = self.lstm1(x)
        # x = self.gru_attention(x)
        # x = self.bn(x)

        x = self.layer6(x)  # Linear

        # ENCODING
        enc_dim = 84
        feat = self.enc(x)
        mu = feat[:, :enc_dim]
        logvar = feat[:, enc_dim:]
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        # print("std.shape: {}, mu.shape: {}".format(std.shape, mu.shape))
        feat = eps.mul(std * 0.001).add_(mu)

        return feat  # x


class Regressor(nn.Module):
    def __init__(self, indim=168, num_classes=4,
                 wl=16):  # indim = [orig_img_avg_pooled_features, trans_img_avg_pooled_features] = 2*enc_dim from forward, num_classes is the number of possible transformations = 4}
        super(Regressor, self).__init__()

        self.num_classes = num_classes

        fc1_outdim = 42

        if wl == 16:
            in_feat = 720
        else:
            in_feat = 1440
        self.rnn = RNN(n_classes=self.num_classes, in_feat=in_feat)

        self.fc1 = nn.Linear(indim, fc1_outdim)
        self.fc2 = nn.Linear(fc1_outdim, num_classes)

        self.relu1 = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x1, x2):  # shape of x1 and x2 should be 5D = [BS, C=3, No. of images=clip_total_frames, 224, 224]
        # print("x2 shape before: ", x2.shape)
        x1 = x1.to(torch.float32)
        x2 = x2.to(torch.float32)

        #print(x1.dtype)
        x1 = self.rnn(x1)
        x2 = self.rnn(x2)  # now the shape of x1 = x2 = BS X 512
        # print("x2 shape: ", x2.shape)
        x = torch.cat((x1, x2), dim=1)
        # print("x shape: ", x.shape)
        x = self.fc1(x)
        # print("After fc1, x shape:  ", x.shape)
        x = self.relu1(x)
        penul_feat = x
        x = self.fc2(penul_feat)
        return x
