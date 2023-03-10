# -*- coding: utf-8 -*-
"""transformer_ultimus.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1XFlpn-ikBW314aiuAeAzE5aBlo1xaYVY
"""

# !pip install torchinfo -q --quiet

import torch.nn as nn
import torch
import torch.nn.functional as F

class Ultimus(nn.Module):
  def __init__(self,in_features=48, out_features=8):
    super(Ultimus,self).__init__()
    self.layer_K = nn.Linear(in_features = in_features, out_features = out_features, bias = False) # output 8
    self.layer_Q = nn.Linear(in_features = in_features, out_features = out_features, bias = False)
    self.layer_V = nn.Linear(in_features = in_features, out_features = out_features, bias = False)
    self.out = nn.Linear(in_features = out_features, out_features = in_features, bias = False)

  def forward(self,x):
    # print(x.shape)
    Q = self.layer_Q(x) # # Calculating k,q,v values from learnanble k,q,v learnable layers # torch.Size([B, 8])
    QT = torch.transpose(Q,-2,-1) # Transpose of Q matrix, please see that last 2 dimensions are transposed # torch.Size([8, B])

    K = self.layer_K(x) # torch.Size([B, 8])
    V = self.layer_V(x) # torch.Size([B, 8])

    # Attention matrix normalized
    AM = F.softmax(torch.matmul(QT,K) / torch.sqrt(torch.tensor(K.shape[-1]))) # SoftMax(QTK)/(8^0.5) #torch.Size([8, 8])

    # Value multiplied by Attention Matrix
    Z = torch.matmul(V,AM)  # torch.Size([B, 8])

    out = self.out(Z)
    return out


class TransformerUltimus(nn.Module):
  def __init__(self):
    super(TransformerUltimus,self).__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels = 16, kernel_size =3, padding=1, bias = False), # output - 16x32x32
        nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.Conv2d(in_channels=16, out_channels = 32, kernel_size =3, padding=1, bias = False), # output - 32x32x32
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.Conv2d(in_channels=32, out_channels = 48, kernel_size =3, padding=1, bias = False), # output - 48x32x32
        nn.ReLU(),
        nn.BatchNorm2d(48)
      )
    self.gap = nn.AdaptiveAvgPool2d(1) # output- 48x1x1

    self.ultimus_blocks = nn.Sequential(
        Ultimus(),
        Ultimus(),
        Ultimus(),
        Ultimus()
     )


    self.ffc = nn.Linear(in_features = 48, out_features = 10, bias = False)

  def forward(self,x):
    X = self.conv(x)
    X = self.gap(X)
    X = X.view(-1,48)
    # print(X.shape)
    X = self.ultimus_blocks(X)

    X = self.ffc(X)
    X = F.log_softmax(X, dim=-1)
    return X

# from torchinfo import summary

# model = TransformerUltimus()
# batch_size = 2
# summary(model, input_size=(batch_size, 3, 32, 32))


# model = Ultimus()
# batch_size = 2
# summary(model, input_size=(batch_size, 1, 1, 48))

