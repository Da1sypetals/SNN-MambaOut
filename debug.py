from env_cfg import env_configure

env_configure()

import torch
import torch.nn as nn
import torch.optim as optim
from spikingjelly.activation_based import functional

# 设置设备
device = "cuda:0"

# 初始化模型
from models.smo import *

net = mambaout_pico().to(device)
functional.set_step_mode(net, step_mode="m")
functional.set_backend(net, backend="cupy")
print(net)


def fb(net, x):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001)
    y_pred = net(x)
    loss = criterion(y_pred, torch.ones_like(y_pred))
    print(f"Loss: {loss.item()}")
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Output shape: {y_pred.shape}")


x = torch.randn(10, 3, 224, 224).to(device)
fb(net, x)
functional.reset_net(net)

x = torch.randn(12, 3, 224, 224).to(device)
fb(net, x)
