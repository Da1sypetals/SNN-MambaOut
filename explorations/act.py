import torch
import torch.nn as nn
import torch.nn.functional as F


class sReLU_Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, beta):
        ctx.save_for_backward(input, torch.tensor(beta))
        return F.relu(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, beta = ctx.saved_tensors
        grad_input = (
            grad_output
            * (1 - torch.exp(-beta * input))
            / (1 + torch.exp(-beta * input))
        )
        return grad_input, None


def srelu(input, beta):
    return sReLU_Fn.apply(input, beta)


class sReLU(nn.Module):
    def __init__(self, beta=1.0):
        super(sReLU, self).__init__()
        self.beta = beta

    def forward(self, input):
        return srelu(input, self.beta)


###############################################################


class SepLU_Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a):
        ctx.save_for_backward(x, torch.tensor(a))
        return torch.where(x > 0, a * x, torch.tensor(0.0))

    @staticmethod
    def backward(ctx, grad_output):
        x, a = ctx.saved_tensors
        grad_x = grad_output * torch.where(x > 0, a * x, a * torch.exp(x))
        return grad_x, None


def seplu(input, beta):
    return SepLU_Fn.apply(input, beta)


class SepLU(nn.Module):
    def __init__(self, a=1.0):
        super(SepLU, self).__init__()
        self.a = a

    def forward(self, x):
        return seplu(x, self.a)
