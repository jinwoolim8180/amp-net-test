import os
import numpy as np
import glob
from utils import *
from scipy import io
import torch
from torch.nn import Module
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

"""
No mask training, no deblocking
AMP-Net-K
"""


class ResBlock(Module):
    def __init__(self, n_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels, n_channels, 3, padding=1, bias=False)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.block(x)
        output += x
        return self.relu(output)


class Denoiser(Module):
    def __init__(self, n_stage=3, scale=1):
        super().__init__()
        self.scale = scale
        self.W_1 = nn.Conv2d(1, 32, 3, padding=1, bias=False)
        self.res_1 = nn.Sequential(*[ResBlock(32) for _ in range(n_stage)])
        self.res_2 = nn.Sequential(*[ResBlock(32) for _ in range(n_stage)])
        self.res_4 = nn.Sequential(*[ResBlock(32) for _ in range(n_stage)])
        self.W_2 = nn.Conv2d(32, 1, 3, padding=1, bias=False)

    def forward(self, inputs, r2=None, r4=None):
        h2, h4 = None, None
        inputs = torch.unsqueeze(torch.reshape(inputs.t(), [-1, 33, 33]), dim=1)
        output = self.W_1(inputs)
        output = self.res_1(output)
        if self.scale > 1:
            h2 = self.res_2(F.max_pool2d(output, kernel_size=2, stride=2))
            output = output + F.interpolate(h2, size=(33, 33))
        if self.scale > 2:
            h4 = self.res_4(F.max_pool2d(output, kernel_size=4, stride=4))
            output = output + F.interpolate(h4, size=(33, 33))
        output = self.W_2(output)

        # output=inputs-output
        output = torch.reshape(torch.squeeze(output), [-1, 33*33]).t()
        return output, h2, h4

class Deblocker(Module):
    def __init__(self):
        super().__init__()
        self.D = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1),

                               nn.ReLU(),
                               nn.Conv2d(32, 32, 3, padding=1),

                               nn.ReLU(),
                               nn.Conv2d(32, 32, 3, padding=1),

                               nn.ReLU(),
                               nn.Conv2d(32, 1, 3, padding=1,bias=False))

    def forward(self, inputs):
        inputs = torch.unsqueeze(inputs,dim=1)
        output = self.D(inputs)
        output = torch.squeeze(output,dim=1)
        return output

class AMP_net_Deblock(Module):
    def __init__(self,layer_num, A):
        super().__init__()
        self.layer_num = layer_num
        self.denoisers = []
        self.deblockers = []
        self.steps = []
        self.register_parameter("A", nn.Parameter(torch.from_numpy(A).float(),requires_grad=True))
        self.register_parameter("Q", nn.Parameter(torch.from_numpy(np.transpose(A)).float(), requires_grad=True))
        for n in range(layer_num):
            if n < 3:
                self.denoisers.append(Denoiser(scale=2**n))
            elif n == 4:
                self.denoisers.append(Denoiser(scale=2))
            else:
                self.denoisers.append(Denoiser(scale=1))
            self.deblockers.append(Deblocker())
            self.register_parameter("step_" + str(n + 1), nn.Parameter(torch.tensor(1.0),requires_grad=False))
            self.steps.append(eval("self.step_" + str(n + 1)))
        for n,denoiser in enumerate(self.denoisers):
            self.add_module("denoiser_"+str(n+1),denoiser)
        for n,deblocker in enumerate(self.deblockers):
            self.add_module("deblocker_"+str(n+1),deblocker)

    def forward(self, inputs, output_layers):
        H = int(inputs.shape[2]/33)
        L = int(inputs.shape[3]/33)
        S = inputs.shape[0]

        y = self.sampling(inputs)
        X = torch.matmul(self.Q,y)
        z = None
        r2 = None
        r4 = None

        for n in range(output_layers):
            step = self.steps[n]
            denoiser = self.denoisers[n]

            for i in range(20):
                r, z = self.block1(X, y, z, step)
            noise, r2, r4 = denoiser(X, r2, r4)
            X = r - torch.matmul(
                (step * torch.matmul(self.A.t(), self.A)) - torch.eye(33 * 33).float().cuda(), noise)

            X = self.together(X,S,H,L)
            X = torch.cat(torch.split(X, split_size_or_sections=33, dim=1), dim=0)
            X = torch.cat(torch.split(X, split_size_or_sections=33, dim=2), dim=0)
            X = torch.reshape(X, [-1, 33 * 33]).t()

        X = self.together(X, S, H, L)
        return torch.unsqueeze(X, dim=1)


    def sampling(self,inputs):
        # inputs = torch.squeeze(inputs)
        # inputs = torch.reshape(inputs,[-1,32*32])
        inputs = torch.squeeze(inputs,dim=1)
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=33, dim=1), dim=0)
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=33, dim=2), dim=0)
        inputs = torch.transpose(torch.reshape(inputs, [-1, 33*33]),0,1)
        outputs = torch.matmul(self.A, inputs)
        return outputs

    def block1(self, X, y, z, step):
        # X = torch.squeeze(X)
        # X = torch.transpose(torch.reshape(X, [-1, 32 * 32]),0,1)
        # z *= 0.1
        z = y - torch.matmul(self.A, X)
        outputs = torch.matmul(self.A.t(), z)
        outputs = step * outputs + X
        # outputs = torch.unsqueeze(torch.reshape(torch.transpose(outputs,0,1),[-1,32,32]),dim=1)
        return outputs, z

    def together(self,inputs,S,H,L):
        inputs = torch.reshape(torch.transpose(inputs,0,1),[-1,33,33])
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=H*S, dim=0), dim=2)
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=S, dim=0), dim=1)
        return inputs