import torch
import torch.nn as nn

class DWTHaar(nn.Module):
    def __init__(self, num_dec):
        super(DWTHaar, self).__init__()
        # self.requires_grad = False
        self.num_dec = num_dec

    def dwt2d(self, x):
        '''
        :param x: expected b c h w
        :return:
        '''
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4

        return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)

    def dwt1d(self, x):
        '''
        :param x: expected b n c
        :return:
        '''
        x01 = x[:, :, 0::2] / 2
        x02 = x[:, :, 1::2] / 2
        x_L = 2**0.5 * (x01 + x02)
        x_H = 2**0.5 * (x01 - x02)

        return x_L, x_H

    def forward(self, x):
        x_iter = x
        x_l = 0
        x_h = []
        for i in range(self.num_dec):
            l, h = self.dwt1d(x_iter)
            x_h.append(h)
            if i == self.num_dec-1:
                x_l = l
            else:
                x_iter = l
        return x_l, x_h