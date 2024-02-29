import torch
import torch.nn as nn
import torch.nn.functional as F

class TransferConv_m(nn.Module):
    def __init__(self, resnet, in_c):
        super().__init__()
        if resnet:
            self.layer1 = nn.Sequential(
                nn.Conv2d(in_c // 2, in_c // 2, kernel_size=1, padding=0),
                nn.BatchNorm2d(in_c // 2, momentum=0.1, affine=True),
                nn.LeakyReLU(0.2, True),
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(in_c // 2, in_c, kernel_size=1, padding=0),
                nn.BatchNorm2d(in_c, momentum=0.1, affine=True),
                nn.LeakyReLU(0.2, True),
                nn.MaxPool2d(2)
            )
        else:
            self.layer1 = nn.Sequential(
                nn.Conv2d(in_c, in_c // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_c // 2, momentum=0.1, affine=True),
                nn.LeakyReLU(0.2, True),
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(in_c // 2, in_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_c, momentum=0.1, affine=True),
                nn.LeakyReLU(0.2, True),
                nn.MaxPool2d(2)
            )
        
    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output)

        return output


class TransferConv_h(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_c, momentum=0.1, affine=True),
            nn.LeakyReLU(0.2, True),

        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=1,padding=0),
            nn.BatchNorm2d(in_c, momentum=0.1, affine=True),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x):
        output = self.layer1(x)
        output = self.layer2(output)

        return output

class CLFR(nn.Module):
    def __init__(self, resnet, in_c):
        super().__init__()
        self.transferconv_h = TransferConv_h(in_c)
        self.transferconv_m = TransferConv_m(resnet, in_c)

    def reconstructing_procedure(self, f_h, f_m):
        _, c, h, w = f_h.shape
        f_m = f_m.view(f_m.size(0), f_m.size(1), -1)
        f_h = f_h.view(f_h.size(0), f_h.size(1), -1)

        f_m_T = torch.transpose(f_m, 2, 1)
        matrix_hm = torch.matmul(f_m_T, f_h)
        l2_m = torch.norm(matrix_hm)
        matrix_hm = torch.tanh(matrix_hm / l2_m)
        f_refine_h = torch.matmul(f_m, matrix_hm) + f_h

        f_refine_h_T = torch.transpose(f_refine_h, 2, 1)
        matrix_mh = torch.matmul(f_refine_h_T, f_m)
        l2_h = torch.norm(matrix_mh)
        matrix_mh = torch.tanh(matrix_mh / l2_h)
        f_refine_m = torch.matmul(f_refine_h, matrix_mh) + f_m

        return f_refine_h.view(-1, c, h, w), f_refine_m.view(-1, c, h, w)

    def forward(self, f_h, f_m):
        f_h = self.transferconv_h(f_h)
        f_m = self.transferconv_m(f_m)
        f_refine_h, f_refine_m = self.reconstructing_procedure(f_h, f_m)

        return f_refine_h, f_refine_m
        
        
        
        

        
        
