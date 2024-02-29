import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbones import Conv_4, ResNet
import math
from models.module.CLFR import CLFR
from .module.CSFA import CSFA


class C2_Net(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.shots = [self.args.train_shot, self.args.train_query_shot]
        self.way = self.args.train_way
        self.resnet = self.args.resnet

        if self.resnet:
            self.num_channel = 640
            self.dim = 640 * 25
            self.feature_extractor = ResNet.resnet12(drop=True)
        else:
            self.num_channel = 64
            self.dim = 64 * 25
            self.feature_extractor = Conv_4.BackBone(self.num_channel)

        self.clfr = CLFR(self.resnet, self.num_channel)
        self.csfa_h = CSFA(self.resnet, self.num_channel)
        self.csfa_m = CSFA(self.resnet, self.num_channel)
        self.scale_h = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        self.scale_m = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)

    def get_feature_vector(self, inp):
        f_h, f_m = self.feature_extractor(inp)

        return f_h, f_m

    def get_neg_l2_dist(self, inp, way, shot, query_shot):

        f_h, f_m = self.get_feature_vector(inp)
        f_refine_h, f_refine_m = self.clfr(f_h, f_m)

        centroid_h, query_h = self.csfa_h(f_refine_h, way, shot)
        centroid_m, query_m = self.csfa_m(f_refine_m, way, shot)

        l2_dist_h = torch.sum(torch.pow(centroid_h - query_h, 2), dim=-1).transpose(0, 1)
        neg_l2_dist_h = l2_dist_h.neg()

        l2_dist_m = torch.sum(torch.pow(centroid_m - query_m, 2), dim=-1).transpose(0, 1)
        neg_l2_dist_m = l2_dist_m.neg()

        return neg_l2_dist_h, neg_l2_dist_m

    def meta_test(self, inp, way, shot, query_shot):

        neg_l2_dist_h, neg_l2_dist_m = self.get_neg_l2_dist(inp=inp,
                                                            way=way,
                                                            shot=shot,
                                                            query_shot=query_shot
                                                            )
        neg_l2_dist_all = neg_l2_dist_h + neg_l2_dist_m
        _, max_index = torch.max(neg_l2_dist_all, 1)

        return max_index

    def forward(self, inp):

        neg_l2_dist_h, neg_l2_dist_m = self.get_neg_l2_dist(inp=inp,
                                                            way=self.way,
                                                            shot=self.shots[0],
                                                            query_shot=self.shots[1]
                                                            )
        logits_h = neg_l2_dist_h / self.dim * self.scale_h
        logits_m = neg_l2_dist_m / self.dim * self.scale_m

        log_prediction_h = F.log_softmax(logits_h, dim=1)
        log_prediction_m = F.log_softmax(logits_m, dim=1)

        return log_prediction_h, log_prediction_m
