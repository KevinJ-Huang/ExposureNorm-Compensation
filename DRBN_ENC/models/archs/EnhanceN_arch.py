# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797
import torch
import torch.nn as nn
import torch.nn.functional as F


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)

        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        feat1 = self.convs(x)
        feat2 = self.LFF(feat1) + x
        return feat2




class DRBN(nn.Module):
    def __init__(self):
        super(DRBN, self).__init__()

        self.recur1 = DRBN_BUAtt()
        self.recur2 = DRBN_BUAtt()
        self.recur3 = DRBN_BUAtt()
        self.recur4 = DRBN_BUAtt()

    def forward(self, x_input, i):
        x = x_input

        res_g1_s1, res_g1_s2, res_g1_s4, feat_g1_s1, feat_g1_s2, feat_g1_s4, instf1, fusef1 = self.recur1(
            [0, torch.cat((x, x), 1), 0, 0, 0, 0, 0, 0])
        res_g2_s1, res_g2_s2, res_g2_s4, feat_g2_s1, feat_g2_s2, feat_g2_s4, instf2, fusef2 = self.recur2(
            [1, torch.cat((res_g1_s1, x), 1), res_g1_s1, res_g1_s2, res_g1_s4, feat_g1_s1, feat_g1_s2, feat_g1_s4])
        res_g3_s1, res_g3_s2, res_g3_s4, feat_g3_s1, feat_g3_s2, feat_g3_s4, instf3, fusef3 = self.recur3(
            [1, torch.cat((res_g2_s1, x), 1), res_g2_s1, res_g2_s2, res_g2_s4, feat_g2_s1, feat_g2_s2, feat_g2_s4])
        res_g4_s1, res_g4_s2, res_g4_s4, feat_g4_s1, feat_g4_s2, feat_g4_s4, instf4, fusef4 = self.recur4(
            [1, torch.cat((res_g3_s1, x), 1), res_g3_s1, res_g3_s2, res_g3_s4, feat_g3_s1, feat_g3_s2, feat_g3_s4])

        feature_save(fusef1,'over',i)


        return res_g4_s1, res_g4_s2, res_g4_s4, instf1, fusef1



def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)



class AttING(nn.Module):
    def __init__(self, in_channels, channels):
        super(AttING, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2_1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.instance = nn.InstanceNorm2d(channels, affine=True)
        self.interative = nn.Sequential(
            nn.Conv2d(channels*2, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Sigmoid()
        )
        self.act = nn.LeakyReLU(0.1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.contrast = stdv_channels
        self.process = nn.Sequential(nn.Conv2d(channels*2, channels//2, kernel_size=3, padding=1, bias=True),
                                     nn.LeakyReLU(0.1),
                                     nn.Conv2d(channels//2, channels*2, kernel_size=3, padding=1, bias=True),
                                     nn.Sigmoid())
        self.conv1x1 = nn.Conv2d(2*channels, channels, 1, 1, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        out_instance = self.instance(x1)
        out_identity = x1
        out1 = self.conv2_1(out_instance)
        out2 = self.conv2_2(out_identity)
        out = torch.cat((out1, out2), 1)
        xp1 = self.interative(out)*out2 + out1
        xp2 = (1-self.interative(out))*out1 + out2
        xp = torch.cat((xp1, xp2), 1)
        xp = self.process(self.contrast(xp)+self.avgpool(xp))*xp
        xp = self.conv1x1(xp)
        xout = xp

        return xout,out_instance





class DRBN_BUAtt(nn.Module):
    def __init__(self):
        super(DRBN_BUAtt, self).__init__()

        G0 = 16
        kSize = 3
        self.D = 6
        G = 8
        C = 4
        n_colors = 3

        # self.SFENet1 = nn.Conv2d(n_colors * 2, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        self.SFENet1 = AttING(n_colors * 2, G0)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        # self.SFENet2 = AttING(G0, G0)

        self.RDBs = nn.ModuleList()

        self.RDBs.append(
            RDB(growRate0=G0, growRate=G, nConvLayers=C)
        )
        self.RDBs.append(
            RDB(growRate0=G0, growRate=G, nConvLayers=C)
        )
        self.RDBs.append(
            RDB(growRate0=2 * G0, growRate=2 * G, nConvLayers=C)
        )
        self.RDBs.append(
            RDB(growRate0=2 * G0, growRate=2 * G, nConvLayers=C)
        )
        self.RDBs.append(
            RDB(growRate0=G0, growRate=G, nConvLayers=C)
        )
        self.RDBs.append(
            RDB(growRate0=G0, growRate=G, nConvLayers=C)
        )

        self.UPNet = nn.Sequential(*[
            nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.Conv2d(G0, 3, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

        self.UPNet2 = nn.Sequential(*[
            nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.Conv2d(G0, 3, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

        self.UPNet4 = nn.Sequential(*[
            nn.Conv2d(G0 * 2, G0, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.Conv2d(G0, 3, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

        self.Down1 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=2)
        self.Down2 = nn.Conv2d(G0, G0 * 2, kSize, padding=(kSize - 1) // 2, stride=2)

        self.Up1 = nn.ConvTranspose2d(G0, G0, kSize + 1, stride=2, padding=1)
        self.Up2 = nn.ConvTranspose2d(G0 * 2, G0, kSize + 1, stride=2, padding=1)

        self.Relu = nn.ReLU()
        self.Img_up = nn.Upsample(scale_factor=2, mode='bilinear')

    def part_forward(self, x):
        #
        # Stage 1
        #
        flag = x[0]
        input_x = x[1]

        prev_s1 = x[2]
        prev_s2 = x[3]
        prev_s4 = x[4]

        prev_feat_s1 = x[5]
        prev_feat_s2 = x[6]
        prev_feat_s4 = x[7]

        f_first, instancef = self.SFENet1(input_x)
        f_s1 = self.Relu(self.SFENet2(self.Relu(f_first)))
        f_s2 = self.Down1(self.RDBs[0](f_s1))
        f_s4 = self.Down2(self.RDBs[1](f_s2))

        if flag == 0:
            f_s4 = f_s4 + self.RDBs[3](self.RDBs[2](f_s4))
            f_s2 = f_s2 + self.RDBs[4](self.Up2(f_s4))
            f_s1 = f_s1 + self.RDBs[5](self.Up1(f_s2)) + f_first
        else:
            f_s4 = f_s4 + self.RDBs[3](self.RDBs[2](f_s4)) + prev_feat_s4
            f_s2 = f_s2 + self.RDBs[4](self.Up2(f_s4)) + prev_feat_s2
            f_s1 = f_s1 + self.RDBs[5](self.Up1(f_s2)) + f_first + prev_feat_s1

        res4 = self.UPNet4(f_s4)
        res2 = self.UPNet2(f_s2) + self.Img_up(res4)
        res1 = self.UPNet(f_s1) + self.Img_up(res2)

        return res1, res2, res4, f_s1, f_s2, f_s4 ,instancef, f_first

    def forward(self, x_input):
        x = x_input

        res1, res2, res4, f_s1, f_s2, f_s4, instancef, fusef = self.part_forward(x)

        return res1, res2, res4, f_s1, f_s2, f_s4, instancef, fusef
