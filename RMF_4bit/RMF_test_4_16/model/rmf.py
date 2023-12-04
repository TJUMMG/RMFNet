# @Author: Xin Wen

import torch
import torch.nn as nn
from model import common
from torch.nn import init

def make_model(args, parent=False):
    return RMF(args)




class Concate(nn.Module):
    def __init__(self):
        super(Concate, self).__init__()

    def forward(self, x, y):
        return torch.cat((x,y),1)

class GroupFusion(nn.Module):
    def __init__(self):
        super(GroupFusion, self).__init__()

    def forward(self, x, Xtime, y, Ytime):
        feature = 32
        x1 = x[:, 0:int(feature * Xtime * 1), :, :]
        x2 = x[:, int(feature * Xtime * 1):int(feature * Xtime * 2), :, :]
        x3 = x[:, int(feature * Xtime * 2):int(feature * Xtime * 3), :, :]
        x4 = x[:, int(feature * Xtime * 3):int(feature * Xtime * 4), :, :]

        y1 = y[:, 0:int(feature * Ytime * 1), :, :]
        y2 = y[:, int(feature * Ytime * 1):int(feature * Ytime * 2), :, :]
        y3 = y[:, int(feature * Ytime * 2):int(feature * Ytime * 3), :, :]
        y4 = y[:, int(feature * Ytime * 3):int(feature * Ytime * 4), :, :]

        fuse1 = torch.cat((x1, y1), 1)

        xx2 = x1 + x2
        yy2 = y1 + y2
        fuse2 = torch.cat((xx2, yy2), 1)

        xx3 = xx2 + x3
        yy3 = yy2 + y3
        fuse3 = torch.cat((xx3, yy3), 1)

        xx4 = xx3 + x4
        yy4 = yy3 + y4
        fuse4 = torch.cat((xx4, yy4), 1)

        return fuse1, fuse2, fuse3, fuse4

class Concate4(nn.Module):
    def __init__(self):
        super(Concate4, self).__init__()

    def forward(self, x, y, z, o):
        return torch.cat((x,y,z,o),1)




class Basic_Block(nn.Module):
    def __init__(self, conv, in_feat, out_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True)):
            super(Basic_Block, self).__init__()
            m = []
            m.append(conv(in_feat,out_feat,kernel_size,bias=bias))
            if bn: m.append(nn.BatchNorm2d(out_feat))
            if act is not None: m.append(act)
            self.body = nn.Sequential(*m)

        
    def forward(self, x):
        return self.body(x)



class RMF(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(RMF, self).__init__()


        b_blocks = args.b_blocks
        a_blocks = args.a_blocks
        n_feats = args.n_feats
        kernel_size = 3
        if args.act =='lrelu':
            self.act = nn.LeakyReLU(0.2, inplace=False)
        else:
            self.act = nn.ReLU(True)


        inputnumber = args.n_colors
        bn  = args.bn

        self.fusion = Concate()
        self.fusion4 = Concate4()
        self.groupCat = GroupFusion()

        self.upsampling = nn.PixelShuffle(2)

        m_lrhead1 = [conv(inputnumber*4, n_feats*2, 3)]
        m_lrbody1 = [Basic_Block(conv, n_feats*2, n_feats*2, kernel_size, bn=bn, act=self.act) for _ in range(b_blocks)]
        m_lrbody11 = [conv(n_feats*2, n_feats*2, 3)]

        m_lrhead2 = [conv(inputnumber*16, n_feats*4, 3)]
        m_lrbody2 = [Basic_Block(conv, n_feats*4, n_feats*4, kernel_size, bn=bn, act=self.act) for _ in range(b_blocks)]
        m_lrbody21 = [conv(n_feats*4, n_feats*4, 3)]


        m_lrhead3 = [conv(inputnumber*64, n_feats*8, 3)]
        m_lrbody3 = [Basic_Block(conv, n_feats*8, n_feats*8, kernel_size, bn=bn, act=self.act) for _ in range(b_blocks)]
        m_lrbody31 = [conv(n_feats*8, n_feats*8, 3)]

  
       
        # define head module
        m_head = [conv(args.n_colors, n_feats, 3)]


        # define body module

        m_lrtail1 = [Basic_Block(conv, n_feats*2, n_feats*2, kernel_size, bn=bn, act=self.act)]
        m_lrtail2 = [Basic_Block(conv, n_feats*4, n_feats*4, kernel_size, bn=bn, act=self.act)]
        m_lrtail3 = [Basic_Block(conv, n_feats*8, n_feats*8, kernel_size, bn=bn, act=self.act)]


        m_lrhead1_0 = [Basic_Block(conv, 3*n_feats, n_feats, kernel_size, bn=bn, act=self.act)]
        m_lrhead2_0 = [Basic_Block(conv, 6*n_feats, n_feats*2, kernel_size, bn=bn, act=self.act)]
        

        m_body0 = [Basic_Block(conv, int(n_feats * 1.5), int(n_feats * 0.5), kernel_size, bn=bn, act=self.act)]
        m_body1 = [Basic_Block(conv, n_feats, n_feats,  kernel_size, bn=bn, act=self.act) for _ in range(a_blocks)]

        m_tail = [conv(n_feats, args.n_colors, kernel_size)]

        #
        m_lrsuper1_0 = [Basic_Block(conv, n_feats*2, n_feats*2, kernel_size, bn=bn, act=self.act) for _ in range(a_blocks)]
        m_lrsuper1_1 = [conv(n_feats*2, inputnumber*4, kernel_size)]

        #
        m_lrsuper2_0 = [Basic_Block(conv, n_feats*4, n_feats*4, kernel_size, bn=bn, act=self.act) for _ in range(a_blocks)]
        m_lrsuper2_1 = [conv(n_feats*4, inputnumber*16, kernel_size)]

        #
        m_lrsuper3_0 = [Basic_Block(conv, n_feats*8, n_feats*8, kernel_size, bn=bn, act=self.act) for _ in range(a_blocks)]
        m_lrsuper3_1 = [conv(n_feats*8, inputnumber*64, kernel_size)]


        #
        m_group21 = [Basic_Block(conv, int(n_feats * 1.5), int(n_feats * 0.5), kernel_size, bn=bn, act=self.act)]
        m_group22 = [Basic_Block(conv, int(n_feats * 1.5), int(n_feats * 0.5), kernel_size, bn=bn, act=self.act)]
        m_group23 = [Basic_Block(conv, int(n_feats * 1.5), int(n_feats * 0.5), kernel_size, bn=bn, act=self.act)]
        m_group24 = [Basic_Block(conv, int(n_feats * 1.5), int(n_feats * 0.5), kernel_size, bn=bn, act=self.act)]
        m_group2 = [Basic_Block(conv, n_feats * 2, n_feats * 2, 1, bn=bn, act=self.act)]
        m_group_final2 = [Basic_Block(conv, n_feats * 2, n_feats * 4, kernel_size, bn=bn, act=self.act)]
        #
        m_group11 = [Basic_Block(conv, int(n_feats * 0.75), int(n_feats * 0.25), kernel_size, bn=bn, act=self.act)]
        m_group12 = [Basic_Block(conv, int(n_feats * 0.75), int(n_feats * 0.25), kernel_size, bn=bn, act=self.act)]
        m_group13 = [Basic_Block(conv, int(n_feats * 0.75), int(n_feats * 0.25), kernel_size, bn=bn, act=self.act)]
        m_group14 = [Basic_Block(conv, int(n_feats * 0.75), int(n_feats * 0.25), kernel_size, bn=bn, act=self.act)]
        m_group1 = [Basic_Block(conv, n_feats, n_feats, 1, bn=bn, act=self.act)]
        m_group_final1 = [Basic_Block(conv, n_feats, n_feats*2, kernel_size, bn=bn, act=self.act)]
        #
        m_group01 = [Basic_Block(conv, int(n_feats * 0.375), int(n_feats * 0.125), kernel_size, bn=bn, act=self.act)]
        m_group02 = [Basic_Block(conv, int(n_feats * 0.375), int(n_feats * 0.125), kernel_size, bn=bn, act=self.act)]
        m_group03 = [Basic_Block(conv, int(n_feats * 0.375), int(n_feats * 0.125), kernel_size, bn=bn, act=self.act)]
        m_group04 = [Basic_Block(conv, int(n_feats * 0.375), int(n_feats * 0.125), kernel_size, bn=bn, act=self.act)]
        m_group0 = [Basic_Block(conv, int(n_feats * 0.5), int(n_feats * 0.5), 1, bn=bn, act=self.act)]
        m_group_final0 = [Basic_Block(conv, int(n_feats * 0.5), n_feats, kernel_size, bn=bn, act=self.act)]

        self.lrhead1 = nn.Sequential(*m_lrhead1)
        self.lrbody1 = nn.Sequential(*m_lrbody1)
        self.lrtail1 = nn.Sequential(*m_lrtail1)
        self.lrhead2 = nn.Sequential(*m_lrhead2)
        self.lrbody2 = nn.Sequential(*m_lrbody2)
        self.lrtail2 = nn.Sequential(*m_lrtail2)
        self.lrhead3 = nn.Sequential(*m_lrhead3)
        self.lrbody3 = nn.Sequential(*m_lrbody3)
        self.lrtail3 = nn.Sequential(*m_lrtail3)	


        self.lrbody11 = nn.Sequential(*m_lrbody11)
        self.lrbody21 = nn.Sequential(*m_lrbody21)
        self.lrbody31 = nn.Sequential(*m_lrbody31)



        self.lrhead1_0 = nn.Sequential(*m_lrhead1_0)
        self.lrhead2_0 = nn.Sequential(*m_lrhead2_0)	


        self.head = nn.Sequential(*m_head)
        self.body0 = nn.Sequential(*m_body0)
        self.body1 = nn.Sequential(*m_body1)
        self.tail = nn.Sequential(*m_tail)

        #
        self.lrsuper1_0 = nn.Sequential(*m_lrsuper1_0)
        self.lrsuper1_1 = nn.Sequential(*m_lrsuper1_1)
        #
        self.lrsuper2_0 = nn.Sequential(*m_lrsuper2_0)
        self.lrsuper2_1 = nn.Sequential(*m_lrsuper2_1)
        #
        self.lrsuper3_0 = nn.Sequential(*m_lrsuper3_0)
        self.lrsuper3_1 = nn.Sequential(*m_lrsuper3_1)


        #
        self.group21 = nn.Sequential(*m_group21)
        self.group22 = nn.Sequential(*m_group22)
        self.group23 = nn.Sequential(*m_group23)
        self.group24 = nn.Sequential(*m_group24)
        self.group2 = nn.Sequential(*m_group2)
        self.group_final2 = nn.Sequential(*m_group_final2)
        #
        self.group11 = nn.Sequential(*m_group11)
        self.group12 = nn.Sequential(*m_group12)
        self.group13 = nn.Sequential(*m_group13)
        self.group14 = nn.Sequential(*m_group14)
        self.group1 = nn.Sequential(*m_group1)
        self.group_final1 = nn.Sequential(*m_group_final1)
        #
        self.group01 = nn.Sequential(*m_group01)
        self.group02 = nn.Sequential(*m_group02)
        self.group03 = nn.Sequential(*m_group03)
        self.group04 = nn.Sequential(*m_group04)
        self.group0 = nn.Sequential(*m_group0)
        self.group_final0 = nn.Sequential(*m_group_final0)


        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)


    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)


    def forward(self, input_m):
        x, y = input_m

        g1 = common.DownSamplingShuffle(x,2)
        g2 = common.DownSamplingShuffle(g1,2)
        g3 = common.DownSamplingShuffle(g2,2)


        t1 = common.DownSamplingShuffle(y, 2)
        t2 = common.DownSamplingShuffle(t1, 2)
        t3 = common.DownSamplingShuffle(t2, 2)



        g3_input = g3
        g3 = self.act(self.lrhead3(g3))
        g3 = self.lrbody31(self.lrbody3(g3))+g3
        g3 = self.lrtail3(g3)

        g3_s = self.lrsuper3_0(g3)
        g3_s = self.lrsuper3_1(g3_s)+g3_input

        g3 = self.upsampling(g3)



        g2_input = g2
        g2 = self.act(self.lrhead2(g2))

        # DDGF
        g2_all = self.lrhead2_0(self.fusion(g2, g3))

        fuse21, fuse22, fuse23, fuse24 = self.groupCat(g2, 1, g3, 0.5)
        fuse21 = self.group21(fuse21)
        fuse22 = self.group22(fuse22)
        fuse23 = self.group23(fuse23)
        fuse24 = self.group24(fuse24)
        g2_group = self.fusion4(fuse21, fuse22, fuse23, fuse24)
        g2_group = self.group2(g2_group)

        g2 = g2_all + g2_group
        g2 = self.group_final2(g2)
        #
        g2 = self.lrbody21(self.lrbody2(g2)) + g2
        g2 = self.lrtail2(g2)

        g2_s = self.lrsuper2_0(g2)
        g2_s = self.lrsuper2_1(g2_s) + g2_input

        g2 = self.upsampling(g2)



        g1_input = g1
        g1 = self.act(self.lrhead1(g1))

        # DDGF
        g1_all = self.lrhead1_0(self.fusion(g1, g2))

        fuse11, fuse12, fuse13, fuse14 = self.groupCat(g1, 0.5, g2, 0.25)
        fuse11 = self.group11(fuse11)
        fuse12 = self.group12(fuse12)
        fuse13 = self.group13(fuse13)
        fuse14 = self.group14(fuse14)
        g1_group = self.fusion4(fuse11, fuse12, fuse13, fuse14)
        g1_group = self.group1(g1_group)

        g1 = g1_all + g1_group
        g1 = self.group_final1(g1)
        #
        g1 = self.lrbody11(self.lrbody1(g1)) + g1
        g1 = self.lrtail1(g1)

        g1_s = self.lrsuper1_0(g1)
        g1_s = self.lrsuper1_1(g1_s) + g1_input

        g1 = self.upsampling(g1)



        residual = self.act(self.head(x))

        # DDGF
        residual_all = self.fusion(g1, residual)
        residual_all = self.body0(residual_all)

        fuse01, fuse02, fuse03, fuse04 = self.groupCat(residual, 0.25, g1, 0.125)
        fuse01 = self.group01(fuse01)
        fuse02 = self.group02(fuse02)
        fuse03 = self.group03(fuse03)
        fuse04 = self.group04(fuse04)
        residual_group = self.fusion4(fuse01, fuse02, fuse03, fuse04)
        residual_group = self.group0(residual_group)

        residual = residual_all + residual_group
        residual = self.group_final0(residual)
        #
        residual = self.body1(residual)
        outcome = self.tail(residual) + x
        

        return outcome, t1, t2, t3, g1_s, g2_s, g3_s







