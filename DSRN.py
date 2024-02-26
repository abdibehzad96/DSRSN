import torch
import torch.nn as nn
import torch.optim as optim
import snntorch as snn
import math

#convert the residual shrinkage block to pytorch
def SoftThresholdLayer(x , thres):
        return  torch.sign(x) * torch.relu(torch.abs(x) - thres)

class ResnetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cellsize, device, betta = 0.88, thrs = 0.99, kernel_size = 3, stride=1, padding=1):
        super(ResnetBlock, self).__init__()

        self.Conv2d = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False)
        self.conv_s = (cellsize - kernel_size + 2*padding)//stride + 1
        self.BN = nn.BatchNorm2d(out_ch)
        self.IF = snn.Leaky(beta = betta, threshold = thrs, reset_mechanism = "zero")    #LIF(out_ch
        self.sigmoind = nn.Sigmoid()
    
    def forward(self, x, mem):
        x = self.BN(x)
        x, mem = self.IF(x, mem)
        x = self.Conv2d(x)
        return x, mem


class SRU(nn.Module):
    def __init__(self, catORadd ,in_ch, out_ch, imsize, device, betta = 0.88, thrs = 0.99, kernel_size = 3, stride=1, padding=1):
        super(SRU, self).__init__()
        self.action = catORadd # 0 for add, 1 for cat
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_ch = in_ch
        self.padding = padding
        self.betta = betta # torch.rand(imsize)
        self.th = thrs
        self.imsize = imsize
        self.device = device
        self.ks = (self.imsize - self.kernel_size + 2*self.padding)//(self.stride) + 1
        self.Res1 = ResnetBlock(self.in_ch, self.out_ch, self.imsize, self.device, betta=self.betta, thrs=self.th)
        self.Res1_s = self.Res1.conv_s
        self.Res2 = ResnetBlock(self.out_ch, self.out_ch, self.Res1_s, self.device, betta=self.betta, thrs=self.th)
        self.Res2_s = self.Res2.conv_s
        self.out_s = self.Res2_s
        self.Res3 = ResnetBlock(self.out_ch, self.out_ch, self.Res2_s, self.device, betta=self.betta, thrs=self.th)
        self.Res3_s = self.Res3.conv_s

        # self.Res4 = ResnetBlock(self.out_ch, self.out_ch, self.Res3_s, self.device, betta=self.betta, thrs=self.th)
        # self.Res4_s = self.Res4.conv_s
        
        #self.AvgPool2d = nn.AvgPool2d(kernel_size= self.kernel_size, stride = self.stride , padding = self.padding)
        self.Avgpool2d = nn.AvgPool2d(kernel_size= self.Res2_s)
        self.flat= nn.Flatten(1)
        self.FC1 = nn.Linear(self.out_ch, self.out_ch*4)
        self.FC2 = nn.Linear(self.out_ch*4, self.out_ch)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
    
    def forward(self, x, mem):
        residual = x
        residual, mem0 = self.Res1(residual, mem[0])
        residual, mem1 = self.Res2(residual, mem[1])
 
        #  Squeeze and Excitation layer
        scales = self.Avgpool2d(residual)
        scales = self.flat(scales)
        feedback = self.FC1(scales)
        feedback = self.FC2(feedback)
        
        # multiply scales with feedback
        thres = torch.multiply(scales, feedback).unsqueeze(-1).unsqueeze(-1)
        #soft thresholding
        residual = SoftThresholdLayer(residual, thres)
        #residual = torch.multiply(x, self.sigmoid(100*(x -thres))-0.5)
        if self.action == 0:
            out = torch.add(x ,residual) # sum layer
        else:
            out = torch.cat((x, residual), 3)  # concat layer
        return out, [mem0, mem1] #, mem3]

class DSRN(nn.Module):
    def __init__(self, nb_blocks, in_ch, out_ch,imsize,device, betta, thresh, kernelsize=3, strides=2, padding=1):
        super(DSRN, self).__init__()
        self.nb_blocks = nb_blocks
        self.out_ch = out_ch
        self.kernel_size = kernelsize
        self.padding = padding
        self.stride = strides
        self.in_ch = in_ch
        self.imsize = imsize
        self.betta = betta #torch.rand(self.conv1_s)
        self.th = thresh
        self.device = device
        self.class_size = 10
        self.Conv2d1 = nn.Conv2d(self.in_ch, self.out_ch, self.kernel_size, stride= self.stride, padding= self.padding, bias=False)
        self.conv1_s = (self.imsize[0] - self.kernel_size + 2*self.padding)//(self.stride) + 1
        self.BN0 = nn.BatchNorm2d(self.out_ch)
        # self.FC1 = nn.Linear(self.conv1_s, self.conv1_s)
        self.IFC1 = snn.Leaky(beta = self.betta, threshold=thresh, reset_mechanism="zero")    #LIF(self.out_ch)
        self.dropout1 = nn.Dropout(p=0.1)
        # self.dropout2 = nn.Dropout(p=0.1)

        self.conv2d2 = nn.Conv2d(self.out_ch, self.out_ch, self.kernel_size, stride= self.stride, padding= self.padding, bias=False)
        self.conv2_s = (self.conv1_s - self.kernel_size + 2*self.padding)//(self.stride) + 1
        self.FC2 = nn.Linear(self.conv2_s, self.conv2_s)
        self.BN1 = nn.BatchNorm2d(self.out_ch)
        self.IFC2 = snn.Leaky(beta = self.betta, learn_threshold=True)    #LIF(self.out_ch)
        # self.conv2_s = self.conv1_s
        self.SRBU1 = SRU(0, self.out_ch, self.out_ch, self.conv2_s, self.device, 
                         betta=self.betta, thrs=self.th, kernel_size= self.kernel_size, stride=self.stride+1, padding = self.padding) # no 
        self.SRBU1_s = self.SRBU1.out_s
        # self.maxpool1 = nn.MaxPool2d(2,2,0)
        # self.maxpool1_s = (self.SRBU1_s-2)//2 + 1 #, (self.SRBU1_s[1]-2)//2 + 1]

        self.SRBU2 = SRU(0, self.out_ch, self.out_ch, self.SRBU1_s, self.device, 
                         betta=self.betta, thrs=self.th, kernel_size= self.kernel_size, stride=self.stride, padding = self.padding)
        self.SRBU2_s = self.SRBU2.out_s
        # self.maxpool2 = nn.MaxPool2d((1,2),(1,2),0)
        # self.maxpool2_s = [(self.maxpool1_s[0]-1)//1 + 1, (self.maxpool1_s[1]-2)//2 + 1]


        self.SRBU3 = SRU(0, self.out_ch, self.out_ch, self.SRBU2_s, self.device, 
                         betta=self.betta, thrs=self.th, kernel_size= self.kernel_size, stride=self.stride, padding = self.padding)
        self.SRBU3_s = self.SRBU3.out_s
        #self.maxpool3 = nn.MaxPool2d(3,2,1)
        #self.maxpool3_s = (self.SRBU3_s -1)//2 + 1 #, (self.SRBU3_s[1] -1)//2 + 1]

        self.SRBU4 = SRU(0, self.out_ch, self.out_ch, self.SRBU3_s, self.device, 
                         betta=self.betta, thrs=self.th, kernel_size= self.kernel_size, stride=self.stride, padding = self.padding)
        self.SRBU4_s = self.SRBU4.out_s
        #self.SRBU4 = SRU(self.nb_blocks//4, self.out_ch, self.out_ch, self.conv1_s, self.device)
        #self.SRBU5 = SRU(self.nb_blocks//4, self.out_ch, self.out_ch, self.conv1_s, self.device)

        self.BN2 = nn.BatchNorm2d(self.out_ch)
        self.flatten = nn.Flatten(1)
        self.flatten_s = self.out_ch*self.SRBU4_s**2
        self.FC3 = nn.Linear(self.flatten_s, self.out_ch*self.SRBU4_s)
        self.IFC3 = snn.Leaky(beta = self.betta, threshold=thresh, reset_mechanism="zero")    #LIF(self.out_ch)
        
        self.FC4 = nn.Linear(self.out_ch*self.SRBU4_s, self.class_size)
        self.IFC4 = snn.Leaky(beta= self.betta, threshold=thresh, reset_mechanism="zero")      #LIF(self.out_ch)
        #self.softmax = nn.Softmax(dim =1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # memc1 = self.IFC1.init_leaky()
        memc2 = self.IFC2.init_leaky()
        memc3 = self.IFC3.init_leaky()
        memc4 = self.IFC4.init_leaky()
        mem1 = [self.SRBU1.Res1.IF.init_leaky(), self.SRBU1.Res2.IF.init_leaky()]#, self.SRBU1.Res3.IF.init_leaky()]#, self.SRBU1.Res4.IF.init_leaky()]
        mem2 = [self.SRBU2.Res1.IF.init_leaky(), self.SRBU2.Res2.IF.init_leaky()] #, self.SRBU2.Res3.IF.init_leaky()]#, self.SRBU2.Res4.IF.init_leaky()] 
        mem3 = [self.SRBU3.Res1.IF.init_leaky(), self.SRBU3.Res2.IF.init_leaky()] #, self.SRBU3.Res3.IF.init_leaky()]#, self.SRBU3.Res4.IF.init_leaky()]
        mem4 = [self.SRBU4.Res1.IF.init_leaky(), self.SRBU4.Res2.IF.init_leaky() ]#, self.SRBU4.Res3.IF.init_leaky()]#, self.SRBU4.Res4.IF.init_leaky()]
        out = []
        for step in range(self.nb_blocks):
            next_l = self.Conv2d1(x)
            next_l = self.BN0(next_l)
            # next_l = self.FC1(next_l)
            #next_l = self.dropout1(next_l)
            # next_l = self.sigmoid(next_l)
            # next_l, memc1 = self.IFC1(next_l, memc1)
            next_l = self.conv2d2(next_l)
            next_l = self.FC2(next_l)
            next_l = self.BN1(next_l)
            next_l, memc2 = self.IFC2(next_l, memc2)
            next_l, mem1 = self.SRBU1(next_l, mem1)
            #next_l = self.maxpool1(next_l)
            next_l, mem2 = self.SRBU2(next_l, mem2)
            #next_l = self.maxpool2(next_l)

            next_l, mem3 = self.SRBU3(next_l, mem3)
            # next_l = self.maxpool3(next_l)

            next_l, mem4= self.SRBU4(next_l, mem4)

            # next_l = self.BN2(next_l)
            next_l = self.flatten(next_l)
            # next_l = self.sigmoid(next_l)
            next_l, memc3 = self.IFC3(next_l, memc3)
            next_l = self.FC3(next_l)
            next_l = self.FC4(next_l)
            # next_l = self.sigmoid(next_l)
            # next_l = torch.mul(next_l, 2.2)
            next_l, memc4 = self.IFC4(next_l, memc4)
            out.append(next_l)
        return torch.stack(out)
