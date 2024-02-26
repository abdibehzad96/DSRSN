import torch
import torch.nn as nn
import torch.optim as optim
import snntorch as snn


#convert the residual shrinkage block to pytorch
class SoftThresholdLayer(nn.Module):
    def __init__(self, threshold_init=1.0):
        super(SoftThresholdLayer, self).__init__()
        self.threshold = nn.Parameter(torch.tensor(threshold_init))
    def forward(self, x, thres):
        return  torch.sign(x) * torch.relu(torch.abs(x) - thres)

class SRU(nn.Module):
    def __init__(self, nb_blocks,in_ch, out_ch, imsize, device, betta = 0.85, thrs = 1,n =1, kernel_size = 3, stride=1, padding=1):
        super(SRU, self).__init__()
        self.nb_blocks = nb_blocks
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.stride = stride
        #self.sigma = 4
        self.in_ch = in_ch
        self.padding = padding
        self.betta = betta # torch.rand(imsize)
        self.th = thrs
        self.imsize = imsize
        self.device = device
        # calculate the output size of each convolving layers
        self.conv1out_s = self.convcalc(self.imsize, self.kernel_size,self.padding , self.stride)
        self.conv2out_s = self.convcalc(self.conv1out_s, self.kernel_size,self.padding , self.stride)
        self.avgpool_s = self.convcalc(self.conv2out_s, self.kernel_size,self.padding , self.stride)
        
        self.BN1 = nn.BatchNorm2d(self.out_ch)
        self.IF1 = snn.Leaky(beta = self.betta, threshold= self.th)    #LIF(self.out_ch)
        self.Conv2d1 = nn.Conv2d(self.in_ch, self.out_ch, self.kernel_size, stride= self.stride, padding=1) # it should output a flatten tensor
        
        self.BN2 = nn.BatchNorm2d(self.out_ch)
        self.IF2 = snn.Leaky(beta= self.betta, threshold= self.th)      #LIF(self.out_ch)
        self.Conv2d2 = nn.Conv2d(self.out_ch, self.out_ch, self.kernel_size, stride=self.stride, padding=1)
        
        #self.AvgPool2d = nn.AvgPool2d(kernel_size= self.kernel_size, stride = self.stride , padding = self.padding)
        self.Avgpool2d = nn.AvgPool2d(kernel_size= (self.conv2out_s, self.conv2out_s*n))
        self.FC1 = nn.Linear(1,1)
        self.FC2 = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()
        self.soft_threshold = SoftThresholdLayer(threshold_init=1.0)
    def meminit(self):
        mem1 = self.IF1.init_leaky()
        mem2 = self.IF2.init_leaky()
        return mem1, mem2
    
    def forward(self, x, mem1, mem2):
        #mem_out = []
        residual = x
        #spkout = torch.empty(x.shape[0], self.out_ch, self.conv2out_s, self.conv2out_s, device= self.device)
        # for step in range(self.nb_blocks):
        residual = self.BN1(residual)
        residual, mem1 = self.IF1(residual, mem1) # leaky integrate and fire
        residual = self.Conv2d1(residual)


        residual = self.BN2(residual) # batch normalization
        residual, mem2 = self.IF2(residual, mem2) # leaky integrate and fire
        ResIF = self.Conv2d2(residual)
        #  Squeeze and Excitation layer
        scales = self.Avgpool2d(ResIF)
        feedback = self.FC1(scales)

        feedback = self.sigmoid(self.FC2(feedback))
        #multiply scales with feedback
        thres = torch.multiply(scales, feedback)
        #soft thresholding
        residual = self.soft_threshold(ResIF, thres)

        out = torch.cat((x , residual), 3) # sum layer
        return out, mem1, mem2
    
    def convcalc(self, inpt, krnl,padding, stride):
        return (inpt - krnl + 2*padding)//stride + 1
    


class DSRN(nn.Module):
    def __init__(self, nb_blocks, in_ch, out_ch,imsize,device, strides=2, kernelsize=5, padding=2):
        super(DSRN, self).__init__()
        self.nb_blocks = nb_blocks
        self.out_ch = out_ch
        self.kernel_size = kernelsize
        self.padding = padding
        self.stride = strides
        self.in_ch = in_ch
        self.imsize = imsize
        self.betta = 0.10 #torch.rand(self.conv1_s)
        self.th = 10
        self.device = device
        self.out_s = (self.imsize - self.kernel_size + 2*self.padding)//self.stride + 1
        self.class_size = 10
        self.Conv2d1 = nn.Conv2d(self.in_ch, self.out_ch, self.kernel_size, stride= self.stride, padding= self.padding)
        self.IFC0 = snn.Leaky(beta = self.betta, threshold= self.th, learn_beta=True)    #LIF(self.out_ch)
        self.conv2d2 = nn.Conv2d(self.out_ch, self.out_ch, self.kernel_size, stride= self.stride, padding= self.padding)
        self.conv1_s = (self.imsize - self.kernel_size + 2*self.padding)//(self.stride) + 1
        self.conv2_s = (self.conv1_s - self.kernel_size + 2*self.padding)//(self.stride) + 1


        
        self.IFc = snn.Leaky(beta = self.betta, threshold= self.th, learn_beta=True)    #LIF(self.out_ch)
        self.SRBU1 = SRU(self.nb_blocks//4, self.out_ch, self.out_ch, self.conv2_s, self.device, betta=self.betta, thrs=self.th,n=1) # no downsampling   

        self.maxpool1 = nn.MaxPool2d(3,2,1)
        self.maxpool1_s = (self.conv2_s*2 -1)//2 + 1
        self.SRBU2 = SRU(self.nb_blocks//4, self.out_ch, self.out_ch, self.maxpool1_s, self.device, betta=self.betta, thrs=self.th,n=2) # no downsampling

        self.maxpool2 = nn.MaxPool2d(3,2,1)
        self.maxpool2_s = (self.maxpool1_s*2 -1)//2 + 1
        self.SRBU3 = SRU(self.nb_blocks//4, self.out_ch, self.out_ch, self.maxpool2_s, self.device, betta=self.betta, thrs=self.th,n=4) # downsampling by 2

        self.maxpool3 = nn.MaxPool2d(3,2,1)
        self.maxpool3_s = (self.maxpool2_s*2 -1)//2 + 1
        self.SRBU4 = SRU(self.nb_blocks//4, self.out_ch, self.out_ch, self.maxpool3_s, self.device, betta=self.betta, thrs=self.th,n=8) # downsampling by 2
        #self.SRBU5 = SRU(self.nb_blocks//4, self.out_ch, self.out_ch, self.conv1_s, self.device)
        #self.SRBU6 = SRU(self.nb_blocks//4, self.out_ch, self.out_ch, self.conv1_s, self.device)
        self.maxpool4 = nn.MaxPool2d(8,4,1)
        self.maxpool4_s = (self.maxpool3_s - 6)//4 + 1
        self.flatten = nn.Flatten(1)
        self.BN1 = nn.BatchNorm1d(32768)
        self.IF1 = snn.Leaky(beta = self.betta, threshold= self.th)    #LIF(self.out_ch)
        self.FC1 = nn.Linear(32768, self.out_s*self.nb_blocks)
        self.FC2 = nn.Linear(self.out_s*self.nb_blocks, self.class_size)
        self.IF2 = snn.Leaky(beta= self.betta, threshold= self.th)      #LIF(self.out_ch)
        #self.softmax = nn.Softmax(dim =1)



    def forward(self, x):
        out = torch.empty(x.shape[0], 0,self.class_size, device= self.device)
        memc0 = self.IFC0.init_leaky()
        memc = self.IFc.init_leaky()
        mem1= self.IF1.init_leaky()
        mem2 = self.IF2.init_leaky()
        mem11, mem12 = self.SRBU1.meminit()
        mem21, mem22 = self.SRBU2.meminit()
        mem31, mem32 = self.SRBU3.meminit()
        mem41, mem42 = self.SRBU4.meminit()

        for step in range(self.nb_blocks):
            next_l = self.Conv2d1(x)
            next_l, memc = self.IFc(next_l, memc)
            next_l = self.conv2d2(next_l)
            next_l, memc0 = self.IFC0(next_l, memc0)
            next_l, mem11, mem12 = self.SRBU1(next_l, mem11, mem12)
            #next_l = self.maxpool1(next_l)
            next_l, mem21, mem22 = self.SRBU2(next_l, mem21, mem22)
            #next_l = self.maxpool2(next_l)
            next_l, mem31, mem32 = self.SRBU3(next_l, mem31, mem32)
            #next_l = self.maxpool3(next_l)
            next_l, mem41, mem42 = self.SRBU4(next_l, mem41, mem42)
            #next_l, mem41, mem42 = self.SRBU4(next_l, mem41, mem42)
            #next_l, mem41, mem42 = self.SRBU4(next_l, mem41, mem42)
            next_l = self.flatten(next_l)
            next_l = self.BN1(next_l)
            next_l, mem1 = self.IF1(next_l, mem1)
            next_l = self.FC1(next_l)
            next_l = self.FC2(next_l)
            next_l, mem2 = self.IF2(next_l, mem2)
            out = torch.cat((out, next_l.unsqueeze(1)), 1)
        return out