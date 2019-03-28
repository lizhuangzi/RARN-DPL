import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from torch.autograd import Variable
import torch.nn.functional as F

class CAttention(nn.Module):
    def __init__(self,inchannel):
        super(CAttention, self).__init__()

        self.embady = nn.Linear(inchannel, inchannel//2, bias=False)
        self.map = nn.Tanh()
        self.decode = nn.Linear(inchannel//2, inchannel, bias=False)

    def forward(self, x):
        pooledfeatures = F.adaptive_avg_pool2d(x, 1)
        pooledfeatures = torch.squeeze(pooledfeatures, dim=3)
        pooledfeatures = torch.squeeze(pooledfeatures, dim=2)

        xemb = self.embady(pooledfeatures)
        map = self.map(xemb)
        decode = self.decode(map)

        attention = F.softmax(decode, dim=1)
        attention = torch.unsqueeze(attention, dim=2)
        attention = torch.unsqueeze(attention, dim=3)
        return attention

class DensenMutileConv2(nn.Module):
    def __init__(self,inchannel,outchannel):
        super(DensenMutileConv2, self).__init__()

        self.conv31 = nn.Sequential(
            nn.Conv2d(inchannel,40,3,stride=1,padding=1),
            nn.PReLU()
        )
        self.conv51 = nn.Sequential(
            nn.Conv2d(inchannel,14,1,stride=1,padding=0),
            nn.PReLU()
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(inchannel,10,5,stride=1,padding=2),
            nn.PReLU()
        )

        self.conv32 = nn.Sequential(
            nn.Conv2d(outchannel,outchannel//2,3,stride=1,padding=1),
            nn.PReLU()
        )
        self.conv52 = nn.Sequential(
            nn.Conv2d(outchannel,outchannel//2,1,stride=1,padding=0),
            nn.PReLU()
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(outchannel,outchannel//2,5,stride=1,padding=2),
            nn.PReLU()
        )

        self.final = nn.Conv2d(outchannel + outchannel//2, 64, 1, stride=1, padding=0)


    def forward(self, x):
        x_31 =self.conv31(x)
        x_51 = self.conv51(x)
        x_11 = self.conv11(x)

        catinput = torch.cat([x_31,x_11,x_51],1)

        x_32 = self.conv32(catinput)
        x_52 = self.conv52(catinput)
        x_12 = self.conv12(catinput)

        catinput = torch.cat([x_32, x_52, x_12], 1)

        return self.final(catinput)


class RAB(nn.Module):
    def __init__(self,inchannel,number=3):
        super(RAB, self).__init__()
        #self.inpuchannel = inchannel
        self.inchannel = inchannel

        self.conv1 = nn.Sequential(
            #nn.BatchNorm2d(self.inchannel),
            nn.ReLU(),
            nn.Conv2d(self.inchannel,self.inchannel,3,stride=1,padding=1)
        )
        self.conv2 = nn.Sequential(
            #nn.BatchNorm2d(self.inchannel),
            nn.ReLU(),
            nn.Conv2d(self.inchannel,self.inchannel,3,stride=1,padding=1)
        )

        self.leng = number
        self.attlists = nn.Sequential()
        self.testattentionlist = None
        for i in range(number):
           self.attlists.add_module('CA%d'%i,CAttention(inchannel))


    def forward(self, x):

        xin = x

        for name,layer in self.attlists.named_children():
            camaps = layer(xin)
            x1 = self.conv1(x)
            x = self.conv2(x1)

            x = x*camaps+xin

        return x



class RAN0(nn.Module):

    def __init__(self, version=1.0, num_classes=1000):
        super(RAN0, self).__init__()

        self.features = DensenMutileConv2(3,64)
        self.glcrb1 = RAB(64,number=6)
        #self.glcrb3 = RAB(64, number=3)
        #self.glcrb4 = RAB(64, number=3)

        self.reconstruction = nn.Sequential(
            nn.Conv2d(64, 64, 1, stride=1, padding=0,bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 3, 3, stride=1, padding=1,bias=False),
        )
        self.trainMode = True

    def forward(self, x):

        features = self.features(x)

        x1 = self.glcrb1(features)

        reconx = self.reconstruction(x1)

        return reconx + 1.15*x
