import torch
from torch import nn
from MyVGG import vgg19
import numpy as np

import torch.nn.functional as F


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        loss_network = vgg19(pretrained=True)
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()

    def forward(self, out_images, target_images):

        batch_size = target_images.data.size(0)

        sr1,sr2,sr3,sr4,sr5 = self.loss_network(out_images)
        hr1,hr2,hr3,hr4,hr5 = self.loss_network(target_images)

        d1 = torch.abs(sr1-hr1).mean(dim=1).mean(dim=1).mean(dim=1,keepdim=True)
        d2 = torch.abs(sr2 - hr2).mean(dim=1).mean(dim=1).mean(dim=1,keepdim=True)
        d3 = torch.abs(sr3 - hr3).mean(dim=1).mean(dim=1).mean(dim=1,keepdim=True)
        d4 = torch.abs(sr4 - hr4).mean(dim=1).mean(dim=1).mean(dim=1,keepdim=True)
        d5 = torch.abs(sr5 - hr5).mean(dim=1).mean(dim=1).mean(dim=1,keepdim=True)

        diff = torch.cat((d1,d2,d3,d4,d5),1)

        weigs = F.softmax(diff,dim=1)
        weigs = weigs.view(batch_size,5).mean(dim=0,keepdim=True)

        a = weigs[:,0]
        perception_loss1 = a.data[0].cpu().numpy().tolist() * self.mse_loss(sr1,hr1)
        a = weigs[:,1]
        perception_loss2 = a.data[0].cpu().numpy().tolist() * self.mse_loss(sr2,hr2)
        a = weigs[:,2]
        perception_loss3 = a.data[0].cpu().numpy().tolist() * self.mse_loss(sr3,hr3)
        a = weigs[:,3]
        perception_loss4 = a.data[0].cpu().numpy().tolist() * self.mse_loss(sr4,hr4)
        a = weigs[:,4]
        perception_loss5 = a.data[0].cpu().numpy().tolist() * self.mse_loss(sr5,hr5)

        perception_loss = perception_loss1 + perception_loss2 + perception_loss3 + perception_loss4 + perception_loss5

        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)

        return image_loss +  perception_loss, (perception_loss1 , perception_loss2 , perception_loss3 , perception_loss4 , perception_loss5)
