import argparse
import os
import torch.utils.data
from torchvision.transforms import ToTensor, ToPILImage, Resize,CenterCrop
import skimage.measure
from FinalModel import RAN0
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--model_name', default='modelS32.pth', type=str, help='generator model epoch name')

opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
MODEL_NAME = opt.model_name


Testdir = 'Testimgs'
testimages = os.listdir(Testdir)

model = RAN0()
model = torch.nn.DataParallel(model,device_ids=[0,1])
model.cuda()
model.load_state_dict(torch.load('RARNGAN6layer.pth'))
model.eval()



for name in testimages:

    imagename = os.path.join(Testdir,name)


    img = Image.open(imagename)

    big = ToTensor()(img)
    big = torch.unsqueeze(big, dim=0)
    big = big.cuda()
    A = model.module.forwardonce(big)



    lqimg = ToTensor()(img)
    lqimg = torch.unsqueeze(lqimg, dim=0)
    lqimg = lqimg.cuda()
    result = model.module.forwardTest(lqimg,A)

    result = result.data[0].permute(1, 2, 0).cpu().numpy()
    result[result < 0.0] = 0.0
    result[result > 1.0] = 1.0



    skimage.io.imsave('Out/enhance_%s' % name, result)
