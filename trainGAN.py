import argparse
import pandas as pd
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from Discriminator import DiscriminatorS
from data_utilize import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from FinalModel import RAN0
from torch.optim import lr_scheduler
from loss import GeneratorLoss
import ssim

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--num_epochs', default=200, type=int, help='train epoch number')

opt = parser.parse_args()

NUM_EPOCHS = opt.num_epochs

train_set = TrainDatasetFromFolder('/dped/iphone/training_data/iphone','/dped/iphone/training_data/canon')
val_set = ValDatasetFromFolder('/dped/iphone/test_data/patches/iphone','/dped/iphone/test_data/patches/canon')
train_loader = DataLoader(dataset=train_set, num_workers=8, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

netG = RAN0()
netG = torch.nn.DataParallel(netG,device_ids=[0,1])

netD = DiscriminatorS()
netD = torch.nn.DataParallel(netD,device_ids=[0,1])
print('# generator parameters:', sum(param.numel() for param in netG.parameters()))

adv_criterion = torch.nn.BCELoss()
generator_criterion = GeneratorLoss()

if torch.cuda.is_available():
    netG.cuda()
    adv_criterion.cuda()
    netD.cuda()
    generator_criterion.cuda()

optimizerG = optim.Adam(netG.parameters(),lr=5e-4)
optimizerD = optim.Adam(netD.parameters(),lr=1e-4)

results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}
Maxapsnr = 0
Maxssim = 0
NetLOSS = {'NetGLoss':[],'NetDLoss':[],'NetGADVLoss':[]}
NetDypL = {'P1':[],'P2':[],'P3':[],'P4':[],'P5':[]}

scheduler = lr_scheduler.StepLR(optimizerG,step_size=5,gamma=0.95)
ss = 0
for epoch in range(1, NUM_EPOCHS + 1):
    train_bar = tqdm(train_loader)
    running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0,'gadv_loss':0}
    running_resultdyp = {'P1':0,'P2':0,'P3':0,'P4':0,'P5':0}

    scheduler.step()
    netG.train()
    netD.train()

    k = 0
    epochdloss = 0
    epochgloss = 0
    for data, target in train_bar:

        batch_size = data.size(0)
        running_results['batch_sizes'] += batch_size

        ############################
        # (1) Update D network: maximize D(x)-1-D(G(z))
        ###########################
        real_img = Variable(target)
        if torch.cuda.is_available():
            real_img = real_img.cuda()

        z = Variable(data)
        if torch.cuda.is_available():
            z = z.cuda()

        fake_img = netG(z)

        netD.zero_grad()
        real_out = netD(real_img)
        fake_out = netD(fake_img)
        targetreal = Variable(torch.rand(batch_size,1)*0.5 + 0.7).cuda()
        targetfake = Variable(torch.rand(batch_size,1)*0.3).cuda()

        d_loss = adv_criterion(real_out,targetreal)+adv_criterion(fake_out, targetfake)

        d_loss.backward(retain_graph=True)

        running_results['d_loss'] += d_loss.data[0].cpu().numpy()
        optimizerD.step()

        ############################
        #  Update G network)
        ###########################
        netG.zero_grad()
        ones_const = Variable(torch.ones(batch_size, 1))
        ones_const = ones_const.cuda()

        fake_img = netG(z)
        generator_adversarial_loss = adv_criterion(netD(fake_img), ones_const)
        lossmain,ploss = generator_criterion(fake_img,real_img)

        totalloss = 0.0005*generator_adversarial_loss+lossmain



        totalloss.backward()
        optimizerG.step()


        train_bar.set_description(desc='%f' % (totalloss.data[0]))
        running_results['g_loss']+=totalloss.data[0].cpu().numpy()
        running_results['gadv_loss'] += generator_adversarial_loss.data[0].cpu().numpy()

        running_resultdyp['P1'] += ploss[0].data[0].cpu().numpy()
        running_resultdyp['P2'] += ploss[1].data[0].cpu().numpy()
        running_resultdyp['P3'] += ploss[2].data[0].cpu().numpy()
        running_resultdyp['P4'] += ploss[3].data[0].cpu().numpy()
        running_resultdyp['P5'] += ploss[4].data[0].cpu().numpy()

    ############################
    #  evluation method
    ###########################
    NetLOSS['NetGLoss'].append(running_results['g_loss']/running_results['batch_sizes'])
    NetLOSS['NetGADVLoss'].append(running_results['gadv_loss'] / running_results['batch_sizes'])
    NetLOSS['NetDLoss'].append(running_results['d_loss']/running_results['batch_sizes'])

    NetDypL['P1'].append(running_resultdyp['P1']/running_results['batch_sizes'])
    NetDypL['P2'].append(running_resultdyp['P2'] / running_results['batch_sizes'])
    NetDypL['P3'].append(running_resultdyp['P3'] / running_results['batch_sizes'])
    NetDypL['P4'].append(running_resultdyp['P4'] / running_results['batch_sizes'])
    NetDypL['P5'].append(running_resultdyp['P5'] / running_results['batch_sizes'])


    ss +=1
    netG.eval()
    val_bar = tqdm(val_loader)
    valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
    result = 0
    result1 = 0
    for val_lr, val_hr in val_bar:
        batch_size = val_lr.size(0)
        valing_results['batch_sizes'] += batch_size
        lr = Variable(val_lr, volatile=True)
        hr = Variable(val_hr, volatile=True)
        if torch.cuda.is_available():
            lr = lr.cuda()
            hr = hr.cuda()
        netG.zero_grad()
        sr = netG(lr)

        srdata = sr.data.permute(0, 2, 3, 1).cpu().numpy()
        hrdata = hr.data.permute(0, 2, 3, 1).cpu().numpy()

        #srdata[srdata < 0.0] = 0.0
        #srdata[srdata > 1.0] = 1.0
        #srdata = skimage.color.rgb2ycbcr(srdata).astype('uint8')
        #hrdata = skimage.color.rgb2ycbcr(hrdata).astype('uint8')

        bb = ssim.psnr_calc(srdata, hrdata)
        cc = ssim.MultiScaleSSIM(srdata * 255, hrdata * 255)
        result += bb
        result1 += cc

    valing_results['psnr'] = result/valing_results['batch_sizes']
    valing_results['ssim'] = result1/valing_results['batch_sizes']
    # save model parameters
    if valing_results['ssim'] > Maxssim:
        torch.save(netG.state_dict(), 'epochs/RARNGAN6layer.pth')
        torch.save(netD.state_dict(), "epochs/RARN80GAN-d.pth")
        Maxssim = valing_results['ssim']
    results['psnr'].append(valing_results['psnr'])
    results['ssim'].append(valing_results['ssim'])
    results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
    results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])

    out_path = 'statistics/'
    data_frame = pd.DataFrame(
        data={'g_loss': results['g_loss'],'d_loss': results['d_loss'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
        index=range(1, ss + 1))
    data_frame.to_csv(out_path + 'RARNGAN6layer.csv', index_label='Epoch')

    data_frame2 = pd.DataFrame(
        data={'NetGLoss': NetLOSS['NetGLoss'],'NetDLoss': NetLOSS['NetDLoss'],'NetGADVLoss': NetLOSS['NetGADVLoss']},
        index=range(1, epoch + 1))
    data_frame2.to_csv(out_path + 'rarnloss2.csv', index_label='Epoch')


    data_frame3 = pd.DataFrame(
        data={'P1': NetDypL['P1'],'P2': NetDypL['P2'],'P3': NetDypL['P3'],'P4':NetDypL['P4'],'P5':NetDypL['P5']},
        index=range(1, epoch + 1))
    data_frame3.to_csv(out_path + 'dyploss2.csv', index_label='Epoch')
