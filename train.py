from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import loss
from torch.nn.modules.loss import MSELoss
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import numpy as np 
from PIL import Image
import itertools
from tqdm import tqdm
import option
from dataset import ImageDataset
import os
import gc
import time
import logging
#from network import Generators, Discriminators, define_D, define_G
# Import PyTorch
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler, autocast
# Import custom modules
from model.style.dataset import CustomDataset
from network import Generators, Net, Discriminator
from utils import weights_init
from utils import LambdaLR
from gan_model import STGAN
import torchvision.models as models
from torchvision.utils import save_image

if __name__ =='__main__':
    opt = option.opt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transforms_ = [ transforms.Resize(int(opt.img_size), Image.BICUBIC), 
                transforms.RandomCrop(opt.img_size),       
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]

    dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True), 
                        batch_size=opt.batch_size, shuffle=True, num_workers=0)

    vgg = models.vgg19(pretrained=True).features
    vgg_encoder = Net(vgg)
    vgg_encoder = vgg_encoder.to(device)


    # 1) network
    netG = Generators(d_model=opt.d_model, d_embedding=opt.d_embedding, 
                               n_head=opt.n_head, dim_feedforward=opt.dim_feedforward, img_size=opt.img_size, 
                               patch_size=opt.patch_size, 
                               num_encoder_layer=opt.num_encoder_layer, num_decoder_layer=opt.num_decoder_layer,
                               dropout=opt.dropout)
    netD = Discriminator()
    netG.cuda()
    netD.cuda()
    netG.apply(weights_init)
    netD.apply(weights_init)
    # netD = netD.to(device)
    # netG = netG.to(device)

    # 2) weights initialize
    
    # 3)define loss
    gloss = nn.MSELoss().to(device)


    # optimizer & LR schedulers
    g_optim = torch.optim.Adam(netG.parameters(), lr= 0.0005,betas=(opt.beta1, 0.999))
    d_optim = torch.optim.Adam(netD.parameters(), lr= 0.0005,betas=(opt.beta1, 0.999))
    #lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(gen_optimizer, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    
    Tensor = torch.cuda.FloatTensor
    target_real = Tensor(opt.batch_size).fill_(1.0) # label for a real image
    target_fake = Tensor(opt.batch_size).fill_(0.0) # label for a fake image


    # Input, output setting`
    for epoch in range(0, opt.n_epochs):
        netG.train()


        for i, img in enumerate(dataloader):

            img_s, img_c = STGAN.set_input(img, device)

            g_optim.zero_grad()
            
            d_optim.zero_grad()

            gen_data = netG(img_c, img_s)
            save_image(gen_data, f'./checkpoint/gen_data/{epoch}_cc.jpg', nrow=4, normalize=True, scale_each=True)
            pred_fake_g = netD(gen_data)
            gen_loss = gloss(pred_fake_g,1) # 생성된 이미지랑 스타일 이미지랑 비교
            gen_loss.backward()
            g_optim.step()

            pred_real = netD(img_s)
            loss_D_real = gloss(pred_real, target_real)

            pred_fake = netD(gen_data.detach()) 
            loss_D_fake = gloss(pred_fake, target_fake)

            d_loss = (loss_D_real + loss_D_fake) *0.5
            d_loss.backward()
            d_optim.step()
            
            
            
            #L_id = F.mse_loss(cc_data, img_c)+F.mse_loss(ss_data, img_s)
            #loss_cc,loss_ss = vgg_encoder(img_c, img_s, gen_data)
            #loss_c, loss_s, loss_cc = vgg_encoder(img_c, img_s, gen_data, ss_data, cc_data)
            
            # loss_c = loss_c.mean()
            # loss_s = loss_s.mean()
            # loss_cc = loss_cc.mean()
            #loss_mse = F.mse_loss(img_c, gen_data)
            
            #loss_style = 10*loss_c + 7*loss_s + 50*L_id + 1*loss_cc

            
        print("[Epoch %d/%d] [Batch %d/%d] [g loss: %f][d loss: %f]" %(epoch, opt.n_epochs, i % len(dataloader), len(dataloader), (gen_loss),(d_loss)))