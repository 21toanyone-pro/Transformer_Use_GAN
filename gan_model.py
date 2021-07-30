import torch
from torch.functional import norm
import torch.nn as nn
import option
from network import Generators, Net, Discriminator
from torchvision.utils import save_image

#code backbone:https://github.com/ligoudaner377/font_translator_gan

class STGAN(nn.Module): 
    
    def __init__(self):

        super(STGAN, self).__init__()
        self.opt = option.opt
        self.dis_2 = True
        self.loss_names = ['G_GAN', 'G_L1', 'D_content', 'D_style']
        self.isTrain=True
        self.optimizers=[]
    

    def set_requires_grad(self, nets, requires_grad=False):

        if not isinstance(nets,list):
            nets = [nets]

        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def create_model(self, opt, device):

        self.netG =  Generators(d_model=opt.d_model, d_embedding=opt.d_embedding, 
                               n_head=opt.n_head, dim_feedforward=opt.dim_feedforward, img_size=opt.img_size, 
                               patch_size=opt.patch_size, 
                               num_encoder_layer=opt.num_encoder_layer, num_decoder_layer=opt.num_decoder_layer,
                               dropout=opt.dropout).to(device)

        self.netD_content = Discriminator(6).to(device)
        self.netD_style  = Discriminator(6).to(device)

        if self.isTrain:

            self.lambda_L1 = opt.lambda_L1
            self.criterionGAN = GANLoss(opt.gan_mode)
            self.criterionL1 = torch.nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters())
            self.optimizers.append(self.optimizer_G)
        
            if self.dis_2:
                self.lambda_style = opt.lambda_style
                self.lambda_content = opt.lambda_content
                self.optimizer_D_content = torch.optim.Adam(self.netD_content.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D_style= torch.optim.Adam(self.netD_style.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D_content)
                self.optimizers.append(self.optimizer_D_style)
            else:
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)


    def set_input(self, input, device):

        self.content_images = input['A'].to(device)

        self.style_images = input['B'].to(device)

        self.gt_images = input['gt'].to(device)

    def forward(self):
        self.generated_images = self.netG(self.content_images, self.style_images)        


    def compute_gan_loss_D(self, real_images, fake_images, netD):
        #Fake
        fake = torch.cat(fake_images, 1)
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        
        #Real
        real = torch.cat(real_images, 1)
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)

        #combine loss
        loss_D = (loss_D_fake + loss_D_real)*0.5
        return loss_D
    
    def compute_gan_loss_G(self, fake_images, netD):
        fake = torch.cat(fake_images, 1)
        pred_fake = netD(fake)
        loss_G_GAN = self.criterionGAN(pred_fake, True, True)
        return loss_G_GAN

    def backward_D(self):

        if self.dis_2: 
            self.loss_D_content = self.compute_gan_loss_D([self.content_images, self.gt_images], [self.content_images, self.generated_images], self.netD_content)
            self.loss_D_style = self.compute_gan_loss_D([self.style_images, self.gt_images], [self.style_images, self.generated_images], self.netD_style)
            self.loss_D = self.lambda_content * self.loss_D_content + self.lambda_style*self.loss_D_style
        else:
            self.loss_D = self.compute_gan_loss_D([self.content_images, self.style_images, self.gt_images], [self.content_images, self.style_images, self.generated_images], self.netD)

        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        if self.dis_2:
            self.loss_G_content = self.compute_gan_loss_G([self.content_images, self.generated_images], self.netD_content)
            self.loss_G_style = self.compute_gan_loss_G([self.style_images, self.generated_images], self.netD_style)
            self.loss_G_GAN = self.lambda_content * self.loss_G_content + self.lambda_style*self.loss_G_style
        else:
            self.loss_G_GAN = self.compute_gan_loss_G([self.content_images, self.style_images, self.generated_images], self.netD)

        #Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.generated_images, self.gt_images) * self.opt.lambda_L1
        #combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()              #compute fake images : G(A)
        #update D
        if self.dis_2:
            self.set_requires_grad([self.netD_content, self.netD_style], True)
            self.optimizer_D_content.zero_grad()
            self.optimizer_D_style.zero_grad()
            self.backward_D()
            self.optimizer_D_content.step()
            self.optimizer_D_style.step()
        else:
            self.set_requires_grad(self.netD, True) #enable backprop for D
            self.optimizer_D.zero_grad() #set D's gradients to zero
            self.backward_D() #calculate gradients for D
            self.optimizer_D.step() #update D's weights

        #update G
        if self.dis_2:
            self.set_requires_grad([self.netD_content, self.netD_style], False)
        else:
            self.set_requires_grad(self.netD, False) #D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()                 #set G's gradients to zero
        self.backward_G()                            #calcuate gradients for G
        self.optimizer_G.step()                      #update G's weights
        
        self.optimizer_G.zero_grad()

    def show_visuals(self, epoch, i, total_epoch):
        save_image(self.content_images, f'./checkpoint/gen_data/{epoch}_cc.jpg', nrow=4, normalize=True, scale_each=True)
        save_image(self.style_images, f'./checkpoint/gen_data/{epoch}_ss.jpg', nrow=4, normalize=True, scale_each=True)
        save_image(self.generated_images, f'./checkpoint/gen_data/{epoch}_fake.jpg', nrow=4, normalize=True, scale_each=True)
        print("[Epoch %d/%d] [Batch %d/%d] [g loss: %f][d content loss: %f][d style loss: %f]" %(epoch, total_epoch, i % len(self.style_images), len(self.style_images), self.loss_G_GAN,self.loss_D_content, self.loss_D_style))





class GANLoss(nn.Module):
    """Define different GAN objectives.
    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.
        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp', 'hinge']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.
        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real, train_gen=False):
        """Calculate loss given Discriminator's output and grount truth labels.
        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images
        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        elif self.gan_mode == 'hinge':
            if train_gen:
                loss = -prediction.mean()
            else:
                if target_is_real:
                    loss = torch.nn.ReLU()(1.0 - prediction).mean()
                else:
                    loss =  torch.nn.ReLU()(1.0 + prediction).mean()
        return loss

