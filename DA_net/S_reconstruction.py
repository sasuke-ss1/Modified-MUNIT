import torch
import torch.nn as nn
import DA_net.networks as networks
from DA_net.losses import GANLoss
from DA_net.TVLoss1 import L1_TVLoss_Charbonnier
from DA_net.ECLoss import DCLoss
from DA_net.utils import get_config, SSIM, VGGPerceptualLoss, save_images


class S_recon(nn.Module):
    def __init__(self, m_config, res_config, d_config = None, train = True):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m_config = get_config(m_config)
        munit = networks.get_MUNIT(m_config)
        res_config = get_config(res_config)
        d_config = get_config(d_config)
        self.S_recon = networks.get_resnet(res_config).to(self.device)
        self.train = train
        if self.train:
            d_config["use_sigmoid"] = False
            self.netD = networks.get_D(d_config)
            state_dict = torch.load(m_config["checkpoint"])
            munit.gen_uw.load_state_dict(state_dict["a"])
            munit.gen_sw.load_state_dict(state_dict['b'])
            munit.cuda()
            munit.eval()
            self.S_encoder = munit.gen_sw.encode;self.S_decoder = munit.gen_sw.decode
            self.R_encoder = munit.gen_uw.encode;self.R_decoder = munit.gen_uw.decode

            self.lambda_recon = res_config["lambda_recon"]
            self.lambda_TV = res_config["lambda_TV"]
            self.lambda_DC = res_config["lambda_DC"]
            self.lambda_GAN = res_config["lambda_GAN"]
            self.patch_size = d_config["patch_size"]
            self.GanLoss = GANLoss(use_ls = True).to(self.device)
            self.ssim = SSIM().to(self.device)
            self.perceptual = VGGPerceptualLoss(resize = False).to(self.device)
            self.ReconLoss = nn.MSELoss()
            self.TVLoss = L1_TVLoss_Charbonnier()
            self.G_optim = torch.optim.Adam(self.S_recon.parameters(), lr = res_config["lr"], betas = (0.9, 0.999))
            self.D_optim = torch.optim.Adam(self.netD.parameters(), lr = d_config["lr"], betas = (d_config["beta1"], 0.999))

        if res_config["checkpoint"] and not self.train:
            self.S_recon.load_state_dict(res_config["checkpoint"])
            self.S_recon.eval()

    def forward(self, syn_img, real_img, clear_img):
        self.syn_img = syn_img.to(self.device)
        self.real_img = real_img.to(self.device)
        self.clear_img = clear_img.to(self.device)
        if self.train:
            Fcon, Fsty = self.S_encoder(self.syn_img)
            Rcon, Rsty = self.R_encoder(self.real_img)
            self.img_r2s = self.S_decoder(Rcon, Fsty).detach()
            self.s_recon_img = self.S_recon(self.syn_img)[-1] ## -1 is sus
            self.r2s_recon_img = self.S_recon(self.img_r2s)[-1] ## -1 is sus
        


        else:
            self.s_recon_img = self.S_recon(syn_img)[-1] ## -1 is sus

    def backward_D_basic(self, netD, real, fake):
        pred_real = netD(real)
        loss_D_real = self.GanLoss(pred_real, True)

        pred_fake = netD(fake.detach())
        loss_D_fake = self.GanLoss(pred_fake, False)

        loss_D = (loss_D_fake + loss_D_real)*0.5

        loss_D.backward()
        return loss_D    

    def backward_D(self):
        self.loss_D = self.backward_D_basic(self.netD, self.clear_img, self.s_recon_img)
    
    def backward_G(self):
        self.loss_S_recon = self.ReconLoss(self.s_recon_img, self.clear_img)*self.lambda_recon
        self.perceptual_loss = self.perceptual(self.s_recon_img, self.clear_img)
        self.loss_r2s_recon_TV = self.TVLoss(self.r2s_recon_img)*self.lambda_TV
        self.loss_r2s_recon_DC = DCLoss((self.r2s_recon_img + 1)/2, self.patch_size)*self.lambda_DC
        self.loss_G = self.GanLoss(self.netD(self.r2s_recon_img), True)*self.lambda_GAN
        self.ssim_loss = self.ssim(self.s_recon_img, self.clear_img)
        self.netLoss = self.loss_G + self.loss_S_recon + self.ssim_loss * 2 +self.perceptual_loss*0.1+ self.loss_r2s_recon_DC + self.loss_r2s_recon_TV
        self.netLoss.backward()

    def optimize(self, syn_img, real_img, clear_img):
        self.forward(syn_img, real_img, clear_img)
        # G
        self.set_grad_false(self.netD)
        self.G_optim.zero_grad()
        self.backward_G()
        self.G_optim.step()
        # D
        self.set_grad_false(self.netD, True)
        self.D_optim.zero_grad()
        self.backward_D()
        self.D_optim.step()
    
    def set_grad_false(self, net, grad = False):
        for param in net.parameters():
            param.requires_grad = grad
    def save(self, fake_img, real_img, clear_img, path):
        save_images([self.s_recon_img.detach().cpu(),self.img_r2s.detach().cpu(), self.r2s_recon_img.detach().cpu(), fake_img.detach().cpu(), real_img.detach().cpu(), clear_img.detach().cpu()], path)





