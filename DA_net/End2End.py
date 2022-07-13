import torch
import torch.nn as nn
import DA_net.networks as networks
from DA_net.losses import GANLoss
from DA_net.TVLoss1 import L1_TVLoss_Charbonnier
from DA_net.ECLoss import DCLoss
from DA_net.utils import get_config


class End2End(nn.Module):
    def __init__(self, m_config, res_config, d_config = None, e2eConfig = None,train = True):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        munit = networks.get_MUNIT(m_config)
        self.munit = munit
        res_config = get_config(res_config)
        d_config = get_config(d_config)
        e2eConfig = get_config(e2eConfig)
        self.S_recon = networks.get_resnet(res_config)
        self.R_recon = networks.get_resnet(res_config)
        self.train = train
        if self.train:
            self.lambda_Dehazing = e2eConfig["lambda_Dehazing"]
            self.lambda_Dehazing_Con = e2eConfig["lambda_Dehazing_Con"]
            #self.lambda_gan_feat = e2eConfig["lambda_gan_feat"]
            #self.lambda_idt = e2eConfig["lambda_identity"]
            self.lambda_S = e2eConfig["lambda_S"]
            self.lambda_R = e2eConfig["lambda_R"]
            self.lambda_TV = e2eConfig["lambda_TV"]
            self.lambda_DC = e2eConfig["lambda_DC"]
            self.lambda_GAN = e2eConfig["lambda_GAN"]
            self.patch_size = e2eConfig["patch_size"]
            d_config["use_sigmoid"] = False
            state_dict = torch.load(m_config["checkpoint"])
            self.munit.gen_uw.load_state_dict(state_dict["a"])
            self.munit.gen_sw.load_state_dict(state_dict['b'])
            self.munit.cuda()
            self.S_encoder = self.munit.gen_sw.encode;self.S_decoder = self.munit.gen_sw.decode
            self.R_encoder = self.munit.gen_uw.encode;self.R_decoder = self.munit.gen_uw.decode
            self.S_recon.load_state_dict(torch.load(e2eConfig["syn_path"]))
            self.R_recon.load_state_dict(torch.load(e2eConfig["real_path"]))
            self.netD_S = self.S_recon.netD()
            self.netD_R = self.R_recon.netD()
            self.L1loss = nn.L1Loss()
            self.Cons = nn.L1Loss()
            self.GanLoss = GANLoss(use_ls = True).to(self.device) #sus
            self.ReconLoss = nn.MSELoss()
            self.TVLoss = L1_TVLoss_Charbonnier()

            self.G_optim = torch.optim.Adam(list(self.S_recon.parameters())+ list(self.R_recon.parameters()),lr = res_config["lr"], betas = (0.95, 0.999))
            self.D_optim = torch.optim.Adam(
                list(self.netD_S.parameters())+
                list(self.netD_R.parameters()), betas=(0.5, 0.9)

            )
        
    def forward(self, syn_img, real_img, clear_img):
        self.syn_img = syn_img.to(self.device)
        self.real_img = real_img.to(self.device)
        self.clear_img = clear_img.to(self.device)
        if self.train:
            Fcon, Fsty = self.S_encoder(self.syn_img)
            Rcon, Rsty = self.R_encoder(self.real_img)
            self.img_s2r = self.R_decoder(Fcon, Rsty)
            self.img_r2s = self.S_decoder(Rcon, Fsty)
            self.s_recon_img = self.S_recon(self.syn_img)[-1] ## -1 is sus
            self.s2r_recon_img = self.R_recon(self.img_s2r)[-1] ## -1 is sus
            self.r2s_recon_img = self.S_recon(self.img_r2s)[-1] ## -1 is sus
            self.R_recon_img = self.R_recon(self.real_img)[-1] ## -1 is sus

    def backward_D_basic(self, netD, real, fake):
        pred_real = netD(real)
        loss_D_real = self.GanLoss(pred_real, True)

        pred_fake = netD(fake.detach())
        loss_D_fake = self.GanLoss(pred_fake, False)

        loss_D = (loss_D_fake + loss_D_real)*0.5

        loss_D.backward()
        return loss_D

    def backward_D_S(self):
        self.loss_D = self.backward_D_basic(self.netD, self.clear_img, self.r2s_recon_img) 

    def backward_D_R(self):
        self.loss_D = self.backward_D_basic(self.netD, self.clear_img, self.s2r_recon_img)

    def backward_G(self):
        ## Recon losses
        self.loss_S_recon = self.ReconLoss(self.s_recon_img, self.clear_img)*self.lambda_Dehazing
        self.loss_R_recon = self.ReconLoss(self.s2r_recon_img, self.clear_img)*self.lambda_Dehazing

        ## TV and DC loss
        self.loss_r2s_recon_TV = self.TVLoss(self.r2s_recon_img)*self.lambda_TV
        self.loss_r2s_recon_DC = DCLoss((self.r2s_recon_img + 1)/2, self.patch_size)*self.lambda_DC
        self.loss_s2r_recon_TV = self.TVLoss(self.R_recon_img)*self.lambda_TV
        self.loss_s2r_recon_DC = DCLoss((self.R_recon_img + 1)/2, self.patch_size)*self.lambda_DC

        self.loss_G_S = self.GanLoss(self.S_recon.netD(self.r2s_recon_img), True)*self.lambda_GAN
        self.loss_G_R = self.GanLoss(self.R_recon.netD(self.R_recon_img), True)*self.lambda_GAN
        ## Cosistancy loss
        self.loss_consistancy = self.Cons(self.r2s_recon_img, self.R_recon_img) * self.lambda_Dehazing_Con

        self.loss_G = self.loss_G_S + self.loss_G_R + self.loss_R_recon + self.loss_S_recon + self.loss_r2s_recon_DC + \
            self.loss_r2s_recon_TV + self.loss_s2r_recon_DC + self.loss_s2r_recon_TV  +self.Cons

    def optimize(self, syn_img, real_img, clear_img):
        self.forward(syn_img, real_img, clear_img)
        self.munit.gen_update(real_img.detach(), syn_img.detach())
        self.munit.dis_update(real_img.detach(), syn_img.detach())
        # G
        self.G_optim.zero_grad()
        self.backward_G()
        self.G_optim.step()
        # D
        self.D_optim.zero_grad()
        self.backward_D_S()
        self.backward_D_R()
        self.D_optim.step()
        