#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 19:00:14 2022

@author: sasuke
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import networks
from ECLoss.ECLoss import BCLoss, DCLoss
import losses
from TVLoss.L1_TVLoss import L1_TVLoss_Charbonnier
from BaseModel import BaseModel

class RUnderwaterRecon(BaseModel):
    def __init__(self, train = True):
        self.Train = train
    
    def initialize(self):
        BaseModel.initialize(self, opt)

        self.S2R = networks.define_G()
        self.R_recon = networks.define_Gen()
        self.Dis = networks.define_D()
        if self.Train:
            #self.fake_pool = ImagePool(opt.pool_size)
            self.criterionrecon = torch.nn.MSELoss()
            self.TVLoss = L1_TVLoss_Charbonnier()
            
            self.criterionGAN = losses.GANLoss().to(self.device)
            
            self.opt_D = optim.Adam()
            self.opt_G = optim.Adam()
            
            self.optimizers = []
            self.optimizer.append(self.opt_G)
            self.optimizer.append(self.opt_D)
        
    def set_input(self, input):
        if self.Train:
            AtoB = self.opt.which_direction == 'AtoB'
            input_A = input["A" if AtoB else "B"]
            input_B = input["B" if AtoB else "A"]
            input_C = input["C"]
            self.syn_img = input_A.to(self.device)
            self.real_img = input_C.to(self.device)
            self.clear_img = input_B.to(self.device)
            #self.image_paths = input['A_paths' if AtoB else 'B_paths']        
            
            self.num = self.syn_img.shape[0]
        else:
            pass
        
    def forward(self):
        if self.Train:
            self.img_s2r = self.S2R(self.syn_img, style = None).detach()
            self.out = self.R_recon(torch.cat(self.img_s2r, self.real_img))
            self.s2r_recon = self.out[-1].narrow(0,0,self.num)
            self.r_recon = self.out[-1].narrow(0, self.num, self.num)
            
        else:
            self.r_recon = self.R_recon(self.real_img)[-1]
        
    def backward_D_basic(self, Dis, real, fake):
        
        pred_real = Dis(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        
        pred_fake = Dis(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D
        
    def backward_D(self):
        pass
        #r_dehazing_img = self.fake_pool.query(self.r_dehazing_img)
        #self.loss_D = self.backward_D_basic(self.netD, self.clear_img, r_dehazing_img)
        
    def backward_G(self):
        lambda_recon = self.opt.lambda_recon
        
        size = len(self.out)
        clear_imgs = None ###
        self.loss_S2R = 0.0
        for(recon_img, clear_img) in zip(self.out[1:], clear_imgs):
            self.loss_S2R += self.criterionrecon(recon_img[:self.num, :, :, :], clear_img) * lambda_recon
        
        
        self.loss_R_recon_TV = self.TVLoss(self.r_recon) * self.opt.lambda_recon_TV
        
        self.loss_R_recon_DC = DCLoss((self.r_recon + 1)/2, self.opt.patch_size) * self.opt.lambda_recon_DC
        
        #GAN Loss
        self.loss_G = self.criterionGAN(self.Dis(self.r_recon), True)*self.opt.lambda_gan
        self.loss_GR_recon = self.loss_S2R + self.loss_R_recon_TV + self.loss_R_recon_DC + self.loss_G
        
        self.loss_GR_recon.backward()
        
    def optimize_parameters(self):
        self.forward()
        self.opt_G.zero_grad()
        self.backward_G
        self.opt_G.step()        
        

                
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        