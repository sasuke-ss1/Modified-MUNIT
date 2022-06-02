#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 05:48:14 2022

@author: sasuke
"""

import torch
import networks
import losses
from BaseModel import BaseModel
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from ECLoss.ECLoss import BCLoss, DCLoss
from TVLoss.TVLossL1 import TVLossL1
from TVLoss.L1_TVLoss import L1_TVLoss_Charbonnier

class SUnderwaterRecon(BaseModel):
    
    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.R2S = networks.define_G()
        self.S2R = networks.define_G()
        self.s_recon = networks.define_Gen()
        
        if self.Train:
            self.Dis = networks.define_D()
        
		#if self.isTrain:
		#	self.init_with_pretrained_model('R2S', self.opt.g_r2s_premodel)
		#	self.init_with_pretrained_model('S2R', self.opt.g_s2r_premodel)
		#	self.init_with_pretrained_model('S_Dehazing', self.opt.S_Dehazing_premodel)
		#	self.netR2S.eval()    
        
		#else:
			#self.init_with_pretrained_model('R2S', self.opt.g_r2s_premodel)
			#self.init_with_pretrained_model('S2R', self.opt.g_s2r_premodel)
			#self.init_with_pretrained_model('S_Dehazing', self.opt.S_Dehazing_premodel)
        
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
                AtoB = self.opt.which_direction == "AtoB"
                input_A = input["A" if AtoB else "B"]
                input_B = input["B" if AtoB else "A"]
                input_C = input["C"]
                self.syn_un_img = input_A.to(self.device)
                self.real_un_img = input_C.to(self.device)
                self.clear_img = input_B.to(self.device)
                self.num = self.syn_un_img.shape[0]
                
            else:
                self.syn_un_img = input["C"].to(self.device)
                
        def forward(self):
            
            if self.Train:
                self.img_s2r = self.S2R(self.syn_un_img)
                self.img_r2s = self.R2S(self.real_un_img)
                self.out = self.SUnderwaterRecon(torch.cat((self.syn_un_img, self._img_r2s), 0))
                
                self.s_recon_img = self.out[-1].narrow(0,0,self.num)
                self.r2s_recon_img = self.out[-1].narrow(0, self.num, self.num)

            else:
                self.s_recon_img = self.SUnderwaterRecon(self.syn_un_img)[-1]

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
        
        #size = len(self.out)
        clear_imgs = None ###
        for(recon_img, clear_img) in zip(self.out[1:], clear_imgs):
            self.loss_S_recon += self.criterionrecon(recon_img[:self.num,:,:,:], clear_img)*lambda_recon
        
        self.loss_R_recon_TV = self.TVLoss(self.r2s_recon_img) * self.opt.lambda_recon_TV
        
        self.loss_R_recon_DC = DCLoss((self.r2s_recon_img + 1)/2, self.opt.patch_size) * self.opt.lambda_recon_DC
        
        self.loss_G = self.criterionGAN(self.Dis(self.r2s_recon_img), True) * self.opt.lambda_gan
        self.loss_GS_recon = self.loss_S_recon + self.loss_G + self.loss_R_recon_DC + self.loss_R_recon_TV
        
        self.loss_GS_recon.backward()
        
    def optimize_parameters(self):
        
        self.forward()
        self.set_requires_grad(self.Dis, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()
        
        self.set_requires_grad(self.Dis, True)
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()
        
        
        
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            