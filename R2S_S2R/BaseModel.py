#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 06:25:28 2022

@author: sasuke
"""

import torch
import networks

class BaseModel():
    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.Train = opt.Train
        self.Tensor = torch.cuda.FloatTensor if self.gpu else torch.Tensor
        self.device = torch.device('cuda' if self.gpu_ids else 'cpu')
        
        
    def set_input(self, input):
        self.input = input
    
    def forward(self):
        pass
    
    def setup(self, opt):
        if self.Train:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
            
        if not self.Train or opt.continue_train:
            self.load_networks(opt.which_epoch)
            
    def eval(self):
        pass
    
    def test(self):
        with torch.no_grad():
            self.forward()

    def set_requires_grad(self, nets, requires_grad = False):
        for net in nets:
            for params in net.parameters():
                params.requires_grad = requires_grad

            