#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 19:17:10 2022

@author: sasuke
"""

import torch
import torch.nn as nn
import numpy as np

class GANLoss(nn.Module):
    def __init__(self, use_ls=True, target_real_label=1.0, target_fake_label=0.0):
        super.__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_ls:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()
    
    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)
    
class DiscLoss():
    pass
    

