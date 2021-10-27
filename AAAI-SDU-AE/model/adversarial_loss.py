'''
Description: 
Author: Li Siheng
Date: 2021-10-26 04:05:23
LastEditTime: 2021-10-27 08:02:20
'''
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoConfig, AutoModel


class AdversarialLoss(object):

    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

        # * divergence function
        self.divergence = getattr(self, args.divergence)

    
    def __call__(self, model, **inputs):

        loss, logits = model(**inputs)
        
        print(loss)
        # * get disturbed inputs
        inputs_embeds = model.bert.embeddings.word_embeddings(inputs['input_ids'])
        noise = inputs_embeds.new_tensor(inputs_embeds).normal_(0, 1) * self.args.noise_var
        noise.requires_grad_()
        inputs_embeds = inputs_embeds.detach() + noise
        inputs['input_ids'] = None
        _, adv_logits = model(**inputs, inputs_embeds=inputs_embeds)

        adv_loss = self.divergence(adv_logits, logits.detach(), reduction='batchmean')
        
        # * now we need to find the best noise according to gradient
        # * theoretically we need the max, to be more efficient, we
        # * approximate with it by gradient assent
        noise_grad = torch.autograd.grad(outputs=adv_loss, inputs=noise)
        print(noise_grad)
        noise = noise + noise_grad * self.args.adv_step_size

        # * normalization
        noise = self.adv_project(noise, norm_type=self.args.project_norm_type, eps=self.args.noise_gamma)
        
        adv_loss = self.divergence(adv_logits, logits)
        loss = loss + self.args.adv_alpha * adv_loss

        return loss

    @staticmethod
    def adv_project(grad, norm_type='inf', eps=1e-6):
        if norm_type == 'l2':
            direction = grad / (torch.norm(grad, dim=-1, keepdim=True) + eps)
        elif norm_type == 'l1':
            direction = grad.sign()
        else:
            direction = grad / (grad.abs().max(-1, keepdim=True)[0] + eps)
        return direction

    
    @staticmethod
    def kl(input, target, reduction="sum"):
        input = input.float()
        target = target.float()
        loss = F.kl_div(F.log_softmax(input, dim=-1, dtype=torch.float32), F.softmax(target, dim=-1, dtype=torch.float32), reduction=reduction)
        return loss


    @staticmethod
    def sym_kl(input, target, reduction="sum"):
        input = input.float()
        target = target.float()
        loss = F.kl_div(F.log_softmax(input, dim=-1, dtype=torch.float32), F.softmax(target.detach(), dim=-1, dtype=torch.float32), reduction=reduction) + \
            F.kl_div(F.log_softmax(target, dim=-1, dtype=torch.float32), F.softmax(input.detach(), dim=-1, dtype=torch.float32), reduction=reduction)
        return loss

    @staticmethod
    def js(input, target, reduction="sum"):
        input = input.float()
        target = target.float()
        m = F.softmax(target.detach(), dim=-1, dtype=torch.float32) + \
            F.softmax(input.detach(), dim=-1, dtype=torch.float32)
        m = 0.5 * m
        loss = F.kl_div(F.log_softmax(input, dim=-1, dtype=torch.float32), m, reduction=reduction) + \
            F.kl_div(F.log_softmax(target, dim=-1, dtype=torch.float32), m, reduction=reduction)
        return loss
    
    @staticmethod
    def hl(input, target, reduction="sum"):
        """Hellinger divergence
        """
        input = input.float()
        target = target.float()
        si = F.softmax(target.detach(), dim=-1, dtype=torch.float32).sqrt_()
        st = F.softmax(input.detach(), dim=-1, dtype=torch.float32).sqrt_()
        loss = F.mse_loss(si, st)
        return loss