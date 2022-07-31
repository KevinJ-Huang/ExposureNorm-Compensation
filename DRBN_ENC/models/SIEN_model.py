import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable
import os
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import CharbonnierLoss
from models.loss_new import SSIMLoss,VGGLoss
import torch.nn.functional as F
import random
from metrics.calculate_PSNR_SSIM import psnr_np
logger = logging.getLogger('base')


class SIEN_Model(BaseModel):
    def __init__(self, opt):
        super(SIEN_Model, self).__init__(opt)

        self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()

#######################   Continue learning model parameter setting
        if train_opt['ewc']:
            self.Importance_Pre = torch.load(os.path.join(self.opt['path']['pretrain'], '0Importance.pth'))
            self.Star_vals_Pre = torch.load(os.path.join(self.opt['path']['pretrain'], '0Star.pth'))
            logger.info("Load Pretrain Importance and Stars!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        else:
            self.Importance = []
            self.Star_vals = []
            for w in self.netG.parameters():
                self.Importance.append(torch.zeros_like(w))
                self.Star_vals.append(torch.zeros_like(w))
            logger.info("Initial Importance and Stars with zeros!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

###############################   Distilation setting
        if train_opt['distill']:
            self.netG_Pre = networks.define_G(opt).to(self.device)
            self.netG_Pre = DataParallel(self.netG_Pre)
            self.load_Pre()
            self.netG_Pre.eval()
####################################################################
        if self.is_train:
            self.netG.train()

            #### loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
                self.cri_ssim = SSIMLoss().to(self.device)
                self.mse = nn.MSELoss().to(self.device)
                # self.cri_vgg = VGGLoss(id=4).to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
                self.cri_ssim = SSIMLoss().to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
                self.cri_ssim = SSIMLoss().to(self.device)
                # self.cri_vgg = VGGLoss(id=4).to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))


            self.l_pix_w = train_opt['pixel_weight']
            self.l_ssim_w = train_opt['ssim_weight']
            self.l_vgg_w = train_opt['vgg_weight']

            #### optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            if train_opt['fix_some_part']:
                normal_params = []
                tsa_fusion_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        if 'tsa_fusion' in k:
                            tsa_fusion_params.append(v)
                        else:
                            normal_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))
                optim_params = [
                    {  # add normal params first
                        'params': normal_params,
                        'lr': train_opt['lr_G']
                    },
                    {
                        'params': tsa_fusion_params,
                        'lr': train_opt['lr_G']
                    },
                ]
            else:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        optim_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))

            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            #### schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError()

            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        LQ_IMG = data['LQ']
        GT_IMG = data['GT']
        # LQright_IMG = data['LQright']
        self.var_L = LQ_IMG.to(self.device)
        # self.varright_L = LQright_IMG.to(self.device)
        if need_GT:
            self.real_H = GT_IMG.to(self.device)

    def set_params_lr_zero(self):
        # fix normal module
        self.optimizers[0].param_groups[0]['lr'] = 0


    def optimize_parameters(self, step):
        if self.opt['train']['fix_some_part'] and step < self.opt['train']['fix_some_part']:
            self.set_params_lr_zero()

        self.netG.zero_grad() ################################################# new add
        self.optimizer_G.zero_grad()

        phr1, phr2, phr4, instf,fusef = self.netG(self.var_L)

        hr4 = self.real_H[:, :, 0::4, 0::4]
        hr2 = self.real_H[:, :, 0::2, 0::2]
        hr1 = self.real_H

        if self.opt['train']['distill']:
            var_fake = self.real_H
            with torch.no_grad():
                self.netG_Pre.eval()
                _,_,_,gtinstf,gtfusef = self.netG_Pre(var_fake.detach())


        # gt = self.real_H

        l_total = self.cri_ssim(phr1, hr1) + self.cri_ssim(phr2, hr2) + self.cri_ssim(phr4, hr4)

                  # + 1.2*self.cri_pix(out, out_pre)
        if self.opt['train']['distill']:
            l_total += self.opt['train']['distill_coff']*(self.cri_pix(instf,gtinstf.detach())+self.cri_pix(fusef,gtfusef.detach()))

        if self.opt['train']['ewc']:
            for i, w in enumerate(self.netG.parameters()):
                l_total += self.opt['train']['ewc_coff']/2 * torch.sum(torch.mul(self.Importance_Pre[i], torch.abs(w - self.Star_vals_Pre[i])))\
                           + self.opt['train']['ewc_coff']/4 * torch.square(torch.sum(torch.mul(self.Importance_Pre[i], torch.abs(w - self.Star_vals_Pre[i]))))


        l_total.backward()
        self.optimizer_G.step()
        self.fake_H = phr1
        psnr = psnr_np(self.fake_H.detach(), self.real_H.detach())

        # set log
        self.log_dict['psnr'] = psnr.item()
        self.log_dict['l_total'] = l_total.item()


    def test(self):
        self.netG.eval()
        with torch.no_grad():
            phr1, phr2, phr4,_,_ = self.netG(self.var_L)
            self.fake_H = phr1
        self.netG.train()

    ################# Initialize importance
    def Init_M(self):
        self.Importance = []
        self.Star_vals = []
        for w in self.netG.parameters():
            self.Importance.append(torch.zeros_like(w))
            self.Star_vals.append(torch.zeros_like(w))
        print("Init importance parameters again!!!!")

################ self compute importance
    def compute_M(self, step):
        phr1,phr2,phr4 = self.netG(self.var_L)

        hr4 = self.real_H[:, :, 0::4, 0::4]
        hr2 = self.real_H[:, :, 0::2, 0::2]
        hr1 = self.real_H

        l_t = self.cri_pix(phr1, hr1)+ self.cri_ssim(phr2, hr2) + self.cri_ssim(phr4, hr4)
        l_total = -l_t
        self.netG.zero_grad()
        self.optimizer_G.zero_grad()
        l_total.backward()
        with torch.no_grad():
            for i, w in enumerate(self.netG.parameters()):
                self.Importance[i].mul_(step / (step + 1))
                self.Importance[i].add_(torch.abs(w.grad.data)/(step+1))
        self.netG.zero_grad()
        self.optimizer_G.zero_grad()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['rlt'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def load_Pre(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading WarmPre model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG_Pre, self.opt['path']['strict_load'])

    def save_M(self,name):
        with torch.no_grad():
            for i, w in enumerate(self.netG.parameters()):
                self.Star_vals[i].copy_(w)
        torch.save(self.Importance, os.path.join(self.opt['path']['models'], name+'Importance.pth'))
        torch.save(self.Star_vals, os.path.join(self.opt['path']['models'], name+'Star.pth'))

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)

    def save_best(self,name):
        self.save_network(self.netG, 'best'+name, 0)
