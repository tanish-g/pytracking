import os
from collections import OrderedDict
from ltr.trainers import BaseTrainer
from ltr.admin.stats import AverageMeter, StatValue
from ltr.admin.tensorboard import TensorboardWriter
import torch
import time
import torch.nn as nn
from tqdm import tqdm
import wandb

class scaler(nn.Module):
    def __init__(self,num_features):
        super(scaler, self).__init__()
        self.weight = nn.parameter.Parameter(torch.empty(num_features)).reshape(1,num_features,1,1).cuda()
        self.weight.retain_grad()
    def forward(self,x):
        out = self.weight * x
        return out
    
class LTRTrainer(BaseTrainer):
    def __init__(self, actor, loaders, optimizer, settings,lr_scheduler=None):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler)

        self._set_default_settings()
        wandb.login(key='be7f0d41e450e88a50bffe21de84b92e10fbb826')
        run = wandb.init(project="Pruning-in-Tracking",
                         name=settings.script_name,
                         group=settings.ckpt_path,
                         entity="tg34",
                         resume="allow")
        # Initialize statistics variables
        self.stats = OrderedDict({loader.name: None for loader in self.loaders})

        # Initialize tensorboard
        tensorboard_writer_dir = os.path.join(self.settings.env.tensorboard_dir, self.settings.project_path)
        self.tensorboard_writer = TensorboardWriter(tensorboard_writer_dir, [l.name for l in loaders])
        self.prune = settings.prune
        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)
    
#     class scaler(nn.Module):
#         def __init__(self,num_features):
#             super(scaler, self).__init__()
#             self.weight = nn.parameter.Parameter(torch.empty(num_features)).reshape(1,num_features,1,1).cuda()
#             self.weight.retain_grad()
#         def forward(self,x):
#             out = self.weight * x
#             return out
    
    
    def _set_default_settings(self):
        # Dict of all default values
        default = {'print_interval': 10,
                   'print_stats': None,
                   'description': ''}

        for param, default_value in default.items():
            if getattr(self.settings, param, None) is None:
                setattr(self.settings, param, default_value)
    
    def updateBN(self):
        for m in self.actor.net.feature_extractor.modules():
            if isinstance(m, nn.BatchNorm2d):
                try:
                    m.weight.grad.data.add_(self.settings.s*torch.sign(m.weight.data))  # L1
                except:
                    continue
    
    def update_BN_mask(self):
        for name, m in self.actor.net.feature_extractor.named_modules():
#             m.weight.grad.data.add_(self.settings.s*torch.sign(m.weight.data))
            if isinstance(m, nn.BatchNorm2d):
                try:
#                     print('Done_Bn')
                    m.weight.grad.data.add_(self.settings.s*torch.sign(m.weight.data))  # L1
                except:
                    continue
 
            if name[-4:] == 'mask':
                try:
                    m.weight.grad.data.add_(self.settings.s*3*torch.sign(m.weight.data))  # L1
                except:
                    continue


            
                


    def cycle_dataset(self, loader):
        """Do a cycle of training or validation."""

        self.actor.train(loader.training)
        torch.set_grad_enabled(loader.training)

        self._init_timing()
        tk = tqdm(loader, total = len(loader))
        if loader.name == 'train':
            self.running_loss_train = 0
            self.running_loss_val = 0
        for i, data in enumerate(tk):
            # get inputs
            if self.move_data_to_gpu:
                data = data.to(self.device)

            data['epoch'] = self.epoch
            data['settings'] = self.settings

            # forward pass
            loss, stats = self.actor(data)
            
            if loader.name == 'train':
                self.running_loss_train += loss.item() * self.settings.batch_size
#                 self.global_loss = self.running_loss_train
            else:
                self.running_loss_val += loss.item() * self.settings.batch_size
#                 self.global_loss = self.running_loss_val
            # backward pass and update weights
            if loader.training:
                self.optimizer.zero_grad()
                loss.backward()
                if self.prune: ##add gamma grad in bn
#                     self.updateBN_mask()
                      self.update_BN_mask()
                self.optimizer.step()

            # update statistics
            self._update_stats(stats, loader.batch_size, loader)
            tk.set_postfix({'epoch': self.epoch, 'loss': loss.item(),  })

            # print statistics
#             self._print_stats(i, loader, loader.batch_size)
        
        if loader.name=='train':
            self.running_loss_train = self.running_loss_train/len(loader)
        else:
            self.running_loss_val = self.running_loss_val/len(loader)
            
    def train_epoch(self):
        """Do one epoch for each loader."""
        for loader in self.loaders:
            if self.epoch % loader.epoch_interval == 0:
                self.cycle_dataset(loader)

        self._stats_new_epoch()
        self._write_tensorboard()
        wandb.log({'epoch':self.epoch,'train_loss':self.running_loss_train,'valid_loss':self.running_loss_val})
        if self.epoch==self.settings.max_epoch:
            wandb.finish()
        
    def _init_timing(self):
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        # Initialize stats if not initialized yet
        if loader.name not in self.stats.keys() or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict({name: AverageMeter() for name in new_stats.keys()})

        for name, val in new_stats.items():
            if name not in self.stats[loader.name].keys():
                self.stats[loader.name][name] = AverageMeter()
            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size):
        self.num_frames += batch_size
        current_time = time.time()
        batch_fps = batch_size / (current_time - self.prev_time)
        average_fps = self.num_frames / (current_time - self.start_time)
        self.prev_time = current_time
        if i % self.settings.print_interval == 0 or i == loader.__len__():
            print_str = '[%s: %d, %d / %d] ' % (loader.name, self.epoch, i, loader.__len__())
            print_str += 'FPS: %.1f (%.1f)  ,  ' % (average_fps, batch_fps)
            for name, val in self.stats[loader.name].items():
                if (self.settings.print_stats is None or name in self.settings.print_stats) and hasattr(val, 'avg'):
                    print_str += '%s: %.5f  ,  ' % (name, val.avg)
            print(print_str[:-5])

    def _stats_new_epoch(self):
        # Record learning rate
        for loader in self.loaders:
            if loader.training:
                lr_list = self.lr_scheduler.get_lr()
                for i, lr in enumerate(lr_list):
                    var_name = 'LearningRate/group{}'.format(i)
                    if var_name not in self.stats[loader.name].keys():
                        self.stats[loader.name][var_name] = StatValue()
                    self.stats[loader.name][var_name].update(lr)

        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat_value in loader_stats.values():
                if hasattr(stat_value, 'new_epoch'):
                    stat_value.new_epoch()

    def _write_tensorboard(self):
        if self.epoch == 1:
            self.tensorboard_writer.write_info(self.settings.module_name, self.settings.script_name, self.settings.description)

        self.tensorboard_writer.write_epoch(self.stats, self.epoch)
