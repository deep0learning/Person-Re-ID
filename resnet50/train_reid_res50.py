import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import time
import numpy as np

import MyResNet
import MTrainIdLoss
import MValIdLoss
import utils

if __name__ == '__main__':
  import Config
  # All training and testing configurations.
  cfg = Config.Config
  TVT, TMO = utils.set_device(cfg.device_id)


class PreTrainReID(object):
  def __init__(self):
    ###########
    # Models  #
    ###########

    self.reidBot = MyResNet.resnet_bottom(
      pretrained=True, bn_momentum=cfg.pytorch_bn_momentum)
    self.reidTop = MyResNet.resnet_top(
      pretrained=True, num_classes=cfg.num_classes,
      bn_momentum=cfg.pytorch_bn_momentum,
      dropout_rate=cfg.pre_reid_dropout_rate)

    #####################
    # Initialize Models #
    #####################

    pass

    #############################
    # Criteria and Optimizers   #
    #############################

    self.reid_criterion = nn.CrossEntropyLoss()

    reid_ft_fs_params = MyResNet.get_fs_ft_params(self.reidTop)
    self.optimReID = optim.SGD(
      [{'params': self.reidBot.parameters()},
       {'params': reid_ft_fs_params[0]},
       {'params': [self.reidTop.fc.weight]},
       {'params': [self.reidTop.fc.bias]}],
      lr=1e-3,
      momentum=cfg.pre_reid_momentum, weight_decay=cfg.pre_reid_weight_decay)
    # Set weight_decay for fc bias to 0
    self.optimReID.param_groups[-1]['weight_decay'] = 0

    self.modules_optims = \
      [self.reidBot, self.reidTop, self.reid_criterion, self.optimReID]

    ################################
    # May Resume Models and Optims #
    ################################

    if cfg.pre_reid_resume:
      utils.may_load_modules_optims_state_dicts(
        self.modules_optims, cfg.pre_reid_resume_ckpt, load_to_cpu=True)

    ######################################################
    # May Transfer Models and Optims to Specified Device #
    ######################################################

    TMO(self.modules_optims)

    ################
    # Training Set #
    ################

    mirror_type = 'random' if cfg.mirror_im else None

    self.train_set = MTrainIdLoss.MTrainIdLoss(
      dataset_root=cfg.dataset_root,
      partition_file=cfg.train_val_partition_file, part=cfg.train_val_part,
      batch_size=cfg.pre_reid_train_set_batch_size, final_batch=True,
      shuffle=True,
      resize_size=cfg.im_resize_size, crop_size=cfg.im_crop_size,
      img_mean=cfg.im_mean, img_mean_file=cfg.im_mean_file,
      img_std=cfg.im_std, img_std_file=cfg.im_std_file,
      scale=cfg.scale_im, mirror_type=mirror_type,
      batch_dims='NCHW', num_prefetch_threads=cfg.prefetch_threads
    )

    ###########################
    # May Need Validation Set #
    ###########################

    # When train on 'train' part, validation is needed

    def feature_func(ims):
      """A function to be called in the val set, to extract features."""
      # Set eval mode
      # Force all BN layers to use global mean and variance, also disable
      # dropout.
      utils.may_set_mode(self.modules_optims, 'eval')
      ims = TVT(Variable(torch.from_numpy(ims).float()))
      feats, _ = self.reidTop(self.reidBot(ims))
      feats = feats.data.cpu().numpy()
      return feats

    self.val_set = None
    if cfg.train_val_part == 'train':
      self.val_set = MValIdLoss.MValIdLoss(
        dataset_root=cfg.dataset_root,
        partition_file=cfg.train_val_partition_file, feature_func=feature_func,
        batch_size=cfg.pre_reid_val_set_batch_size,
        resize_size=cfg.im_resize_size, crop_size=cfg.im_crop_size,
        img_mean=cfg.im_mean, img_mean_file=cfg.im_mean_file,
        img_std=cfg.im_std, img_std_file=cfg.im_std_file,
        scale=cfg.scale_im, mirror_type=None,
        batch_dims='NCHW', num_prefetch_threads=cfg.prefetch_threads)

  def adjust_lr(self, ep):
    utils.adjust_lr(
      param_groups=self.optimReID.param_groups,
      base_lrs=[cfg.pre_reid_ft_base_lr,
                cfg.pre_reid_ft_base_lr,
                cfg.pre_reid_fc_weight_base_lr,
                cfg.pre_reid_fc_bias_base_lr],
      decay_epochs=cfg.pre_reid_lr_decay_epochs, epoch=ep, verbose=True)

  def pretrain_reid(self):
    """Training reid, and may validate on val set."""

    start_ep = cfg.pre_reid_resume_ep if cfg.pre_reid_resume else 0
    for ep in range(start_ep, cfg.pre_reid_num_epochs):

      self.adjust_lr(ep)
      # Force all BN layers to use global mean and variance
      utils.may_set_mode(self.modules_optims, 'eval')
      # Enable dropout
      utils.may_set_mode(self.reidTop.dropout, 'train')

      epoch_done = False
      ep_losses = []
      ep_st = time.time()
      step = 0
      while not epoch_done:

        step += 1
        step_st = time.time()

        ims, im_names, labels, ims_mirrored, epoch_done = \
          self.train_set.next_batch()
        ims = TVT(Variable(torch.from_numpy(ims).float()))
        labels = TVT(Variable(torch.LongTensor(labels)))
        _, logits = self.reidTop(self.reidBot(ims))
        print(logits)
        loss = self.reid_criterion(logits, labels)
        self.optimReID.zero_grad()
        loss.backward()
        self.optimReID.step()

        ep_losses.append(utils.to_scalar(loss))

        # Step logs
        if step % cfg.pre_reid_log_steps == 0:
          print '[Step {}/Ep {}], [{:.2f}s], [loss: {}]'.format(
            step + 1, ep + 1, time.time() - step_st, utils.to_scalar(loss))

      # Epoch logs
      print '===========> [Epoch {}], [{:.2f}s], [ep_avg_loss: {}]'.format(
        ep + 1, time.time() - ep_st, np.mean(ep_losses))

      # validation

      if cfg.train_val_part == 'train':
        self.val_set.eval_single_query(True)
        self.val_set.eval_multi_query(False)

      # epoch saving
      if (ep + 1) % cfg.pre_reid_epochs_per_saving_ckpt == 0 \
          or ep + 1 == cfg.pre_reid_num_epochs:
        utils.may_save_modules_optims_state_dicts(
          self.modules_optims, cfg.pre_reid_ckpt_saving_tmpl.format(ep + 1))

    self.train_set.stop_prefetching_threads()
    if cfg.train_val_part == 'train':
      self.val_set.stop_prefetching_threads()


if __name__ == '__main__':
  PreTrainReID().pretrain_reid()
