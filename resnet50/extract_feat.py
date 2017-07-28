"""Extract and save feature for use in matlab evaluation."""

import os.path as osp
import time

import torch
from torch.autograd import Variable

import MTestIdLoss
import utils
import common

if __name__ == '__main__':
  cfg = common.get_config()
  TVT, TMO = utils.set_device(cfg.device_id)
  if cfg.use_caffe_model:
    import MyCaffeResNet as MyResNet
    MyResNet.caffe_resnet_model_dir = cfg.caffe_resnet_model_dir
  else:
    import MyResNet as MyResNet


class Test(object):
  """Test reid performance on one or more trained models."""
  def __init__(self):
    ###########
    # Models  #
    ###########

    self.reidBot = MyResNet.resnet_bottom(pretrained=False,
                                          bn_momentum=cfg.pytorch_bn_momentum)
    self.reidTop = MyResNet.resnet_top(pretrained=False, num_classes=None,
                                       bn_momentum=cfg.pytorch_bn_momentum)
    self.models = [self.reidBot, self.reidTop]

    ###########################################
    # May Transfer Models to Specified Device #
    ###########################################

    TMO(self.models)

    ############
    # Test Set #
    ############

    def feature_func(ims):
      """A function to be called in the val set, to extract features."""
      # Set eval mode
      utils.may_set_mode(self.models, 'eval')
      ims = TVT(Variable(torch.FloatTensor(ims)))
      feats = self.reidTop(self.reidBot(ims))
      feats = feats.data.cpu().numpy()
      return feats

    self.test_set = MTestIdLoss.MTestIdLoss(
      dataset_root=cfg.dataset_root,
      feature_func=feature_func,
      cache_dir=None, batch_size=cfg.test_batch_size,
      resize_size=cfg.im_resize_size, crop_size=cfg.im_crop_size,
      img_mean=cfg.im_mean, img_mean_file=cfg.im_mean_file,
      img_std=cfg.im_std, img_std_file=cfg.im_std_file,
      scale=cfg.scale_im, mirror_type=None,
      batch_dims='NCHW', prefetch_sizes=cfg.test_prefetch_sizes)

    if cfg.test_prefetch_sizes is not None:
      self.test_set.start_prefetch()

  def test(self):

    for ckpt in cfg.co_train_saved_ckpts:

      # Load state dicts to cpu
      assert osp.isfile(ckpt), \
        "=> no checkpoint found at '{}'".format(ckpt)
      state_dicts = torch.load(
        ckpt, map_location=(lambda storage, loc: storage))
      print "=> loaded checkpoint '{}'".format(ckpt)

      # Initialize weights
      # utils.load_module_state_dict(
      #   self.reidBot, state_dicts[cfg.co_train_saved_ckpt_reidBot_ind])
      # utils.load_module_state_dict(
      #   self.reidTop, state_dicts[cfg.co_train_saved_ckpt_reidTop_ind])
      utils.load_module_state_dict(
        self.reidBot, state_dicts)
      utils.load_module_state_dict(
        self.reidTop, state_dicts)

      # t_st = time.time()
      # self.test_set.eval_single_query(True)
      # print 'Single query done, {:.3f}s'.format(time.time() - t_st)
      #
      # t_st = time.time()
      # self.test_set.eval_multi_query(False)
      # print 'Multi query done, {:.3f}s'.format(time.time() - t_st)

      utils.save_mat(
        self.test_set.query_ids, osp.join(cfg.experiment_root, 'q_ids.mat'))
      utils.save_mat(
        self.test_set.query_cams, osp.join(cfg.experiment_root, 'q_cams.mat'))
      utils.save_mat(
        self.test_set.gallery_ids, osp.join(cfg.experiment_root, 'g_ids.mat'))
      utils.save_mat(
        self.test_set.gallery_cams, osp.join(cfg.experiment_root, 'g_cams.mat'))

      t_st = time.time()
      q_feat = self.test_set.extract_feat('query', normalize=False)
      utils.save_mat(q_feat, osp.join(cfg.experiment_root, 'q_feat.mat'))
      print 'Extracting query feat done, {:.3f}s'.format(time.time() - t_st)

      t_st = time.time()
      g_feat = self.test_set.extract_feat('gallery', normalize=False)
      utils.save_mat(g_feat, osp.join(cfg.experiment_root, 'g_feat.mat'))
      print 'Extracting gallery feat done, {:.3f}s'.format(time.time() - t_st)

      t_st = time.time()
      av_feat, max_feat = self.test_set.extract_pooled_query_feat(normalize=False)
      utils.save_mat(av_feat, osp.join(cfg.experiment_root, 'av_feat.mat'))
      utils.save_mat(max_feat, osp.join(cfg.experiment_root, 'max_feat.mat'))
      print 'Extracting multi-query feat done, {:.3f}s'.format(time.time() - t_st)

    self.test_set.may_stop_prefetch_thread()


Test().test()
