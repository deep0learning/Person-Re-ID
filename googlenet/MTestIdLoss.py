import numpy as np
import os
import os.path as osp
import time
import matplotlib.pyplot as plt
import Market1501_utils as M_utils
import utils
import reid.evaluation_metrics.ranking as ranking
from multiprocessing.pool import ThreadPool as Pool


class ImageSetWrapper(object):
  """This wrapper is to avoid duplicate code among query set, gallery set and 
  gt_bbox set. It's now generic."""
  def __init__(self, img_dir, img_names, pre_process_img_func,
               extract_feat_func, batch_size, num_threads,
               multi_thread_stacking=False):
    """
    Args:
      extract_feat_func: External model for extracting features. It takes a 
        batch of images and returns a batch of features.
      multi_thread_stacking: bool, whether to use multi threads to speed up
        `np.stack()` or not. When the system is memory overburdened, using 
        `np.stack()` to stack a batch of images takes ridiculously long time.
        E.g. it may take several seconds to stack a batch of 64 images.
    """
    self.img_dir = img_dir
    self.img_names = img_names
    self.pre_process_img_func = pre_process_img_func
    self.extract_feat_func = extract_feat_func
    self.prefetcher = utils.Prefetcher(
      self.get_sample, len(img_names), batch_size, num_threads=num_threads)
    self.epoch_done = True
    self.multi_thread_stacking = multi_thread_stacking
    if multi_thread_stacking:
      self.pool = Pool(processes=8)

  def get_sample(self, ptr):
    img_path = osp.join(self.img_dir, self.img_names[ptr])
    im = plt.imread(img_path)
    im, _ = self.pre_process_img_func(im)
    id = M_utils.parse_img_name(self.img_names[ptr], 'id')
    cam = M_utils.parse_img_name(self.img_names[ptr], 'cam')
    return im, id, cam

  def set_batch_size(self, batch_size):
    """You can only change batch size at the beginning of a new epoch."""
    self.prefetcher.set_batch_size(batch_size)

  def next_batch(self):
    # Start enqueuing and other preparation at the beginning of an epoch.
    if self.epoch_done:
      self.prefetcher.start_ep_prefetching()
    samples, self.epoch_done = self.prefetcher.next_batch()
    im_list, ids, cams = zip(*samples)
    # t = time.time()
    # Transform the list into a numpy array with shape [N, ...]
    if self.multi_thread_stacking:
      ims = np.empty([len(im_list)] + list(im_list[0].shape))
      def func(i): ims[i] = im_list[i]
      self.pool.map(func, range(len(im_list)))
    else:
      ims = np.stack(im_list, axis=0)
    # print '---stacking time {:.4f}s'.format(time.time() - t)
    ids = np.array(ids)
    cams = np.array(cams)
    return ims, ids, cams, self.epoch_done

  def extract_feat(self, normalize):
    """Extract the features of the whole image set.
    Args:
      normalize: True or False, whether to normalize each image feature to 
        vector with unit-length
    Returns:
      feats: numpy array with shape [N, feat_dim]
      ids: numpy array with shape [N]
      cams: numpy array with shape [N]
    """
    feats, ids, cams = [], [], []
    done = False
    while not done:
      ims_, ids_, cams_, done = self.next_batch()
      feats.append(self.extract_feat_func(ims_))
      ids.append(ids_)
      cams.append(cams_)
    feats = np.vstack(feats)
    ids = np.hstack(ids)
    cams = np.hstack(cams)

    if normalize:
      feats = M_utils.normalize(feats, axis=1)

    return feats, ids, cams

  def stop_prefetching_threads(self):
    """After finishing using the dataset, or when existing the main program, 
    this should be called."""
    self.prefetcher.stop()


class MTestIdLoss(object):
  """Using test part (query, gallery, gt_bbox) of the Market1501 dataset, for 
  network trained with identification loss."""

  def __init__(
      self, dataset_root=None, feature_func=None,
      batch_size=48, resize_size=None, crop_size=None,
      img_mean=None, img_mean_file=None,
      img_std=None, img_std_file=None,
      scale=True, mirror_type=None,
      batch_dims='NCHW', num_prefetch_threads=1):

    self.dataset_root = dataset_root
    self.query_dir = self.get_query_dir()
    self.gallery_dir = self.get_gallery_dir()
    self.gt_bbox_dir = self.get_gt_bbox_dir()

    self.query_img_names = self.get_query_img_names()
    self.gallery_img_names = self.get_gallery_img_names()
    self.gt_bbox_img_names, self.query_ids, self.query_cams = \
      self.get_gt_bbox_img_names()

    # The mean (shape = [3]) of the trainval set.
    # Mean is subtracted before scaling.
    if img_mean is None and img_mean_file is not None:
      img_mean = utils.load_pickle(img_mean_file)

    # The std (shape = [3]) of the trainval set.
    # std is used only when mean is used.
    if img_mean is None:
      img_std = None
    elif img_std is None and img_std_file is not None:
      img_std = utils.load_pickle(img_std_file)

    pre_process_img_func = M_utils.PreProcessImg(
      resize_size=resize_size, crop_size=crop_size, img_mean=img_mean,
      img_std=img_std,
      scale=scale, mirror_type=mirror_type, batch_dims=batch_dims
    )

    self.q_loader = ImageSetWrapper(
      self.query_dir, self.query_img_names, pre_process_img_func, feature_func,
      batch_size, num_prefetch_threads)
    self.g_loader = ImageSetWrapper(
      self.gallery_dir, self.gallery_img_names, pre_process_img_func,
      feature_func, batch_size, num_prefetch_threads)
    self.gb_loader = ImageSetWrapper(
      self.gt_bbox_dir, self.gt_bbox_img_names, pre_process_img_func,
      feature_func, batch_size, num_prefetch_threads)

    # Caching gallery features, shared between single and multi query
    # evaluation.
    self.g_feats, self.g_ids, self.g_cams = (None,) * 3

  def get_query_dir(self):
    return osp.join(self.dataset_root, 'query')

  def get_gallery_dir(self):
    return osp.join(self.dataset_root, 'bounding_box_test')

  def get_gt_bbox_dir(self):
    return osp.join(self.dataset_root, 'gt_bbox')

  def get_query_img_names(self):
    return M_utils.get_img_names(self.query_dir)

  def get_gallery_img_names(self):
    img_names = M_utils.get_img_names(self.gallery_dir)
    # Filter out id '-1' which is useless in the experiment.
    img_names = [name for name in img_names if not name.startswith('-1')]
    return img_names

  def get_gt_bbox_img_names(self):
    """For multi-query, only care for those same-id same-cam counterparts in
    the gt_bbox dir."""

    gb_names_ = M_utils.get_img_names(self.gt_bbox_dir)

    q_ids = np.array(
      [M_utils.parse_img_name(name, 'id') for name in self.query_img_names])
    q_cams = np.array(
      [M_utils.parse_img_name(name, 'cam') for name in self.query_img_names])
    gb_ids = np.array(
      [M_utils.parse_img_name(name, 'id') for name in gb_names_])
    gb_cams = np.array(
      [M_utils.parse_img_name(name, 'cam') for name in gb_names_])

    gb_names = []
    for q_id, q_cam in zip(q_ids, q_cams):
      gb_names.append(
        gb_names_[np.logical_and(gb_ids == q_id, gb_cams == q_cam)])
    gb_names = np.hstack(gb_names)

    return gb_names, q_ids, q_cams

  def clear_inner_cache(self):
    """To clear previously calculated features. On changing to a new 
    feature model, this should be called."""
    self.g_feats, self.g_ids, self.g_cams = (None,) * 3

  def eval_single_query(self, clear_cache):
    """Evaluate in the single-query case, using metric CMC and mAP.
    Args:
      clear_cache: True or False, whether to clear previously calculated 
      features or not. On changing to a new feature model, this should be set 
      to True.
    """
    if clear_cache:
      self.clear_inner_cache()

    # print 'start feat extraction'
    t_st = time.time()
    q_feats, q_ids, q_cams = self.q_loader.extract_feat(True)
    # print 'query feat extraction done, {:.2f}s'.format(time.time() - t_st)
    if self.g_feats is None:
      t_st = time.time()
      self.g_feats, self.g_ids, self.g_cams = self.g_loader.extract_feat(True)
      # print 'gallery feat extraction done, {:.2f}s'.format(time.time() - t_st)

    q_g_dist = M_utils.compute_dist(q_feats, self.g_feats)

    t_st = time.time()
    self.eval_cmc_map(q_g_dist, q_ids, self.g_ids, q_cams, self.g_cams)
    # print 'Single query eval done, {:.2f}s'.format(time.time() - t_st)

  def eval_multi_query(self, clear_cache, pool_type='average'):
    """Evaluate in the multi-query case, using metric CMC and mAP.
    Args:
      clear_cache: True or False, whether to clear previously calculated 
      features or not. On changing to a new feature model, this should be set 
      to True.
      pool_type: 'average' or 'max'
    """
    if clear_cache:
      self.clear_inner_cache()

    # Gallery features.

    t_st = time.time()
    # print 'start feat extraction'
    gb_feats, gb_ids, gb_cams = self.gb_loader.extract_feat(False)
    # print 'gt_bbox feat extraction done, {:.2f}s'.format(time.time() - t_st)
    if self.g_feats is None:
      t_st = time.time()
      self.g_feats, self.g_ids, self.g_cams = self.g_loader.extract_feat(True)
      # print 'gallery feat extraction done, {:.2f}s'.format(time.time() - t_st)

    # Pooled query features.

    pooled_feats = []
    for q_id, q_cam in zip(self.query_ids, self.query_cams):
      feat = gb_feats[np.logical_and(gb_ids == q_id, gb_cams == q_cam)]
      assert pool_type in ['average', 'max']
      if pool_type == 'average':
        feat = np.mean(feat, axis=0)
      else:
        feat = np.max(feat, axis=0)
      pooled_feats.append(feat)
    pooled_feats = np.vstack(pooled_feats)

    q_g_dist = M_utils.compute_dist(pooled_feats, self.g_feats)

    t_st = time.time()
    self.eval_cmc_map(q_g_dist, self.query_ids, self.g_ids,
                      self.query_cams, self.g_cams)
    # print 'Multi query eval done, {:.2f}s'.format(time.time() - t_st)

  def eval_cmc_map(self, q_g_dist, q_ids, g_ids, q_cams, g_cams):
    """Compute CMC and mAP.
    Args:
      q_g_dist: numpy array with shape [num_query, num_gallery], the 
        pairwise distance between query and gallery samples
    Returns:
      mAP: numpy array with shape [num_query], the AP averaged across query 
        samples
      cmc_scores: numpy array (with shape [num_query-1]?), the cmc curve 
        averaged across query samples
    """
    # Compute mean AP
    mAP = ranking.mean_ap(distmat=q_g_dist,
                          query_ids=q_ids, gallery_ids=g_ids,
                          query_cams=q_cams, gallery_cams=g_cams)
    # Compute CMC scores
    cmc_scores = ranking.cmc(
      distmat=q_g_dist, query_ids=q_ids, gallery_ids=g_ids,
      query_cams=q_cams, gallery_cams=g_cams, separate_camera_set=False,
      single_gallery_shot=False, first_match_break=True)

    print '[mAP: {:5.2%}], [cmc1: {:5.2%}], [cmc5: {:5.2%}], [cmc10: {:5.2%}]' \
      .format(mAP, *cmc_scores[[0, 4, 9]])

    return mAP, cmc_scores

  def stop_prefetching_threads(self):
    """After finishing using the dataset, or when existing the main program, 
    this should be called."""
    self.q_loader.stop_prefetching_threads()
    self.g_loader.stop_prefetching_threads()
    self.gb_loader.stop_prefetching_threads()
