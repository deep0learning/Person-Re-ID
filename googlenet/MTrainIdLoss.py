import numpy as np
import os
import os.path as osp
import matplotlib.pyplot as plt
import Market1501_utils as M_utils
import utils
from multiprocessing.pool import ThreadPool as Pool


class MTrainIdLoss(object):
  def __init__(self, dataset_root=None, partition_file=None,
               part='trainval',
               batch_size=48, final_batch=True, shuffle=True,
               resize_size=None, crop_size=None,
               img_mean=None, img_mean_file=None,
               img_std=None, img_std_file=None,
               scale=True, mirror_type=None,
               batch_dims='NCHW', num_prefetch_threads=1,
               multi_thread_stacking=False):
    """
    Using train or train+val part of the Market1501 dataset, trained with 
      identification loss.
    Args:
      final_batch: bool. The last batch may not be complete, if to abandon this 
        batch, set 'final_batch' to False.
      multi_thread_stacking: bool, whether to use multi threads to speed up
        `np.stack()` or not. When the system is memory overburdened, using 
        `np.stack()` to stack a batch of images takes ridiculously long time.
        E.g. it may take several seconds to stack a batch of 64 images.
    """
    self.img_dir = osp.join(dataset_root, 'bounding_box_train')

    # Get the train and val set partition.
    assert osp.exists(partition_file), 'Train/Val partition file not found!'
    partitions = utils.load_pickle(partition_file)

    # Image names to use for training.
    assert part in ['trainval', 'train']
    self.img_names = partitions[0] if part == 'trainval' else partitions[1]

    # Transform the ids to labels for training identification loss.
    self.ids2labels = partitions[-2] if part == 'trainval' else partitions[-1]

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

    self.pre_process_img_func = M_utils.PreProcessImg(
      resize_size=resize_size, crop_size=crop_size, img_mean=img_mean,
      img_std=img_std, scale=scale, mirror_type=mirror_type,
      batch_dims=batch_dims
    )

    self.prefetcher = utils.Prefetcher(
      self.get_sample, len(self.img_names), batch_size,
      final_batch=final_batch, num_threads=num_prefetch_threads)
    self.shuffle = shuffle
    self.epoch_done = True
    self.multi_thread_stacking = multi_thread_stacking
    if multi_thread_stacking:
      self.pool = Pool(processes=8)

  def set_mirror_type(self, mirror_type):
    self.pre_process_img_func.set_mirror_type(mirror_type)

  def get_sample(self, ptr):
    img_name = self.img_names[ptr]
    img_path = osp.join(self.img_dir, img_name)
    im = plt.imread(img_path)
    im, mirrored = self.pre_process_img_func(im)
    id = M_utils.parse_img_name(img_name, 'id')
    label = self.ids2labels[id]
    return im, img_name, label, mirrored

  def set_batch_size(self, batch_size):
    """You can only change batch size at the beginning of a new epoch."""
    self.prefetcher.set_batch_size(batch_size)

  def next_batch(self):
    """Next batch of images and labels.
    Returns:
      ims: numpy array with shape [N, H, W, C] or [N, C, H, W], N >= 1
      img_names: a numpy array of image names, len(img_names) >= 1
      labels: a numpy array of image labels, len(labels) >= 1
      mirrored: a numpy array of booleans, whether the images are mirrored
      self.epoch_done: whether the epoch is over
    """
    # Start enqueuing and other preparation at the beginning of an epoch.
    if self.epoch_done:
      if self.shuffle: np.random.shuffle(self.img_names)
      self.prefetcher.start_ep_prefetching()
    samples, self.epoch_done = self.prefetcher.next_batch()
    im_list, img_names, labels, mirrored = zip(*samples)
    # t = time.time()
    # Transform the list into a numpy array with shape [N, ...]
    if self.multi_thread_stacking:
      ims = np.empty([len(im_list)] + list(im_list[0].shape))
      def func(i): ims[i] = im_list[i]
      self.pool.map(func, range(len(im_list)))
    else:
      ims = np.stack(im_list, axis=0)
    # print '---stacking time {:.4f}s'.format(time.time() - t)
    img_names = np.array(img_names)
    labels = np.array(labels)
    mirrored = np.array(mirrored)
    return ims, img_names, labels, mirrored, self.epoch_done

  def stop_prefetching_threads(self):
    """After finishing using the dataset, or when existing the main program, 
    this should be called."""
    self.prefetcher.stop()
