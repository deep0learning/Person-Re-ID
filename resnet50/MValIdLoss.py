import os.path as osp
import MTestIdLoss
import utils


class MValIdLoss(MTestIdLoss.MTestIdLoss):
  """Using val part of the Market1501 dataset, for network trained with 
  identification loss. Partition the val part into query and gallery set. Then
  the eval procedure is the same as testing, so inherit from MTestIdLoss."""

  def __init__(self, partition_file=None, **kwargs):

    # Get the partitions.
    assert osp.exists(partition_file), 'Train/Val partition file not found!'
    self.partitions = utils.load_pickle(partition_file)

    super(MValIdLoss, self).__init__(**kwargs)

  def get_query_dir(self):
    return osp.join(self.dataset_root, 'bounding_box_train')

  def get_gallery_dir(self):
    return osp.join(self.dataset_root, 'bounding_box_train')

  def get_query_img_names(self):
    return self.partitions[3]

  def get_gallery_img_names(self):
    return self.partitions[4]
