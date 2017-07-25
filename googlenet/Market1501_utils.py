import os
import os.path as osp
import numpy as np
import glob
import cv2
import matplotlib.pyplot as plt
import pickle


def get_img_names(img_dir, pattern='*.jpg'):
  """Get the image names in a dir. Return numpy array for slicing, etc."""

  img_names = [osp.basename(path) for path in
               glob.glob(osp.join(img_dir, pattern))]
  return np.array(img_names)


def parse_img_name(img_name, parse_type='id'):
  """Get the person id or cam from an image name."""
  assert parse_type in ('id', 'cam')
  if parse_type == 'id':
    parsed = -1 if img_name.startswith('-1') else int(img_name[:4])
  else:
    parsed = int(img_name[4]) if img_name.startswith('-1') else int(img_name[6])
  return parsed


def resize_img(im, new_size):
  """Resize `im` to `new_size`: [new_w, new_h]."""

  im = cv2.resize(im, new_size, interpolation=cv2.INTER_LINEAR)
  return im


def rand_crop_img(im, new_size):
  """Crop `im` to `new_size`: [new_w, new_h]."""

  h_start = np.random.randint(0, im.shape[0] - new_size[1])
  w_start = np.random.randint(0, im.shape[1] - new_size[0])
  im = np.copy(
    im[h_start: h_start + new_size[1], w_start: w_start + new_size[0], :])

  return im


class PreProcessImg(object):
  def __init__(self, resize_size=None, crop_size=None, img_mean=None,
               scale=True, img_std=None, mirror_type=None, batch_dims='NCHW'):
    """
    Args:
      resize_size: (width, height) after resizing. If `None`, no resizing.
      crop_size: (width, height) after cropping. If `None`, no cropping.
      batch_dims: either 'NCHW' or 'NHWC'. 'N': batch size, 'C': num channels, 
        'H': img height, 'W': img width. PyTorch uses 'NCHW', while TensorFlow 
        uses 'NHWC'.
      
    """
    self.resize_size = resize_size
    self.crop_size = crop_size
    self.img_mean = img_mean
    self.scale = scale
    self.img_std = img_std
    self.check_mirror_type(mirror_type)
    self.mirror_type = mirror_type
    self.check_batch_dims(batch_dims)
    self.batch_dims = batch_dims

  def __call__(self, im):
    return self.pre_process_img(im)

  @staticmethod
  def check_mirror_type(mirror_type):
    assert mirror_type in [None, 'random', 'always']

  @staticmethod
  def check_batch_dims(batch_dims):
    # 'N': batch size, 'C': num channels, 'H': img height, 'W': img width
    # PyTorch uses 'NCHW', while TensorFlow uses 'NHWC'.
    assert batch_dims in ['NCHW', 'NHWC']

  def set_mirror_type(self, mirror_type):
    self.check_mirror_type(mirror_type)
    self.mirror_type = mirror_type

  def pre_process_img(self, im):
    """Pre-process image. `im` is a numpy array returned by 
    matplotlib.pyplot.imread()."""
    # Resize.
    if self.resize_size is not None:
      im = resize_img(im, self.resize_size)
    # Randomly crop a sub-image.
    if self.crop_size is not None:
      im = rand_crop_img(im, self.crop_size)
    # Subtract mean and scaled by 1/255.
    # im -= self.img_mean # This causes an error:
    # Cannot cast ufunc subtract output from dtype('float64') to
    # dtype('uint8') with casting rule 'same_kind'
    if self.img_mean is not None:
      im = im - np.array(self.img_mean)
    if self.scale:
      im = im / 255.
    if self.img_mean is not None and self.img_std is not None:
      im = im / np.array(self.img_std)
    # May mirror image.
    mirrored = False

    if self.mirror_type == 'always' \
        or (self.mirror_type == 'random' and np.random.uniform() > 0.5):
      im = im[:, ::-1, :]
      mirrored = True
    # The original image has dims 'HWC', transform it to 'CHW'.
    if self.batch_dims == 'NCHW':
      im = im.transpose(2, 0, 1)

    return im, mirrored


def calculate_img_mean(img_dir=None, pattern='*.jpg', img_paths=None,
                       mean_file=None):
  """Calculate the mean values of R, G and B.
  Args:
    img_dir: a dir containing images. If `img_paths` is provided, this is not 
    used.
    pattern: the file pattern for glob.glob()
    img_paths: a list of image paths
    mean_file: a file to save image mean. If None, results will not be saved.
  Returns:
    A numpy array with shape [3], for R, G, B mean value respectively.
  """

  # Get image paths.
  if img_paths is None:
    img_names = get_img_names(img_dir, pattern=pattern)
    img_paths = [osp.join(img_dir, name) for name in img_names]

  # Calculate mean.
  num_pixels = []
  values_sum = np.zeros([3])
  for path in img_paths:
    im = plt.imread(path)
    num_pixels.append(im.shape[0] * im.shape[1])
    values_sum += np.sum(np.sum(im, axis=0), axis=0)
  img_mean = values_sum / np.sum(num_pixels)

  # Write the mean to file.
  if mean_file is not None:
    mean_dir = osp.dirname(mean_file)
    if not osp.exists(mean_dir):
      os.makedirs(mean_dir)
    with open(mean_file, 'w') as f:
      pickle.dump(img_mean, f)
      print 'Saved image mean to file {}.'.format(mean_file)

  return img_mean


# deprecated
def partition_train_val_set(img_dir=None, pattern='*.jpg', img_paths=None,
                            val_prop=0.2, seed=2017, partition_file=None):
  """Partition the trainval set into train and val set. This function also
  returns the mapping from id to label, for use in network with identification 
  loss.
  Args:
    img_dir: a dir containing trainval images. If `img_paths` is provided, 
    this is not used.
    pattern: the file pattern for glob.glob()
    img_paths: a list of trainval image paths
    val_prop: the proportion of validation images
    seed: the random seed to reproduce the partition results
    partition_file: a pickle file containing partition results: a tuple 
    (trainval_img_paths, train_img_paths, val_img_paths, ids2labels) or 
    (trainval_img_names, train_img_names, val_img_names, ids2labels) for the 
    cases of `img_paths` being provided or not provided respectively. If None, 
    results will not be saved.
  Returns:
    The train+val, train, val partitions.
  """

  # Get image names or paths.
  if img_paths is None:
    img_names_or_paths = get_img_names(img_dir, pattern=pattern)
  else:
    img_names_or_paths = np.array(img_paths)

  # Partition the trainval set.

  # Both image paths and image names can be handled.
  ids = parse_ids(img_names_or_paths, is_path=True)
  val_inds = []
  unique_ids = np.unique(ids)
  # For each id, select a proportion of images for validation.
  for unique_id in unique_ids:
    # The indices where this id appears in list `ids`.
    inds = np.argwhere(unique_id == ids).squeeze()
    cnt = len(inds)
    val_cnt = int(cnt * val_prop)
    # Make sure that for each id, both train and val have at least two
    # images.
    if cnt - val_cnt < 2 or val_cnt < 2:
      continue
    # np.random.seed(seed)
    val_inds.append(inds[np.random.permutation(np.arange(cnt))[:val_cnt]])
  val_inds = np.hstack(val_inds)
  trainval_inds = np.arange(len(img_names_or_paths))
  train_inds = np.setdiff1d(trainval_inds, val_inds)

  # Mapping ids to labels.
  ids2labels = dict(zip(unique_ids, range(len(unique_ids))))

  partitions = (img_names_or_paths, img_names_or_paths[train_inds],
                img_names_or_paths[val_inds], ids2labels)

  # Write the partitions to file.
  if partition_file is not None:
    part_dir = osp.dirname(partition_file)
    if not osp.exists(part_dir):
      os.makedirs(part_dir)
    with open(partition_file, 'w') as f:
      pickle.dump(partitions, f)
      print 'Saved partitions to file {}.'.format(partition_file)

  return partitions


def partition_train_val_set_v2(img_dir=None, pattern='*.jpg', img_paths=None,
                               val_prop=0.15, seed=2017, partition_file=None):
  """Partition the trainval set into train and val set. This function also
  returns the mapping from id to label, for use in network with identification 
  loss. Besides, in the val set, query and gallery sets are also separated.
  Args:
    img_dir: a dir containing trainval images. If `img_paths` is provided, 
    this is not used.
    pattern: the file pattern for glob.glob()
    img_paths: a list of trainval image paths
    val_prop: the proportion of validation ids
    seed: the random seed to reproduce the partition results
    partition_file: a pickle file containing partition results -- a tuple 
      (trainval_img_names_or_paths, train_img_names_or_paths,
      val_img_names_or_paths, query_img_names_or_paths,
      gallery_img_names_or_paths, trainval_ids2labels,
      train_ids2labels) 
    If `partition_file` is None, results will not be saved.
  Returns:
    The tuple mentioned above.
  """

  # Get image names or paths.
  if img_paths is None:
    img_names_or_paths = get_img_names(img_dir, pattern=pattern)
  else:
    img_names_or_paths = np.array(img_paths)

  # Select a proportion of ids as val set.

  np.random.shuffle(img_names_or_paths)
  # Both image paths and image names can be handled by parse_ids() and
  # parse_cams().
  ids = parse_ids(img_names_or_paths, is_path=True)
  cams = parse_cams(img_names_or_paths, is_path=True)
  unique_ids = np.unique(ids)
  np.random.shuffle(unique_ids)

  # Query indices and gallery indices in the trainval set.
  query_inds = []
  gallery_inds = []
  val_ids = []

  num_selected_ids = 0
  for unique_id in unique_ids:
    query_inds_ = []
    # The indices of this id in trainval set.
    inds = np.argwhere(unique_id == ids).flatten()
    # The cams that this id has.
    unique_cams = np.unique(cams[inds])
    # For each cam, select one image for query set.
    for unique_cam in unique_cams:
      query_inds_.append(
        inds[np.argwhere(cams[inds] == unique_cam).flatten()[0]])
    gallery_inds_ = list(set(inds) - set(query_inds_))
    # For each query image, if there is no same-id different-cam images in
    # gallery, put it in gallery.
    for query_ind in query_inds_:
      if len(gallery_inds_) == 0 \
          or len(np.argwhere(cams[gallery_inds_] != cams[query_ind])
                     .flatten()) == 0:
        query_inds_.remove(query_ind)
        gallery_inds_.append(query_ind)
    # If no query image is left, leave this id in train set.
    if len(query_inds_) == 0:
      continue
    query_inds.append(query_inds_)
    gallery_inds.append(gallery_inds_)
    val_ids.append(unique_id)
    num_selected_ids += 1
    if num_selected_ids >= len(unique_ids) * val_prop:
      break

  query_inds = np.hstack(query_inds)
  gallery_inds = np.hstack(gallery_inds)
  val_inds = np.hstack([query_inds, gallery_inds])
  trainval_inds = np.arange(len(img_names_or_paths))
  train_inds = np.setdiff1d(trainval_inds, val_inds)

  val_ids = np.array(val_ids)
  train_ids = np.setdiff1d(unique_ids, val_ids)

  # Mapping ids to labels.
  trainval_ids2labels = dict(zip(unique_ids, range(len(unique_ids))))
  train_ids2labels = dict(zip(train_ids, range(len(train_ids))))

  partitions = (img_names_or_paths, img_names_or_paths[train_inds],
                img_names_or_paths[val_inds], img_names_or_paths[query_inds],
                img_names_or_paths[gallery_inds], trainval_ids2labels,
                train_ids2labels)

  # Write the partitions to file.
  if partition_file is not None:
    part_dir = osp.dirname(partition_file)
    if not osp.exists(part_dir):
      os.makedirs(part_dir)
    with open(partition_file, 'w') as f:
      pickle.dump(partitions, f)
      print 'Saved partitions to file {}.'.format(partition_file)

  return partitions


# Not used
def partition_query_gallery_set(img_names_or_paths=None, query_prop=0.5,
                                seed=2017):
  """Partition a set into query and gallery sets.
  Args:
    img_names_or_paths: a list of image names or paths
    query_prop: the proportion of query images
    seed: the random seed to reproduce the partition results
  Returns:
    The query, gallery partitions.
  """

  # Both image paths and image names can be handled.
  ids = parse_ids(img_names_or_paths, is_path=True)
  query_inds = []
  unique_ids = np.unique(ids)
  # For each id, select a proportion of images as query set.
  for unique_id in unique_ids:
    # The indices where this id appears in list `ids`.
    inds = np.argwhere(unique_id == ids).squeeze()
    cnt = len(inds)
    query_cnt = int(cnt * query_prop)
    gallery_cnt = cnt - query_cnt
    # Avoid the case in which an id only appears in query set.
    if cnt < 2 or gallery_cnt == 0 or query_cnt == 0:
      print '[Warning] id {} is only placed in gallery set.'.format(unique_id)
      continue

    # np.random.seed(seed)
    query_inds.append(inds[np.random.permutation(np.arange(cnt))[:query_cnt]])
  query_inds = np.hstack(query_inds)
  all_inds = np.arange(len(img_names_or_paths))
  gallery_inds = np.setdiff1d(all_inds, query_inds)

  partitions = (img_names_or_paths[query_inds],
                img_names_or_paths[gallery_inds])

  return partitions


def ids_to_labels(ids, mapping_dict):
  """Map ids to labels."""

  return np.array([mapping_dict[id_] for id_ in ids])


def normalize(nparray, order=2, axis=0):
  """Normalize a N-D numpy array along the specified axis."""

  norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
  return nparray / (norm + np.finfo(np.float32).eps)


def compute_dist(array1, array2):
  """Compute the euclidean distance of all pairs.
  Args:
    array1: numpy array with shape [m1, n], each row normalized to unit length
    array2: numpy array with shape [m2, n], each row normalized to unit length
  Returns:
    numpy array with shape [m1, m2]
  """

  cosine_dist = np.matmul(array1, array2.T)
  # It holds because vectors are with unit length.
  euclidean_dist = 2.0 * (1 - cosine_dist)

  return euclidean_dist


def enqueue(queue, func, event):
  """This thread function pre-fetches batches and stores them to a queue.
  Args:
    queue: The queue to store stuff
    func: This thread call this function and put the returned results into
    the queue.
    event: A threading.Event object, if set, this thread should stop.
  """

  while not event.isSet():
    if not queue.full():
      queue.put(item=func())
