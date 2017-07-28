import os


def get_num_classes(train_val_partition_file):
  import pickle
  with open(train_val_partition_file, 'r') as f:
    partitions = pickle.load(f)
  num_classes = len(partitions[-1].keys())
  return num_classes


class Config:

  # gpu id, or -1 which means using cpu
  device_id = 0
  pytorch_bn_momentum = 0.001

  # How many threads to prefetch data. >= 1
  prefetch_threads = 4

  ###########
  # Dataset #
  ###########

  dataset_root = os.path.expanduser('~/practise_pytorch/Market1502/Market-1501-v15.09.15')
  train_val_partition_file = 'train_val_partition_file.pkl'
  train_val_part = ['train', 'trainval'][1]
  # num of classes in reid net. It should change according to using train or
  # trainval part or after repartition the trainval set
  if train_val_part == 'train':
    num_classes = get_num_classes(train_val_partition_file)
  else:
    num_classes = 751

  ####################
  # Image Processing #
  ####################

  im_resize_size = (224, 224)
  im_crop_size = None
  # If `None` then image mean is calculated from trainval set;
  # Or it can be length-3 list or numpy array.
  # `im_mean` has priority higher than `im_mean_file`.
  im_mean = [124, 117, 104]
  im_mean_file = 'img_mean.pkl'
  # Whether to scale by 1/255
  scale_im = True
  # Whether to divide by std, set to `None` to disable.
  # Dividing is applied only after subtracting mean.
  im_std = [0.229, 0.224, 0.225]
  im_std_file = None
  mirror_im = True

  ####################
  # Experiment Dirs  #
  ####################

  # The root dir of logs.
  experiment_root = 'experiments'
  pre_reid_dir = os.path.join(experiment_root, 'pre_reid')

  ####################
  # Pretraining ReID #
  ####################

  pre_reid_dropout_rate = 0.6

  # for conv layers of resnet
  pre_reid_ft_base_lr = 0.001
  # for fc layer of resnet
  pre_reid_fc_weight_base_lr = 0.001
  pre_reid_fc_bias_base_lr = 0.001 * 2
  # If not want to decay, set to a large number.
  pre_reid_lr_decay_epochs = 30
  pre_reid_momentum = 0.9
  pre_reid_weight_decay = 0.0005

  # Number of epochs to train
  pre_reid_num_epochs = 80
  # How often (in batches) to log
  pre_reid_log_steps = 20

  # How often (in epochs) to save ckpt
  pre_reid_epochs_per_saving_ckpt = 5
  pre_reid_ckpt_saving_tmpl = os.path.join(pre_reid_dir, 'epoch_{}.pth')

  pre_reid_resume = False
  # The epoch to resume from
  pre_reid_resume_ep = 0
  pre_reid_resume_ckpt = pre_reid_ckpt_saving_tmpl.format(pre_reid_resume_ep)

  pre_reid_train_set_batch_size = 16
  pre_reid_val_set_batch_size = 16

  ###########
  # Testing #
  ###########

  # The checkpoint files saved by the training reid task.
  # Here test on every 5 epochs.
  ckpts = [pre_reid_ckpt_saving_tmpl.format(i) for i in range(5, 85, 5)]
  # The ind of reidBot/reidTop ckpt in a ckpt object (a list).
  ckpt_reidBot_ind = 0
  ckpt_reidTop_ind = 1

  test_batch_size = 16
