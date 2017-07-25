import os
import os.path as osp
import pickle
import numpy as np
from scipy import io
import threading
import Queue
import time

import torch
from torch.autograd import Variable

def load_state_dict(model, state_dict):
  """Copies parameters and buffers from `state_dict` into `model` and its
  descendants. The keys of `state_dict` NEED NOT exactly match the keys
  returned by model's `state_dict()` function. For dict key mismatch, just
  skip it; for copying error, just output warnings and proceed.

  Arguments:
    model: A torch.nn.Module object.
    state_dict (dict): A dict containing parameters and persistent buffers.
  Note:
    This is copied and modified from torch.nn.modules.module.load_state_dict().
    Just to allow name mismatch between `model.state_dict()` and `state_dict`.
  """
  import warnings
  from torch.nn import Parameter

  own_state = model.state_dict()
  for name, param in state_dict.items():
    if name not in own_state:
      warnings.warn('Skipping unexpected key "{}" in state_dict'.format(name))
      continue
    if isinstance(param, Parameter):
      # backwards compatibility for serialized parameters
      param = param.data
    try:
      own_state[name].copy_(param)
    except Exception, msg:
      warnings.warn("Error occurs when copying from state_dict['{}']: {}"
                    .format(name, str(msg)))

  missing = set(own_state.keys()) - set(state_dict.keys())
  if len(missing) > 0:
    warnings.warn(
      "Keys not found in state_dict and thus not overwritten: '{}'"
        .format(missing))

# Useless
class MyImage(object):
  def __init__(self, im, im_name, id, cam, mask_inds):
    self.im = im
    self.im_name = im_name
    self.id = id
    self.cam = cam
    self.mask_inds = mask_inds


def load_pickle(path):
  """Check and load pickle object."""
  assert osp.exists(path)
  with open(path, 'r') as f:
    ret = pickle.load(f)
  return ret


def save_pickle(obj, path):
  """Create dir and save file."""
  may_make_dir(osp.dirname(path))
  with open(path, 'w') as f:
    pickle.dump(obj, f)


def save_mat(ndarray, path):
  """Save a numpy ndarray as .mat file."""
  io.savemat(path, dict(ndarray=ndarray))


def to_scalar(vt):
  """Transform a length-1 Variable or Tensor to scalar. 
  Suppose tx is a torch Tensor with shape tx.size() = torch.Size([1]), 
  then npx = tx.cpu().numpy() has shape (1,), not 1."""
  if isinstance(vt, Variable):
    return vt.data.cpu().numpy()[0]
  if torch.is_tensor(vt):
    return vt.cpu().numpy()[0]
  raise TypeError('Input should be a variable or tensor')


def transfer_optim_state(state, device_id=-1):
  """Transfer an optimizer.state to cpu or specified gpu, which means 
  transferring tensors of the optimizer.state to specified device. 
  The modification is in place for the state.
  Args:
    state: An torch.optim.Optimizer.state
    device_id: gpu id, or -1 which means transferring to cpu
  """
  for key, val in state.items():
    if isinstance(val, dict):
      transfer_optim_state(val, device_id=device_id)
    elif isinstance(val, Variable):
      raise RuntimeError("Oops, state[{}] is a Variable!".format(key))
    elif isinstance(val, torch.nn.Parameter):
      raise RuntimeError("Oops, state[{}] is a Parameter!".format(key))
    else:
      try:
        if device_id == -1:
          state[key] = val.cpu()
        else:
          state[key] = val.cuda(device=device_id)
      except:
        pass


def may_transfer_optims(optims, device_id=-1):
  """Transfer optimizers to cpu or specified gpu, which means transferring 
  tensors of the optimizer to specified device. The modification is in place 
  for the optimizers.
  Args:
    optims: A list, which members are either torch.nn.optimizer or None.
    device_id: gpu id, or -1 which means transferring to cpu
  """
  for optim in optims:
    if isinstance(optim, torch.optim.Optimizer):
      transfer_optim_state(optim.state, device_id=device_id)


def may_transfer_modules_optims(modules_and_or_optims, device_id=-1):
  """Transfer optimizers/modules to cpu or specified gpu.
  Args:
    modules_and_or_optims: A list, which members are either torch.nn.optimizer 
      or torch.nn.Module or None.
    device_id: gpu id, or -1 which means transferring to cpu
  """
  for item in modules_and_or_optims:
    if isinstance(item, torch.optim.Optimizer):
      transfer_optim_state(item.state, device_id=device_id)
    elif isinstance(item, torch.nn.Module):
      if device_id == -1:
        item.cpu()
      else:
        item.cuda(device_id=device_id)
    elif item is not None:
      print '[Warning] Invalid type {}'.format(item.__class__.__name__)


class TransferVarTensor(object):
  """Return a copy of the input Variable or Tensor on specified device."""

  def __init__(self, device_id=-1):
    self.device_id = device_id

  def __call__(self, var_or_tensor):
    return var_or_tensor.cpu() if self.device_id == -1 \
      else var_or_tensor.cuda(self.device_id)


class TransferModulesOptims(object):
  """Transfer optimizers/modules to cpu or specified gpu."""

  def __init__(self, device_id=-1):
    self.device_id = device_id

  def __call__(self, modules_and_or_optims):
    may_transfer_modules_optims(modules_and_or_optims, self.device_id)


def set_device(sys_device_id):
  """
  This is a util function for the case of using single gpu. It sets that gpu
  to be the only visible device and returns some wrappers to transferring
  Variables/Tensors and Modules/Optimizers.
  Args:
    sys_device_id: which gpu device to use, if -1, it means cpu
  Returns:
    
  """
  device_id = -1
  cuda = (sys_device_id != -1)
  if cuda:
    # CUDA_VISIBLE_DEVICE is a list, and device_id is the index of its members.
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(sys_device_id)
    device_id = 0
  TVT = TransferVarTensor(device_id)
  TMO = TransferModulesOptims(device_id)
  return TVT, TMO


def may_load_modules_optims_state_dicts(modules_and_or_optims, ckpt_file,
                                        load_to_cpu=True):
  """Load state_dict's of modules/optimizers from file.
  Args:
    modules_and_or_optims: A list, which members are either torch.nn.optimizer 
      or torch.nn.Module or None.
    ckpt_file: The file path.
    load_to_cpu: Boolean. Whether to transform tensors in modules/optimizers 
    to cpu type.
  """
  map_location = (lambda storage, loc: storage) if load_to_cpu else None
  state_dicts = torch.load(ckpt_file, map_location=map_location)
  for m, sd in zip(modules_and_or_optims, state_dicts):
    if None not in [m, sd]:
      m.load_state_dict(sd)


def may_save_modules_optims_state_dicts(modules_and_or_optims, ckpt_file):
  """Save state_dict's of modules/optimizers to file. 
  Args:
    modules_and_or_optims: A list, which members are either torch.nn.optimizer 
      or torch.nn.Module or None.
    ckpt_file: The file path.
  Note:
    torch.save() reserves device type and id of tensors to save, so you have
    to inform torch.load() to load these tensors to cpu or your desired gpu, 
    if you change devices.
  """
  state_dicts = [None] * len(modules_and_or_optims)
  for i, m in enumerate(modules_and_or_optims):
    if m is not None:
      state_dicts[i] = m.state_dict()
  may_make_dir(osp.dirname(ckpt_file))
  torch.save(state_dicts, ckpt_file)


def load_module_state_dict(model, state_dict):
  """Copies parameters and buffers from `state_dict` into `model` and its 
  descendants. The keys of `state_dict` NEED NOT exactly match the keys 
  returned by model's `state_dict()` function. For dict key mismatch, just
  skip it; for copying error, just output warnings and proceed.

  Arguments:
    model: A torch.nn.Module object. 
    state_dict (dict): A dict containing parameters and persistent buffers.
  Note:
    This is copied and modified from torch.nn.modules.module.load_state_dict().
    Just to allow name mismatch between `model.state_dict()` and `state_dict`.
  """
  import warnings
  from torch.nn import Parameter

  own_state = model.state_dict()
  for name, param in state_dict.items():
    if name not in own_state:
      warnings.warn('Skipping unexpected key "{}" in state_dict'.format(name))
      continue
    if isinstance(param, Parameter):
      # backwards compatibility for serialized parameters
      param = param.data
    try:
      own_state[name].copy_(param)
    except Exception, msg:
      warnings.warn("Error occurs when copying from state_dict['{}']: {}"
                    .format(name, str(msg)))

  missing = set(own_state.keys()) - set(state_dict.keys())
  if len(missing) > 0:
    warnings.warn(
      "Keys not found in state_dict and thus not overwritten: '{}'"
        .format(missing))


def may_set_mode(maybe_modules, mode):
  """maybe_modules: an object or a list of objects."""
  assert mode in ['train', 'eval']
  if not is_iterable(maybe_modules):
    maybe_modules = [maybe_modules]
  for m in maybe_modules:
    if isinstance(m, torch.nn.Module):
      if mode == 'train':
        m.train()
      else:
        m.eval()


def may_make_dir(path):
  """
  Args:
    path: a dir, or result of `osp.dirname(file_path)`
  Note:
    `osp.exists('')` returns `False`, while `osp.exists('.')` returns `True`!
  """

  # This clause has mistakes:
  # if path is None or '':

  if path in [None, '']:
    return
  if not osp.exists(path):
    os.makedirs(path)


def is_iterable(obj):
  return hasattr(obj, '__len__')


def adjust_lr(param_groups=None, base_lrs=None, decay_epochs=None, epoch=None,
              verbose=False):
  """Decay the learning rates in a staircase manner.
  Args:
    param_groups: typically returned by `some_optimizer.param_groups`
    base_lrs: a scalar or a list
    decay_epochs: a scalar or a list
    epoch: the current epoch number
  Returns:
    lrs: the learning rates after adjusting
  """

  if not is_iterable(base_lrs):
    base_lrs = [base_lrs for _ in param_groups]
  if not is_iterable(decay_epochs):
    decay_epochs = [decay_epochs for _ in param_groups]

  lrs = []
  for param_group, base_lr, decay_epoch in \
      zip(param_groups, base_lrs, decay_epochs):
    lr = base_lr * (0.1 ** (epoch // decay_epoch))
    param_group['lr'] = lr
    lrs.append(lr)

  if verbose:
    print '===> lrs adjusted to', lrs

  return lrs


def adjust_lr_poly(param_groups=None, base_lrs=None, total_epochs=None,
                   epoch=None, pow=0.5, verbose=False):
  """Decay the learning rates using a polynomial curve.
  Args:
    param_groups: typically returned by `some_optimizer.param_groups`
    base_lrs: a scalar or a list
    total_epochs: a scalar
    epoch: the current epoch number
  Returns:
    lrs: the learning rates after adjusting
  """

  if not is_iterable(base_lrs):
    base_lrs = [base_lrs for _ in param_groups]

  lrs = []
  for param_group, base_lr in zip(param_groups, base_lrs):
    lr = base_lr * np.power(float(total_epochs - epoch) / total_epochs, pow)
    param_group['lr'] = lr
    lrs.append(lr)

  if verbose:
    print '===> lrs adjusted to', lrs

  return lrs


def make_sure_str_list(may_be_list):
  if isinstance(may_be_list, str):
    may_be_list = [may_be_list]
  return may_be_list


class Counter(object):
  """A thread safe counter."""

  def __init__(self, val=0, max_val=0):
    self._value = val
    self.max_value = max_val
    self._lock = threading.Lock()

  def reset(self):
    with self._lock:
      self._value = 0

  def set_max_value(self, max_val):
    self.max_value = max_val

  def increment(self):
    with self._lock:
      if self._value < self.max_value:
        self._value += 1
        incremented = True
      else:
        incremented = False
      return incremented, self._value

  def get_value(self):
    with self._lock:
      return self._value


class Enqueuer(object):
  def __init__(self, get_element, num_elements, num_threads=1, queue_size=20):
    """
    Args:
      get_element: a function that takes a pointer and returns an element
      num_elements: total number of elements to put into the queue
      num_threads: num of parallel threads, >= 1
      queue_size: the maximum size of the queue. Set to some positive integer 
        to save memory, otherwise, set to 0. 
    """
    self.get_element = get_element
    self.num_threads = num_threads
    self.queue = Queue.Queue(maxsize=queue_size)
    # The pointer shared by threads.
    self.ptr = Counter(max_val=num_elements)
    # The event to wake up threads, it's set at the beginning of an epoch, and
    # after an epoch is enqueued, it's cleared.
    self.event = threading.Event()
    # The event to terminate the threads.
    self.stop_event = threading.Event()
    self.threads = []
    for _ in range(num_threads):
      thread = threading.Thread(target=self.enqueue)
      thread.start()
      self.threads.append(thread)

  def set_num_elements(self, num_elements):
    self.ptr.set_max_value(num_elements)

  def start(self):
    self.ptr.reset()
    self.event.set()

  def enqueue(self):
    while not self.stop_event.isSet():
      # If the enqueuing event is not set, the thread just waits.
      if not self.event.wait(0.5): continue
      # Increment the counter to claim that this element has been enqueued by
      # this thread.
      incremented, ptr = self.ptr.increment()
      if incremented:
        element = self.get_element(ptr-1)
        # This operation will wait until a free slot in the queue is available.
        self.queue.put(element)
      else:
        # When all elements are enqueued, let threads sleep to save resources.
        self.event.clear()

  def stop(self):
    self.stop_event.set()
    for thread in self.threads:
      thread.join()


class Prefetcher(object):
  """This helper class enables sample enqueuing and batch dequeuing, to speed 
  up batch fetching. It abstracts away the enqueuing and dequeuing logic."""
  def __init__(self, get_sample, dataset_size, batch_size, final_batch=True,
               num_threads=1, prefetch_size=100):
    """
    Args:
      get_sample: a function that takes a pointer (index) and returns a sample
      dataset_size: total number of samples in the dataset
      final_batch: True or False, whether to keep or drop the final incomplete 
        batch
      num_threads: num of parallel threads, >= 1
      prefetch_size: the maximum size of the queue. Set to some positive integer 
        to save memory, otherwise, set to 0.
    """
    self.full_dataset_size = dataset_size
    self.final_batch = final_batch
    final_sz = self.full_dataset_size % batch_size
    if not final_batch:
      dataset_size = self.full_dataset_size - final_sz
    self.dataset_size = dataset_size
    self.batch_size = batch_size
    self.enqueuer = Enqueuer(get_element=get_sample, num_elements=dataset_size,
                             num_threads=num_threads, queue_size=prefetch_size)
    # The pointer indicating whether an epoch has been fetched from the queue
    self.ptr = 0

  def set_batch_size(self, batch_size):
    """You can only change batch size at the beginning of a new epoch."""
    final_sz = self.full_dataset_size % batch_size
    if not self.final_batch:
      self.dataset_size = self.full_dataset_size - final_sz
    self.enqueuer.set_num_elements(self.dataset_size)
    self.batch_size = batch_size

  def next_batch(self):
    """Return a batch of samples, meanwhile indicates whether the epoch is 
    done. The purpose of this func is mainly to abstract away the loop and the
    corner verification logic.
    Returns:
      samples: a list of samples
      done: bool, whether the epoch is done
    """
    # Whether an epoch is done.
    done = False
    samples = []
    for _ in range(self.batch_size):
      # Indeed, `>` will not occur.
      if self.ptr >= self.dataset_size:
        done = True
        break
      else:
        self.ptr += 1
        sample = self.enqueuer.queue.get()
        samples.append(sample)
    # print 'queue size: {}'.format(self.enqueuer.queue.qsize())
    # Indeed, `>` will not occur.
    if self.ptr >= self.dataset_size:
      done = True
    return samples, done

  def start_ep_prefetching(self):
    """
    NOTE: Has to be called at the start of every epoch.
    """
    self.enqueuer.start()
    self.ptr = 0

  def stop(self):
    """After finishing using the dataset, or when existing the python main 
    program, this should be called."""
    self.enqueuer.stop()


def repeat_select_with_replacement(samples, num_select):
  """Repeat {select one sample with replacement} for n times.
  Args:
    samples: a numpy array with shape [num_samples]
    num_select: an int, number of selections
  Returns:
    all_select: a numpy array with shape [num_comb, num_select], where 
      num_comb is the number of all possible combinations
  """
  num_samples = len(samples)
  num_comb = num_samples ** num_select
  all_select = np.zeros([num_comb, num_select], dtype=samples.dtype)
  for i in range(num_select):
    num_repeat = num_samples ** (num_select - i - 1)
    # num_tile = num_comb / (num_repeat * num_samples)
    num_tile = num_samples ** i
    all_select[:, i] = np.tile(np.repeat(samples, num_repeat), num_tile)
  return all_select
