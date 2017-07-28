import os.path as osp
import Config as cfg
import torch
from torch.autograd import Variable
import MyGoogleNet
import MTestIdLoss
import utils

if __name__ == '__main__':
    import Config
    # All training and testing configurations.
    cfg = Config.Config
    TVT, TMO = utils.set_device(cfg.device_id)

class Test(object):
    """Test reid performance on one or more trained models."""
    def __init__(self):

        ###########
        # Models  #
        ###########

        self.googlenet = MyGoogleNet.inception_v3(pretrained=False, num_classes=cfg.num_classes)
        self.models = [self.googlenet]

        ###########################################
        # May Transfer Models to Specified Device #
        ###########################################

        TMO(self.models)

        ############
        # Test Set #
        ############

        def feature_func(ims):
            """A function to be called in the test/val set, to extract features."""
            # Set eval mode
            utils.may_set_mode(self.models, 'eval')
            ims = TVT(Variable(torch.from_numpy(ims).float()))
            feats = self.googlenet(ims)
            feats = feats.data.cpu().numpy()
            return feats

        self.test_set = MTestIdLoss.MTestIdLoss(
            dataset_root=cfg.dataset_root,
            feature_func=feature_func,
            batch_size=cfg.test_batch_size,
            resize_size=cfg.im_resize_size,
            crop_size=cfg.im_crop_size,
            img_mean=cfg.im_mean,
            mg_mean_file=cfg.im_mean_file,
            img_std=cfg.im_std,
            img_std_file=cfg.im_std_file,
            scale=cfg.scale_im,
            mirror_type=None,
            batch_dims='NCHW',
            num_prefetch_threads=cfg.prefetch_threads
        )

    def test(self):

        for ckpt in cfg.ckpts:

            # Load state dicts to cpu
            assert osp.isfile(ckpt), "=> no checkpoint found at '{}'".format(ckpt)
            state_dicts = torch.load(ckpt, map_location=(lambda storage, loc: storage))
            print "=> loaded checkpoint '{}'".format(ckpt)

            # Initialize weights
            utils.load_module_state_dict(self.googlenet, state_dicts[cfg.ckpt_googlenet_ind])
            self.test_set.eval_single_query(True)
            self.test_set.eval_multi_query(False)

        self.test_set.stop_prefetching_threads()


Test().test()
