import os
import torch
import numpy as np

class Exp_Basic(object):
    def __init__(self, args, ii):
        self.args = args
        self.ii = ii
        self.device = self._acquire_device()
        _, _, leaf_num, num_of_tree = self._get_data(flag='test')
        args.leaf_num_ = leaf_num
        args.num_of_tree = num_of_tree

        self.model = self._build_model(leaf_num, num_of_tree, self.device).to(self.device)

        # For classification
        self.root_path = args.root_path

    def _build_model(self, leaf_num, num_of_tree, device):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
