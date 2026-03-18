import importlib
import os
import random
import numpy as np
import torch

from data_provider.data_factory import data_provider


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self._set_seed()

        self.model = self._build_model()

        path = os.path.join(self.args.checkpoints, args.model_id)
        self.checkpoint_path = path
        if not os.path.exists(path):
            os.makedirs(path)

        self.model = self.model.to(self.device)
        total_params = count_parameters(self.model)
        print(f"Total trainable parameters: {total_params}")

    def _set_seed(self):
        seed = self.args.seed
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    def _build_model(self):
        model_module = importlib.import_module(f'models.{self.args.model}')
        model = model_module.Model(self.args).float()
        return model

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            )
            device = torch.device("cuda:{}".format(self.args.gpu))
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _get_data(self, flag):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
