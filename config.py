# encoding: utf-8
import warnings
import numpy as np


class DefaultConfig(object):
    seed = 0

    # dataset options
    dataset = 'Market1501'
    datatype = 'person'
    mode = 'retrieval'
    # optimization options
    loss = 'triplet'  #triplet triplet+center
    optim = 'adam'
    max_epoch = 60
    train_batch = 32
    test_batch = 32
    adjust_lr = False
    lr = 0.001
    gamma = 0.1
    weight_decay = 5e-4
    momentum = 0.9
    random_crop = False
    margin = None
    num_instances = 4
    num_gpu = 1
    evaluate = False
    savefig = None 
    re_ranking = True

    # model options
    model_name = 'strongBaseline'  # triplet, softmax_triplet, bfe, ide, strongBaseline
    last_stride = 1
    pretrained_model = None
    
    # miscs
    print_freq = 30
    eval_step = 10
    save_dir = './pytorch-ckpt/market'
    workers = 10
    start_epoch = 0
    best_rank = -np.inf

    #center loss
    center_weight=0.0005
    center_lr=0.5


    def _parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)
            if 'cls' in self.dataset:
                self.mode='class'
            if 'market' in self.dataset or 'cuhk' in self.dataset or 'duke' in self.dataset:
                self.datatype = 'person'
            elif 'cub' in self.dataset:
                self.datatype = 'cub'
            elif 'car' in self.dataset:
                self.datatype = 'car'
            elif 'clothes' in self.dataset:
                self.datatype = 'clothes'
            elif 'product' in self.dataset:
                self.datatype = 'product'

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in DefaultConfig.__dict__.items()
                if not k.startswith('_')}

opt = DefaultConfig()
