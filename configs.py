import torch
import numpy as np
import torch.nn as nn


class TrainConfig:
    def __init__(self):

        self.num_of_clients = 10

        # train dataset setup
        self.batch_size = 64
        self.shuffle = False
        self.collate_fn = None
        self.batch_sampler = None
        self.sampler = None
        self.num_workers = 0
        self.pin_memory = False
        self.drop_last = True
        self.timeout = 0
        self.worker_init_fn = None

        # train network setup
        self.epochs = 1000

        # fed setup
        self.com_epochs = 100

        # train dataset setup
        self.dataset = "MNIST"
        # self.dataset = "CIFAR10"

        # generated samples store
        self.store_generated_root = 'results/'
        self.sample_rate = 100

        # CUDA setup
        self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        # make use of GPU environment when existing

        self.no_cuda = not torch.cuda.is_available()

        # others
        self.valid_loss_min = np.Inf
        self.lr = 0.01
        self.momentum = 0.5
        self.seed = 1


    # @property

    @property
    def order(self):
        return not self.shuffle

    @property
    def num_data_owned_setup(self):
        return int(50000 / self.num_of_clients)

    # model setup
    @property
    def latent_dim(self):
        if self.dataset == "MNIST":
            return 100
        elif self.dataset == "CIFAR10":
            return 100

    @property
    def n_classes(self):
        if self.dataset == "MNIST":
            return 10
        elif self.dataset == "CIFAR10":
            return 10

    @property
    def img_size(self):
        if self.dataset == "MNIST":
            return 28
        elif self.dataset == "CIFAR10":
            return 32

    @property
    def channels(self):
        if self.dataset == "MNIST":
            return 1
        elif self.dataset == "CIFAR10":
            return 3

    @property
    def img_shape(self):
        if self.dataset == "MNIST":
            return (self.channels, self.img_size, self.img_size)
        elif self.dataset == "CIFAR10":
            return (self.channels, self.img_size, self.img_size)

    # generated samples store
    @property
    def if_img(self):
        if self.sample_rate != 0:
            return True
        else:
            return False


class TestConfig:
    def __init__(self):
        pass
