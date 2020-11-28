import torch
import numpy as np
from torch.utils import data
from torchvision.utils import save_image

from models.cnn import Cnn
import copy
from utils import *
import pandas as pd

from models.cGan import Generator, Discriminator


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


class Client:
    def __init__(self, id, config):
        self.id = id

        self.local_dir = 'clients/' + str(id) + '/'
        dir_setup(self.local_dir)

        self.dataset = []
        dir_setup(self.local_dir + 'dataset/')

        self.label = []
        dir_setup(self.local_dir + 'label/')

        dir_setup(self.local_dir + config.store_generated_root)

        self.generator = Generator(config).to(config.device)
        dir_setup(self.local_dir + 'model/')
        self.generator_name = "generator.pkl"

        self.discriminator = Discriminator(config).to(config.device)
        dir_setup(self.local_dir + 'model/')
        self.discriminator_name = "discriminator.pkl"

        # The number of samples the client owns (before really load data)
        self.num_data_owned_setup = 0

        # self config
        self.config = config

    def load_data(self, data_label_list):
        self.dataset.append(data_label_list[0])
        self.label.append(data_label_list[1])

    def load_model_from_path(self, model_path):
        self.generator = torch.load(model_path)

    def load_model(self, generator, discriminator):
        self.generator = copy.deepcopy(generator)
        self.discriminator = copy.deepcopy(discriminator)

    def train_data_load(self):
        # Transform to torch tensors
        config = self.config

        tensor_samples = torch.stack([s.float() for s in self.dataset])
        tensor_targets = torch.stack([t for t in self.label])

        train_dataset = data.TensorDataset(tensor_samples, tensor_targets)
        return data.DataLoader(dataset=train_dataset,
                               batch_size=config.batch_size,
                               shuffle=config.shuffle,
                               collate_fn=config.collate_fn,
                               batch_sampler=config.batch_sampler,
                               num_workers=config.num_workers,
                               pin_memory=config.pin_memory,
                               drop_last=config.drop_last,
                               timeout=config.timeout,
                               worker_init_fn=config.worker_init_fn)

    def num_data_owned(self):
        return len(self.dataset)

    # client writes logs
    def log_write(self, epoch, loss_g, loss_d):
        loss_data_frame = pd.DataFrame(columns=None, index=[epoch], data=[[loss_d, loss_g]])
        loss_data_frame.to_csv("clients/" + str(self.id) + "/" + "log.csv", mode='a', header=False)

    # store generated samples during train process by some rate
    def store_train_samples(self, sample, root, img_name):
        shape = self.config.img_shape
        sample_images = sample.reshape(sample.size(0), shape[0], shape[1], shape[2])
        save_image(denorm(sample_images),
                   os.path.abspath(os.getcwd()) + "/clients/" + str(self.id) + "/" + root + img_name)
