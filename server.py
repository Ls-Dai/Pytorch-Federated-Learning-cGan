from models.cnn import Cnn
import torch

from utils import dir_setup
from models.cGan import Generator, Discriminator


class Server:
    def __init__(self, id_num, config):
        self.model_name = None
        self.id = id_num

        self.local_dir = "servers/" + str(self.id) + "/"
        dir_setup(self.local_dir)

        self.model_dir = self.local_dir + "model/"
        dir_setup(self.model_dir)

        self.generator = Generator(config).to(config.device)

        self.discriminator = Discriminator(config).to(config.device)

        self.generator_name = "generator.pkl"

        self.discriminator_name = "discriminator.pkl"

        self.config = config

    def save_model(self):
        torch.save(self.generator, self.model_dir + self.generator_name)
        torch.save(self.discriminator, self.model_dir + self.discriminator_name)

    def load_model(self):
        if self.config.no_cuda:
            self.generator = torch.load(self.model_dir + self.generator_name, map_location=torch.device("cpu"))
            self.discriminator = torch.load(self.model_dir + self.discriminator_name, map_location=torch.device("cpu"))
        else:
            self.generator = torch.load(self.model_dir + self.generator_name)
            self.discriminator = torch.load(self.model_dir + self.discriminator_name)
