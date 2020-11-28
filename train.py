import torch
import torch.nn as nn
import torch.optim as optim
import copy

from torch.autograd import Variable

from init import init_federated
from models.fed_merge import fedavg
import numpy as np


def train_epoch(optimizers_dict, loss_func_dict, epoch, client, config):
    generator = client.generator
    discriminator = client.discriminator
    generator.train()
    discriminator.train()

    optimizers_g = optimizers_dict['g']
    optimizers_d = optimizers_dict['d']

    device = config.device
    train_loader = client.train_data_load()

    for batch_idx, (sample, target) in enumerate(train_loader):
        sample, target = sample.to(device), target.to(device)
        data_v = Variable(sample)
        target_v = Variable(target)

        loss_func_g = loss_func_dict['g']
        loss_func_d = loss_func_dict['d']

        # Adversarial ground truths
        if config.no_cuda:
            from torch import FloatTensor, LongTensor
        else:
            from torch.cuda import FloatTensor, LongTensor

        valid = Variable(FloatTensor(config.batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(config.batch_size, 1).fill_(0.0), requires_grad=False)

        # #########
        # ---------
        # Train Generator
        # ---------
        optimizers_g.zero_grad()

        # Sample noise and labels
        z = Variable(FloatTensor(np.random.normal(0, 1, (config.batch_size, config.latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, config.n_classes, config.batch_size)))

        gen_samples = generator(z, gen_labels)

        output = discriminator(gen_samples, gen_labels)
        loss_g = loss_func_g(output, valid)

        loss_g.backward()
        optimizers_g.step()

        # ---------
        # Train Generator
        # ---------
        # #########

        # #########
        # ---------
        # Train Discriminator
        # ---------
        optimizers_d.zero_grad()

        # Loss for real images
        validity_real = discriminator(data_v, target_v)
        d_real_loss = loss_func_d(validity_real, valid)

        # Loss for fake images
        validity_fake = discriminator(gen_samples.detach(), gen_labels)
        d_fake_loss = loss_func_d(validity_fake, fake)

        # Total discriminator loss
        loss_d = (d_real_loss + d_fake_loss) / 2

        loss_d.backward()
        optimizers_d.step()

        # ---------
        # Train Discriminator
        # ---------
        # #########

        train_loss_g = loss_g.data.item()
        train_loss_d = loss_d.data.item()

        # if store generated images
        if config.if_img:
            if epoch % config.sample_rate == 0 and batch_idx + 1 == len(train_loader):
                client.store_train_samples(sample=gen_samples, root=config.store_generated_root,
                                           img_name=str(epoch) + '.png')

    return {'g': generator.state_dict(), 'd': discriminator.state_dict()}, {'g': train_loss_g,
                                                                            'd': train_loss_d}


def train_federated(config, clients, server):
    count = 1

    for epoch in range(1, config.epochs + 1):
        # A parameter collector
        para_collector_g = []
        para_collector_d = []

        # All clients update their local models
        for client in clients:
            g_optimizer = optim.Adam(client.generator.parameters())
            d_optimizer = optim.Adam(client.discriminator.parameters())
            loss_func = nn.MSELoss()

            # This func would return the parameters of the model trained in this turn
            para_dict, train_loss_dict = train_epoch(
                optimizers_dict={"g": g_optimizer, "d": d_optimizer},
                loss_func_dict={"g": loss_func, "d": loss_func},
                epoch=epoch,
                client=client,
                config=config
            )

            # echo
            print('Client {}\tTrain Epoch: {}\tLoss: G:{:6f}, D:{:6f}'.format(client.id,
                                                                              epoch,
                                                                              train_loss_dict['g'],
                                                                              train_loss_dict['d']))

            # log write for this client
            client.log_write(epoch=epoch, loss_g=train_loss_dict['g'], loss_d=train_loss_dict['d'])

            if epoch % config.com_epochs == 0:
                para_collector_g.append(copy.deepcopy(para_dict['g']))
                para_collector_d.append(copy.deepcopy(para_dict['d']))

        # federated!
        if epoch % config.com_epochs == 0:
            # merge + update global
            para_global_g = fedavg(para_collector_g)
            para_global_d = fedavg(para_collector_d)

            server.generator.load_state_dict(para_global_g)
            server.discriminator.load_state_dict(para_global_d)

            # echo
            print("Server's model has been update, Fed No.: {}".format(count))
            count += 1

            # model download local
            for client in clients:
                client.load_model(generator=copy.deepcopy(server.generator),
                                  discriminator=copy.deepcopy(server.discriminator))
                print("Client {}'s model has been updated from the server, Fed No.{}".format(client.id,
                                                                                             count))
    # Save the server model
    server.save_model()
    print("Global model has been saved on the server!")


if __name__ == '__main__':
    clients, server, config = init_federated()
    train_federated(config, clients, server)
