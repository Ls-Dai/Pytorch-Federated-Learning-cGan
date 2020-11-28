import os
import argparse

from configs import TrainConfig


def dir_setup(path):
    if not os.path.exists(path):
        os.makedirs(path)


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument("-e", type=int, default=1000, help='Epochs for the training')
    parser.add_argument("-c", type=int, default=10, help='Number of clients')
    parser.add_argument("-s", type=int, default=1, help='If Shuffle (IID)')
    parser.add_argument("-r", type=int, default=100, help='How many epochs for communication round')
    parser.add_argument("-d", type=str, default="MNIST", help='Dataset used for the training')
    parser.add_argument("-i", type=int, default=100, help='Train process visualization sample rate')

    args = parser.parse_args()

    config = TrainConfig()

    config.epochs = args.e
    config.num_of_clients = args.c
    config.shuffle = bool(args.s)
    config.com_epochs = args.r
    config.dataset = args.d
    config.sample_rate = args.i

    # echo, for linux > command to write to the logs, record the command
    print("python train.py -e {} -c {} -s {} -r {} -d {} -i {}".format(args.e,
                                                                       args.c,
                                                                       args.s,
                                                                       args.r,
                                                                       args.d,
                                                                       args.i,))

    return config


if __name__ == "__main__":
    parse()
