# MIT License
#
# Copyright (c) 2020 Mehran Maghoumi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ----------------------------------------------------------------------------------------------------------------------

import argparse
import numpy as np
import os
import platform
import time
import torch
import uuid

from dataloader.dataset import Dataset


# ----------------------------------------------------------------------------------------------------------------------
class Options:
    """This class defines options passed via command-line arguments."""

    def __init__(self):
        """
        The constructor
        """
        self.parser = argparse.ArgumentParser(description='DeepNAG Training')

        self.parser.add_argument('--datapath', default='data',
                                 help='Path where the datasets are located')
        self.parser.add_argument('--dataset-name', choices=Dataset.dataset_names, default='dollar-gds',
                                 help=F"Dataset to train on: {' | '.join(Dataset.dataset_names)}")
        self.parser.add_argument('--seed', default=1234, type=int,
                                 help='Random number generator seed, pass nothing or -1 to use a random seed')

        # Training arguments
        self.parser.add_argument('--model', type=str, default='DeepNAG', choices=['DeepNAG', 'DeepGAN'],
                                 help="The network model to train: ('DeepNAG' | 'DeepGAN')")
        self.parser.add_argument('--latent-dim', type=int, default=32,
                                 help='Latent space dimension')
        self.parser.add_argument('--resample-n', type=int, default=64,
                                 help='The number of equidistant points to spatially resample every sample to')
        self.parser.add_argument('--lr', type=float, default=1e-4,
                                 help="Adam's learning rate")
        self.parser.add_argument('--beta0', type=float, default=0.5,
                                 help="Adam's beta0 value")
        self.parser.add_argument('--beta1', type=float, default=0.9,
                                 help="Adam's beta1 value")
        self.parser.add_argument('--batch-size', type=int, default=64,
                                 help='Batch size')
        self.parser.add_argument('--epochs', type=int, default=None,
                                 help='Number of epochs to run the training for. Default optimals will be used if not provided.')
        # DeepGAN-specific parameters
        self.parser.add_argument('--deepgan-critic-iters', type=int, default=5,
                                 help="Number of steps to train DeepGAN's critic per every training step of the generator.")
        self.parser.add_argument('--deepgan-lambda', type=float, default=10,
                                 help="WGAN-GP's loss regularizer.")

        # Logging
        self.parser.add_argument('--use-tensorboard', type=int, default=0, choices=[0, 1],
                                 help='Determines whether to use tensorboard for logging')
        self.parser.add_argument('--vis-frequency', type=int, default=100,
                                 help='Determines after how many epochs to visualize the results (zero will disable visualization)')
        self.parser.add_argument('--checkpoint-frequency', type=int, default=10000,
                                 help='Determines after how many epochs to save a checkpoint of the trained model (zero will disable frequent checkpointing)')
        self.parser.add_argument('--experiment-name', type=str, default=None,
                                 help="Optional name for the experiment. This name will be used in the log directory's name")
        # Trained model evaluation
        self.parser.add_argument('--evaluate', type=str, default=None,
                                 help='Path to a saved checkpoint to evaluate')

        self.datapath = None
        self.dataset_name = None
        self.seed = None

        self.model = None
        self.latent_dim = None
        self.resample_n = None
        self.lr = None
        self.beta0 = None
        self.beta1 = None
        self.batch_size = None
        self.epochs = None
        self.deepgan_critic_iters = None
        self.deepgan_lambda = None

        self.use_tensorboard = None
        self.vis_frequency = None
        self.checkpoint_frequency = None
        self.experiment_name = None

        self.evaluate = None

        # Run-specific parameters
        self.run_name = None
        self.run_dir = None
        self.run_tb_dir = None  # The tensorboard directory of this run
        self.run_checkpoint_dir = None  # The checkpoint directory of this run
        self.run_log_file = None  # The stdout dump file of this run
        self.run_err_file = None  # The stderr dump file of this run

        self.use_cuda = torch.cuda.is_available()

    def parse(self):
        """
        Parse the command-line arguments
        """
        args = self.parser.parse_args()
        self.datapath = args.datapath
        self.dataset_name = args.dataset_name
        self.seed = args.seed

        self.model = args.model
        self.latent_dim = args.latent_dim
        self.resample_n = args.resample_n
        self.lr = args.lr
        self.beta0 = args.beta0
        self.beta1 = args.beta1
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.deepgan_critic_iters = args.deepgan_critic_iters
        self.deepgan_lambda = args.deepgan_lambda

        self.use_tensorboard = args.use_tensorboard
        self.vis_frequency = args.vis_frequency
        self.checkpoint_frequency = args.checkpoint_frequency

        if args.experiment_name is not None:
            self.experiment_name = args.experiment_name.replace(' ', '-')

        self.evaluate = args.evaluate

        # Take action with some of the parameters
        self.seed = hash(uuid.uuid4()) if (self.seed is None or self.seed == -1) else self.seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # If the number of epochs is not set, use some optimal defaults (only if training).
        if self.epochs is None and self.evaluate is None:
            self.epochs = 25000 if self.model == 'DeepNAG' else 150000
            print(F'The number of training epochs was automatically set to {self.epochs}')

        if self.deepgan_critic_iters < 1:
            raise Exception(F"The value '{self.deepgan_critic_iters}' is invalid for --deepgan-critic-iters")

        # Run-specific names and unique strings
        self.make_run_dir()

    def make_run_dir(self):
        """
        Creates the run directories to dump the logs and checkpoints.
        """
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        # Build the run system's name
        system_name = str(platform.node())

        self.run_name = F"{self.dataset_name}-{self.model}-{timestamp}-{system_name}"

        # Append the optional experiment name
        if self.experiment_name is not None:
            self.run_name += F"-{self.experiment_name}"

        # Make all the run directories
        if not os.path.isdir('logs'):
            os.mkdir('logs')

        self.run_dir = os.path.join('logs', self.run_name)
        os.mkdir(self.run_dir)
        self.run_tb_dir = os.path.join(self.run_dir, 'tensorboard')
        os.mkdir(self.run_tb_dir)
        self.run_checkpoint_dir = os.path.join(self.run_dir, 'checkpoints')
        os.mkdir(self.run_checkpoint_dir)
        os.mkdir(os.path.join(self.run_dir, 'logs'))
        self.run_log_file = os.path.join(self.run_dir, 'logs', 'log.txt')
        self.run_err_file = os.path.join(self.run_dir, 'logs', 'err.txt')

    def get_path_from_root(self, path):
        """
        :return: the specified path, prepended by the data path root.
        """
        return os.path.join(self.datapath, path)
