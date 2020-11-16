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
                                 help='Dataset to train on: ' + ' | '.join(Dataset.dataset_names))
        self.parser.add_argument('--seed', default=1234, type=int,
                                 help='Random number generator seed, pass nothing or -1 to use a random seed')

        # Training arguments
        self.parser.add_argument('--latent-dim', type=int, default=32,
                                 help="Latent space dimension")
        self.parser.add_argument('--resample-n', type=int, default=64,
                                 help="The number of equidistant points to spatially resample every sample to")
        self.parser.add_argument('--lr', type=float, default=1e-4,
                                 help="Adam's learning rate")
        self.parser.add_argument('--beta0', type=float, default=0.5,
                                 help="Adam's beta0 value")
        self.parser.add_argument('--beta1', type=float, default=0.9,
                                 help="Adam's beta1 value")
        self.parser.add_argument('--batch-size', type=int, default=64,
                                 help='Batch size')
        self.parser.add_argument('--epochs', type=int, default=25000,
                                 help='Number of epochs to run the training for')
        # Logging
        self.parser.add_argument('--use-tensorboard', type=int, default=0, choices=[0, 1],
                                 help='Determines whether to use tensorboard for logging')
        self.parser.add_argument('--vis-frequency', type=int, default=100,
                                 help='Determines after how many epochs to visualize the results (zero will disable visualization)')
        # Trained model evaluation
        self.parser.add_argument('--evaluate', type=str, default=None,
                                 help='Path to a saved checkpoint to evaluate')

        self.datapath = None
        self.dataset_name = None
        self.seed = None

        self.latent_dim = None
        self.resample_n = None
        self.lr = None
        self.beta0 = None
        self.beta1 = None
        self.batch_size = None
        self.epochs = None

        self.use_tensorboard = None
        self.vis_frequency = None

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

        self.latent_dim = args.latent_dim
        self.resample_n = args.resample_n
        self.lr = args.lr
        self.beta0 = args.beta0
        self.beta1 = args.beta1
        self.batch_size = args.batch_size
        self.epochs = args.epochs

        self.use_tensorboard = args.use_tensorboard
        self.vis_frequency = args.vis_frequency

        self.evaluate = args.evaluate

        # Take action with some of the parameters
        self.seed = hash(uuid.uuid4()) if (self.seed is None or self.seed == -1) else self.seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Run-specific names and unique strings
        self.make_run_dir()

    def make_run_dir(self):
        """
        Creates the run directories to dump the logs and checkpoints.
        """
        timestamp = time.strftime("%Y%m%d-%H%M%S")

        # Build the run system's name
        system_name = str(platform.node())

        self.run_name = F"{self.dataset_name}-{timestamp}-{system_name}"

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
