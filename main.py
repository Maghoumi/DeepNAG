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

import torch
import sys

from options import Options
from dataloader.dataset import Dataset
from models.DeepNAG import DeepNAG


# ----------------------------------------------------------------------------------------------------------------------
class Logger(object):
    """Helper class for redirecting print calls to both the console and a file"""

    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, "w")

    def write(self, message):
        """
        Write the message
        :param message: the message to write
        """
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        """
        Flush the message into file
        """
        self.terminal.flush()
        self.log.flush()


# ----------------------------------------------------------------------------------------------------------------------
def train(dataset, device):
    """
    Trains a model using the given arguments.

    :param dataset: the dataset
    :param device: the computation device to use
    """
    data_split = dataset.get_split()

    model = DeepNAG(dataset.num_classes, dataset.num_features, dataset.opt, device, dataset.visualizer)
    model.run_training_loop(data_split)
    model.save()


# ----------------------------------------------------------------------------------------------------------------------
def evaluate(dataset, device):
    """
    Visually evaluate a trained model.

    :param dataset: the dataset
    :param device: the computation device to use
    """
    visualizer = dataset.visualizer
    data_split = dataset.get_split()

    model = DeepNAG(dataset.num_classes, dataset.num_features, dataset.opt, device, visualizer)
    model.load(opt.evaluate)

    visualizer.visualize(model, data_split.get_data_loader())
    visualizer.show()


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Ensure correct python version
    if sys.version_info[0] < 3:
        raise Exception("Python 3 or a more recent version is required.")

    opt = Options()
    opt.parse()

    # Setup output redirection
    sys.stdout = Logger(opt.run_log_file, sys.stdout)
    sys.stderr = Logger(opt.run_err_file, sys.stderr)
    print(F"Run directory: {opt.run_dir}")

    # Instantiate the dataset
    dataset = Dataset.instantiate(opt)

    # Determine the computation device to use
    if opt.use_cuda:
        print("Using CUDA")
        device = torch.device('cuda:0')
    else:
        print("Using CPU")
        device = torch.device('cpu')

    if opt.evaluate is None:
        train(dataset, device)
    else:
        evaluate(dataset, device)
