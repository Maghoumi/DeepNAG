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

import copy
import numpy as np
import os
from random import Random
import torch
from torch.nn.functional import one_hot
import time
from datetime import timedelta


# ----------------------------------------------------------------------------------------------------------------------
class ModelBase(object):
    """The base model interface and the training logic."""

    def __init__(self, num_classes, num_features, opt, device='cpu', visualizer=None):
        """
        The constructor

        :param num_classes: the total number of gesture classes
        :param num_features: the dimensionality of each feature vector
        :param opt: an instance of Options
        :param device: the compute device to use
        :param visualizer: an optional visualizer to plot the training progress
        """
        self._num_classes = num_classes
        self._num_features = num_features
        self._opt = opt
        self._visualizer = visualizer
        self._device = device

        self._random = Random(self._opt.seed)
        # The data loader instance (used for the training loop)
        self._data_loader = None
        # The data normalizer (scalar) that's used to transform the data from the range [-1, 1] to
        # the original feature scale.
        self._normalizer = None

        # The generator network and the optimizer
        self._generator = None
        self._optimizer = None

        # The name of the metrics to keep track of
        self.metric_names = []

        # Random noise for latent space representation
        self._latent = torch.FloatTensor(self._opt.batch_size,
                                         self._opt.resample_n,
                                         self._opt.latent_dim).to(self._device)

        self._stats = {}
        self._best_model = None
        self._best_model_metric = np.inf
        self._best_model_which_metric = None  # What key in self.stat to use to save the best model?

        # Create tensorboard writer
        if opt.use_tensorboard > 0:
            # Lazy-load tensorboard stuff
            from torch.utils.tensorboard import SummaryWriter
            self._tb_writer = SummaryWriter(self._opt.run_tb_dir, filename_suffix='.tfevents')
            print(F"Tensorboard logs will be dumped in '{self._tb_writer.get_logdir()}'")
        else:
            self._tb_writer = None

    #
    # Public properties and function
    #

    @property
    def device(self):
        """Returns the device that this network is on."""
        return self._device

    @property
    def normalizer(self):
        """Returns the normalizer of this model."""
        return self._normalizer

    def generate(self, labels, latent_vector=None, unnormalize=False):
        """
        Generates a batch of fake samples of the given labels

        :param labels: the labels to generate the data for
        :param latent_vector: the latent vector to use for generation. If nothing is provided, a
            new latent vector is generated.
        :param unnormalize: whether the returned samples should be "unnormalized", i.e.
            transformed back to the original feature space scale.
        :return: a batch of generated data
        """
        labels_one_hot = self._to_one_hot(labels)
        curr_batch_size = labels_one_hot.shape[0]

        # Generate or reuse the latent vector
        if latent_vector is None:
            self._generate_new_latent(curr_batch_size)
            latent_vector = self._latent
        elif latent_vector.shape[0] != curr_batch_size:  # Sanity check
            raise ValueError("The batch size of the provided latent vector does not match that of the labels vector.")

        result = self._generator(latent_vector, labels_one_hot)

        if unnormalize:
            result = self._normalizer.unnormalize_list(result)

        return result

    def run_training_loop(self, data_split):
        """
        Runs the training loop on the given data loader

        :param data_split: the split of data to train on
        """
        self._data_loader = data_split.get_data_loader()

        if self._normalizer is None:
            self._normalizer = data_split.normalizer

        print('Beginning training')

        for epoch in range(self._opt.epochs):
            self.begin_epoch()
            self._run_one_epoch(epoch)
            self.end_epoch()

            # Do some logging
            self._log_to_tensorboard('train', epoch)
            stats = self.get_stats()
            print(F"Epoch {epoch} \t\t(took {stats['deltatime']})\t"
                  F"{self._best_model_which_metric}={stats[self._best_model_which_metric]}")

            # Save a checkpoint (if enabled)
            if self._opt.checkpoint_frequency > 0 and \
               epoch > 0 and epoch % self._opt.checkpoint_frequency == 0:
                self.save(save_best=False, suffix=str(epoch))

        print("Training finished!")
        # Save the current state into a checkpoint
        self.save(save_best=False, suffix=str(self._opt.epochs))

    def bookkeep(self):
        """
        Bookkeep the stats.
        """
        self._stats['cnt'] += 1

        # Process all metrics
        for metric_name in self.metric_names:
            if hasattr(self, metric_name):
                attr = getattr(self, metric_name)

                if metric_name not in self._stats:
                    self._stats[metric_name] = []

                self._stats[metric_name].append(attr.item() if attr is not None else 0)
            else:
                raise Exception(F"The metric with the name {metric_name} was not found")

    def begin_epoch(self):
        """
        Mark the start of a training epoch.
        """
        self._stats = {'cnt': 0, 'start_time': time.time()}

        # Reset all metrics
        for metric_name in self.metric_names:
            if hasattr(self, metric_name):
                setattr(self, metric_name, None)

    def end_epoch(self):
        """
        Mark the end of a training epoch.
        """
        self._stats['stop_time'] = time.time()
        best_metric_candidate = self.get_stats()[self._best_model_which_metric]

        if best_metric_candidate < self._best_model_metric:
            self._best_model = self._get_state()
            self._best_model_metric = best_metric_candidate

    def get_stats(self):
        """
        :return: the stats collected during training
        """
        stats = copy.deepcopy(self._stats)

        for key in stats.keys():
            if key == 'cnt' or 'time' in key:
                continue

            stats[key] = np.mean(stats[key])

        # Calculate delta epoch time
        stats['deltatime'] = str(timedelta(seconds=stats['stop_time'] - stats['start_time']))
        del stats['start_time']
        del stats['stop_time']

        return stats

    def stat_str(self):
        """
        :return: the string representation of the collected training stats.
        """
        stat = self.get_stats()
        result = F"{{'best_{self._best_model_which_metric}': {self._best_model_metric}}} {str(stat)}"
        return result

    def save(self, save_best=True, suffix=None):
        """
        Saves the model into a checkpoint file.

        :param save_best: if `True`, will save the model yielding the best metric.
        :param suffix: an optional suffix to use for the saved filename.
        """
        # Decide on the model and the filename
        if save_best:
            model = self._best_model
            fname = 'checkpoint-best.tar'
        else:
            model = self._get_state()
            fname = F'checkpoint-{suffix}.tar' if suffix is not None else 'checkpoint.tar'

        path = os.path.join(self._opt.run_checkpoint_dir, fname)
        print(F"Saving the {'best ' if save_best else ''}checkpoint in '{path}'")
        torch.save(model, path)

    def load(self, path):
        """
        Loads the model stored in a checkpoint file.

        :param path: the path of the checkpoint file.
        """
        print(F"Loading the checkpoint from '{path}'...")
        loaded = torch.load(path)
        self._load_state(loaded)

    #
    # Private function
    #

    def _to_device(self):
        """
        Transfer everything to the correct computation device
        """
        raise NotImplementedError()

    def _to_one_hot(self, labels):
        """
        Converts a list of labels to their one-hot representation with correct format for label-conditioning

        :param labels: input labels
        :return: the one-hot representation of the input labels
        """
        one_hot_converted = one_hot(labels, self._num_classes).to(self._device, dtype=torch.float32, non_blocking=True)
        return one_hot_converted.unsqueeze(1).expand(-1, self._opt.resample_n, -1)

    def _generate_new_latent(self, batch_size):
        """
        Generates a new latent vector and stores internally.

        :param batch_size: the batch size of the generated latent
        """
        self._latent.resize_(batch_size,
                             self._opt.resample_n,
                             self._opt.latent_dim).normal_(0, 1)
        # Some PyTorch versions have a bug that results in the generation of NaN values after the above
        # operation. Here, we just replace those NaN values with zeros.
        self._latent[torch.isnan(self._latent)] = 0

    def _log_to_tensorboard(self, suffix, epoch):
        """
        Logs the current training progress to tensorboard (if enabled).

        :param suffix: the suffix to use for logging
        :param epoch:  the epoch counter
        """
        if not self._opt.use_tensorboard:
            return

        stat = self.get_stats()

        # Log the overall best metric
        self._tb_writer.add_scalar(F'best_{self._best_model_which_metric}_{suffix}',
                                   self._best_model_metric,
                                   epoch)
        # Log the metric that we track and save as the best metric
        self._tb_writer.add_scalar(F'{self._best_model_which_metric}_{suffix}',
                                   stat[self._best_model_which_metric],
                                   epoch)

        # Log the metric values
        self._tb_writer.add_scalar(F'{self._best_model_which_metric}_{suffix}',
                                   stat[self._best_model_which_metric],
                                   epoch)

        for metric_name in self.metric_names:
            if metric_name in stat:
                self._tb_writer.add_scalar(F'{metric_name}_{suffix}',
                                           stat[metric_name],
                                           epoch)

        # Should we visualize?
        if self._opt.vis_frequency > 0 and \
                epoch % self._opt.vis_frequency == 0 and \
                self._visualizer is not None and \
                self._num_features == 2:  # Only visualize if the features are 2D
            # Visualize the constant latent
            self._visualizer.visualize(self, self._data_loader)
            self._tb_writer.add_figure('const_latent', self._visualizer.fig, epoch)

    def _run_one_epoch(self, epoch):
        """
        Runs a single epoch (or step) of training.

        :param epoch: the epoch counter
        """
        raise NotImplementedError()

    def _get_state(self):
        """
        Helper function to get the internal state of the model.

        :return: a dictionary containing the internal state of the model.
        """
        raise NotImplementedError()

    def _load_state(self, state_dict):
        """
        Helper function to load the internal state of a model from a dictionary.

        :param state_dict: the dictionary containing the state to load.
        """
        raise NotImplementedError()

# ----------------------------------------------------------------------------------------------------------------------
