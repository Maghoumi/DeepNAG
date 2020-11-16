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
import torch.nn as nn

from dataloader.utils import SampleCollator
from models.model_base import ModelBase
from models.hausdorff_distance import HausdorffDistance


# ----------------------------------------------------------------------------------------------------------------------
class DeepNAG(ModelBase):
    """The DeepNAG model and the training logic"""

    def __init__(self, num_classes, num_features, opt, device, visualizer=None):
        """
        The constructor

        :param num_classes: the total number of gesture classes
        :param num_features: the dimensionality of each feature vector
        :param opt: an instance of Options
        :param device: the compute device to use
        :param visualizer: an optional visualizer to plot the training progress
        """
        super(DeepNAG, self).__init__(num_classes, num_features, opt, device, visualizer)

        # Instantiate the network and the optimizer
        self._generator = DeepNAGNet(self._opt.latent_dim,
                                     self._num_features,
                                     self._num_classes)
        self._optimizer = torch.optim.Adam(self._generator.parameters(),
                                           lr=self._opt.lr,
                                           betas=(self._opt.beta0, self._opt.beta1))

        # Loss-related objects
        self.loss_names += ['loss_ed', 'loss_cos', 'loss_resample']
        self.loss_ed = None  # The ED loss value
        self.loss_cos = None  # The COS loss value
        self.loss_resample = None  # The resample loss value

        self._criterion_l1 = nn.L1Loss()
        self._criterion_l2 = nn.MSELoss()
        self._criterion_ed = HausdorffDistance('sdtw-ed', 'ahd', self._num_classes, use_cuda=self._opt.use_cuda, sdtw_gamma=0.1)
        self._criterion_cos = HausdorffDistance('sdtw-cos', 'ahd', self._num_classes, use_cuda=self._opt.use_cuda, sdtw_gamma=0.1)

        self._to_device()

    def _to_device(self):
        """
        Transfer everything to the correct computation device
        """
        # Put everything in the correct device
        self._generator = torch.nn.DataParallel(self._generator).to(self._device)
        self._criterion_l1.to(self._device)
        self._criterion_l2.to(self._device)

    def _make_another_batch(self, which_list, labels, uuid):
        """
        Makes another batch containing examples labeled similarly to this batch,
        but the examples are guaranteed to be different.
        """

        labels = labels.flatten().tolist()
        picked_samples = []

        for label, path in zip(labels, uuid):
            same_class = [sample for sample in self._data_loader.get_samples_of_class(which_list, label) if
                          sample.unique_id != path and sample not in picked_samples]

            if len(same_class) == 0:
                # Means the batch has all the data available for this class. So need to do something ugly!!
                # I'll just reuse, their loss will become zero, so no harm done I guess.
                same_class = [sample for sample in self._data_loader.get_samples_of_class(which_list, label) if sample.unique_id != path]

            picked_samples += self._random.sample(same_class, 1)

        # Mimic a batch of data with the correct collation format
        result = SampleCollator()(picked_samples)[0].to(self._device, non_blocking=True)
        return result

    def _compute_losses(self, real_1, real_2, fake_1, fake_2_detached, labels):
        """
        Computes the losses using the given batches of data.

        :param real_1: a batch of real samples
        :param real_2: another batch of real samples
        :param fake_1: a batch of fake samples. The gradients are computed w.r.t this batch.
        :param fake_2_detached: another batch of fake samples. Gradients are not computed for this
        :param labels: the labels of each sample in the real data
        """
        #
        # Compute the ED loss
        #
        hd_real_fake = self._criterion_ed(real_1, fake_1, labels)
        hd_real_real = self._criterion_ed(real_1, real_2, labels)
        hd_fake_fake = self._criterion_ed(fake_1, fake_2_detached, labels)
        hd_global = self._criterion_l1(hd_fake_fake, hd_real_real)

        self.loss_ed = hd_real_fake + hd_global

        #
        # Compute the COS loss
        #
        hd_real_fake_cos = self._criterion_cos(real_1, fake_1, labels)
        hd_real_real_cos = self._criterion_cos(real_1, real_2, labels)
        hd_fake_fake_cos = self._criterion_cos(fake_1, fake_2_detached, labels)
        hd_global_cos = self._criterion_l1(hd_fake_fake_cos, hd_real_real_cos)

        self.loss_cos = hd_real_fake_cos + hd_global_cos

        #
        # Compute resample N loss to enforce equidistant points
        #
        between_point_dists = (fake_1[:, 1:, :] - fake_1[:, :-1, :]).norm(dim=2)
        path_len = between_point_dists.sum(dim=1, keepdim=True)
        target_between_point_dists = (path_len / self._opt.resample_n).expand_as(between_point_dists)
        self.loss_resample = 1000 * self._criterion_l2(between_point_dists, target_between_point_dists)

        self.loss = (self.loss_ed + self.loss_cos + self.loss_resample) / 3

    def _run_one_epoch(self, epoch):
        """
        Runs a single epoch of training.

        :param epoch: the epoch counter
        """
        for batch_idx, (examples, labels, paths) in enumerate(self._data_loader['train']):
            # Need two batches of real data. We create the second batch by picking other similarly labeled
            # samples in the training data
            real_2 = self._make_another_batch('train', labels, paths)
            real_1 = examples.to(self._device, non_blocking=True)
            labels = labels.to(self._device, non_blocking=True)

            # Generate two batches of fake data with these labels
            fake_1 = self.generate(labels)
            fake_2 = self.generate(labels).detach()  # Detach, because we don't want gradients to propagate to this batch

            self._compute_losses(real_1, real_2, fake_1, fake_2, labels)

            # Optimize
            self._optimizer.zero_grad()
            self.loss.backward()
            self._optimizer.step()

            self.bookkeep()


# ----------------------------------------------------------------------------------------------------------------------
class DeepNAGNet(nn.Module):
    def __init__(self, latent_dim, output_dim, num_classes):
        """
        Initializes a new DeepNAG model.

        :param latent_dim:  the latent space dimension
        :param output_dim: the dimension of the output tensors
        :param num_classes: the total number of sample classes
        """
        super(DeepNAGNet, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_classes = num_classes

        self.gru1 = nn.GRU(latent_dim + num_classes, 128, 1, batch_first=True)
        self.gru2 = nn.GRU(128, 256, 1, batch_first=True)
        self.gru3 = nn.GRU(256, 512, 1, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(512, output_dim),
            nn.Tanh(),
        )

    def forward(self, z, labels_one_hot):
        """
        The forward pass

        :param z: the latent vector (random normal noise)
        :param labels_one_hot: the one-hot labels to generate samples for
        :return: the generated samples
        """
        z_conditioned = torch.cat([z, labels_one_hot], dim=2)
        output, _ = self.gru1(z_conditioned)
        output, _ = self.gru2(output)
        output, _ = self.gru3(output)

        return self.fc(output)
