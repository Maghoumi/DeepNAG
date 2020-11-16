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

import numpy as np
import torch
from torch.utils.data import Sampler


# ----------------------------------------------------------------------------------------------------------------------
class MaybeRandomSampler(Sampler):
    """Similar to PyTorch's RandomSampler, with the ability to disable randomness"""

    def __init__(self, data_source, shuffle=True):
        """
        The constructor

        :param data_source: the data source
        :param shuffle: flag indicating whether shuffling should be enabled
        """
        super(MaybeRandomSampler, self).__init__(data_source)
        self.data_source = data_source
        self.shuffle = shuffle

    def __iter__(self):
        n = len(self)
        if self.shuffle:
            return iter(torch.randperm(n).tolist())
        else:
            return iter(range(n))

    def __len__(self):
        return len(self.data_source)


# ----------------------------------------------------------------------------------------------------------------------
class SampleCollator:
    """A collator for a list of samples. Used in data loading."""
    def __call__(self, batch):
        """
        Collates a batch of samples into PyTorch tensors suitable for training

        :param batch: a batch of data
        :return: a tuple of (x, y, uuid) of all the samples in the batch
        """
        return torch.stack([sample.x for sample in batch]), \
            torch.stack([sample.y for sample in batch]).squeeze(), \
            [sample.unique_id for sample in batch]


# ----------------------------------------------------------------------------------------------------------------------
class MinMaxNormalizer(object):
    """A MinMax normalizer. Uses the min and max feature values to scale all data points into the range [-1, 1]."""

    def __init__(self):
        """
        The constructor
        """
        print("Initializing a MinMaxNormalizer")
        self.min = None
        self.max = None
        self.factors_computed = False

    def __str__(self):
        return F"MinMax normalizer:\n\tMin: {self.min}\n\tMax:{self.max}"

    def compute_factors(self, train_set):
        """
        Compute the normalization factors from the training set

        :param train_set: the training set to use
        """
        self.factors_computed = True
        concat = np.vstack([sample.x for sample in train_set])
        self.min = concat.min(axis=0, keepdims=True)
        self.max = concat.max(axis=0, keepdims=True)

    def normalize_list(self, sample_list):
        """
        Normalize a list of samples (inplace)

        :param sample_list: the list of samples to normalize
        """
        print(F'Normalizing one list of samples with {len(sample_list)} samples...')

        for idx, sample in enumerate(sample_list):
            sample.x = self.normalize(sample.x)

            if len(sample_list) > 50000 and (idx + 1) % 100000 == 0:
                print(F"\t{idx}...")

    def normalize(self, sample):
        """
        Normalize a single sample

        :param sample: the sample to normalize
        :return: the normalized sample
        """
        result = 2 * (sample - self.min) / (self.max - self.min) - 1
        return result

    def unnormalize_list(self, samples):
        """
        Unnormalize a list of samples

        :param samples: the list of samples
        :return: an array of unnormalized samples
        """
        if isinstance(samples, torch.Tensor):
            samples = samples.detach().cpu().numpy()

        unnormalized = []

        for sample in samples:
            unnormalized.append(self.unnormalize(sample))

        return np.asarray(unnormalized)

    def unnormalize(self, sample):
        """
        Unnormalize a single sample

        :param sample: the sample to unnormalize
        :return: the unnormalized sample
        """
        return self.min + (sample + 1) * (self.max - self.min) / 2.0
