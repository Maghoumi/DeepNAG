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
from torch.utils.data import DataLoader

from dataloader.io.load_jk2017 import load_jk2017, BONES, JOINT_CNT
from dataloader.io.load_dollar_gds import load_dollar_gds
from dataloader.utils import MaybeRandomSampler, SampleCollator, MinMaxNormalizer
from visualizer import Visualizer2D, Visualizer3D


# ----------------------------------------------------------------------------------------------------------------------
class Dataset:
    """Defines the super class for all the datasets in the system"""

    dataset_names = [
        'jk2017-kinect',
        'dollar-gds',
    ]

    @staticmethod
    def instantiate(opt):
        """
        Instantiate a new dataset using the passed command-line arguments

        :param opt: the command line arguments to use for instantiation
        :return: a new Dataset instance
        """
        if opt.dataset_name not in Dataset.dataset_names:
            raise Exception(F'Unknown dataset "{opt.dataset_name}"')

        if opt.dataset_name == "jk2017-kinect":
            dataset = DatasetJK2017Kinect(opt)
        elif opt.dataset_name == "dollar-gds":
            dataset = DatasetDollarGDS(opt)
        else:
            raise Exception(F"Unknown dataset '{opt.dataset_name}'")

        print(F"Dataset '{opt.dataset_name}' instantiated")

        return dataset

    def __init__(self, opt):
        """
        The constructor

        :param opt: the command-line arguments
        """
        self.opt = opt

        self.num_classes = None
        self.num_features = None

        self.class_to_idx = {}
        self.idx_to_class = {}
        self.samples = []

        # Used for PyTorch DataLoader
        self._cache = None
        self._data_loaders = None

        self._visualizer = None

    def __str__(self):
        return F"Dataset: '{self.opt.dataset_name}'\n" \
            F"\tClasses: {self.idx_to_class}\n" \
            F"\tNum samples: {len(self.samples)}\n"

    @property
    def visualizer(self):
        return self._visualizer

    def get_split(self, normalizer=None):
        """
        Create a DataSplit instance from this dataset.

        :param normalizer: the data normalizer to use. If 'None', a MinMaxNormalizer will be created.
        :return: a DataSplit instance
        """
        return DataSplit(self, normalizer)

    def _fill(self, samples, dataset_classes=None):
        """
        Populates the dataset using the given samples.

        :param samples: the list of samples
        :param dataset_classes: optional dataset classes
        """
        self.samples = samples
        to_assign_idx = 0
        self.num_features = self.samples[0].x.shape[1]

        # Do we already know what the classes are?
        if dataset_classes is not None:
            for cls in dataset_classes:
                self.class_to_idx[cls] = to_assign_idx
                self.idx_to_class[to_assign_idx] = cls
                to_assign_idx += 1
        else:
            # Extract all the classes
            for sample in self.samples:
                # Add class label/idx
                if sample.label not in self.class_to_idx:
                    self.class_to_idx[sample.label] = to_assign_idx
                    self.idx_to_class[to_assign_idx] = sample.label
                    to_assign_idx += 1

        self.num_classes = len(self.class_to_idx.keys())

        # Go through everything again, and assign y
        for sample in self.samples:
            if dataset_classes is None:
                sample.y = self.class_to_idx[sample.label]

        # Print a summary about the dataset
        print(self)


# ----------------------------------------------------------------------------------------------------------------------
class DataSplit:
    """
    Represents one split of the data, optionally normalized.
    To keep things simple, we assume we just have training data to work with (no validation/test sets)
    """

    def __init__(self, dataset, normalizer=None):
        """
        The constructor.

        :param dataset: the dataset
        :param normalizer: the data normalizer to use. If 'None', then a MinMaxNormalizer instance will be created.
        """
        self.dataset = dataset
        self.opt = dataset.opt
        self._data_loaders = None
        self.samples = {
            'train': [],
        }

        for sample in self.dataset.samples:
            self.samples['train'].append(copy.deepcopy(sample))

        # Print a quick summary
        print(F"\tTraining samples {len(self.samples['train'])}")

        # Determine the normalization method
        if normalizer is None:
            self.normalizer = MinMaxNormalizer()
        else:
            self.normalizer = normalizer

    def has_set(self, which_list):
        """
        Determines whether this object has the specified sample list

        :param which_list: the name of the sample list to check
        """
        return which_list in self.samples

    def get_data_loader(self):
        """
        :return: a PyTorch dataloader based on this data split
        """
        return TorchDataLoader(self)


# ----------------------------------------------------------------------------------------------------------------------
class TorchDataLoader:
    """
    Represents a DataSplit but loaded for PyTorch
    """

    def __init__(self, data_split):
        """
        The constructor

        :param data_split: the data split to use
        """
        super(TorchDataLoader, self).__init__()
        self.data_split = data_split
        self.opt = data_split.opt
        self._samples = {}
        self._data_loaders = {}
        # Helper cache to keep the list of samples of a specific class
        self._samples_by_class_cache = {
            'train': {},
        }

        # Convert everything to torch tensors
        for which_list in ['train']:
            if data_split.has_set(which_list):
                self._samples[which_list] = self._process_samples(data_split, which_list)
                shuffle = which_list == 'train'
                self._data_loaders[which_list] = \
                    DataLoader(self._samples[which_list],
                               shuffle=False,
                               batch_size=data_split.dataset.opt.batch_size,
                               collate_fn=SampleCollator(),
                               pin_memory=True,
                               sampler=MaybeRandomSampler(self._samples[which_list], shuffle))

    def has_set(self, which_list):
        """
        Determines whether this object has the specified sample list

        :param which_list: the name of the sample list to check
        """
        return which_list in self._samples and len(self._samples[which_list]) > 0

    def __getitem__(self, which_list):
        return self._data_loaders[which_list]

    def get_samples_of_class(self, which_list, y):
        """
        Returns a list of the samples of the specified class label

        :param which_list: the list name to query
        :param y: the label to match
        :return: the list of samples whose label matches the specified label
        """
        if which_list not in self._samples:
            raise Exception(F"Invalid list '{which_list}' specified")

        # If we haven't found these before, go through all the samples whose labels match the given one
        # and cache the results
        if y not in self._samples_by_class_cache[which_list]:
            # Go through all the samples whose labels match the given one
            self._samples_by_class_cache[which_list][y] = \
                [sample for sample in self._samples[which_list] if sample.y == y]

        return self._samples_by_class_cache[which_list][y]

    def _process_samples(self, data_split, which_list):
        """
        Processes the samples of the given list for consumption by the dataloader.

        :param data_split: the data split to use
        :param which_list: the list to process
        :return: list of processed samples
        """
        samples = copy.deepcopy(data_split.samples[which_list])

        if which_list == 'train':
            if not data_split.normalizer.factors_computed:
                data_split.normalizer.compute_factors(samples)
        else:
            raise Exception('Only processing the training data is implemented')

        data_split.normalizer.normalize_list(samples)
        return [sample.to_torch(self.opt.resample_n) for sample in samples]


# ----------------------------------------------------------------------------------------------------------------------
class DatasetJK2017Kinect(Dataset):
    """
    Dataset of JK2017 (Kinect) examples.
    See https://github.com/ISUE/Jackknife for more details.
    """

    def __init__(self, opt):
        """
        The constructor

        :param opt: the command-line arguments
        """
        super(DatasetJK2017Kinect, self).__init__(opt)
        self._fill(load_jk2017(opt.get_path_from_root('jk2017')))
        self._visualizer = Visualizer3D(self.num_classes, opt.resample_n, opt.latent_dim, BONES, JOINT_CNT)


# ----------------------------------------------------------------------------------------------------------------------
class DatasetDollarGDS(Dataset):
    """
    $1-GDS dataset.
    See http://depts.washington.edu/acelab/proj/dollar/index.html for more details.
    """

    def __init__(self, opt):
        """
        The constructor

        :param opt: the command-line arguments
        """
        super(DatasetDollarGDS, self).__init__(opt)
        self._fill(load_dollar_gds(opt.get_path_from_root('dollar_gds')))
        self._visualizer = Visualizer2D(self.num_classes, opt.resample_n, opt.latent_dim)
