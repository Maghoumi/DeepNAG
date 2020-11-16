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
import numpy as np
try:
    from pytorch_softdtw_cuda.soft_dtw_cuda import SoftDTW
except ImportError as error:
        if 'softdtw' in error.name:
            print('--------------------------------------------------------------------------------------------')
            print("Could not import 'soft_dtw_cuda'. Did you forget to initialize this repository's submodules?")
            print("Run 'git submodule update --init --recursive' and try again!")
            print('--------------------------------------------------------------------------------------------')
        raise


# ----------------------------------------------------------------------------------------------------------------------
def masked_mean(tensor, mask, dim):
    """
    Finds the mean of the tensor along the specified dim, while taking into account the mask.
    See https://www.codefull.net/2020/03/masked-tensor-operations-in-pytorch for more details

    :param tensor: input tensor
    :param mask: the mask
    :param dim: the dimension to compute the mean along
    :return: the mean tensor
    """
    masked = torch.mul(tensor, mask)
    return masked.sum(dim=dim) / mask.sum(dim=dim)


# ----------------------------------------------------------------------------------------------------------------------
def masked_max(tensor, mask, dim):
    """
    Finds the max of the tensor along the specified dim, while taking into account the mask.
    See https://www.codefull.net/2020/03/masked-tensor-operations-in-pytorch for more details

    :param tensor: input tensor
    :param mask: the mask
    :param dim: the dimension to compute the max along
    :return: the max tensor
    """
    masked = torch.mul(tensor, mask)
    neg_inf = torch.zeros_like(tensor)
    neg_inf[~mask.byte()] = -np.inf
    return (masked + neg_inf).max(dim=dim)[0]


# ----------------------------------------------------------------------------------------------------------------------
def dist_func_cos(x, y):
    """
    Helper function to compute pair-wise cosine dissimilarity. This is meant to be used with sDTW.

    :param x: input tensor
    :param y: input tensor
    :return: output tensor, suitable for sDTW computation
    """
    # Convert to direction vectors
    x = x[:, 1:, :] - x[:, :-1, :]
    y = y[:, 1:, :] - y[:, :-1, :]
    n = x.size(1)
    m = y.size(1)
    d = x.size(2)
    x = x.unsqueeze(2).expand(-1, n, m, d)
    y = y.unsqueeze(1).expand(-1, n, m, d)

    # Convert to dissimilarity
    return 1.0 - torch.nn.CosineSimilarity(dim=3)(x, y)


# ----------------------------------------------------------------------------------------------------------------------
class HausdorffDistance(object):
    """
    An efficient vectorized implementation of the Hausdorff distance between two point sets A and B.
    This implementation is "class-aware": only the rows that have matching labels are accounted in the computation.
    """

    def __init__(self, metric, distance_type, num_classes, use_cuda, sdtw_bandwidth=None, sdtw_gamma=0.1):
        """
        Initializes a distance computer with the specified properties.

        :param metric: Options are 'sdtw-ed', 'stdw-cos', 'cos' and 'ed'
        :param distance_type: Options are regular Hausdorff distance ('hd'),
            modified Hausdorff distance ('mhd') and  average Hausdorff distance ('ahd')
        :param num_classes: The number of dataset's classes
        :param use_cuda: Flag indicating whether to use CUDA.
        :param sdtw_bandwidth: The Sakoe-Chiba bandwidth to use for sDTW (only used if 'distance_type' is a sDTW variant)
        :param sdtw_gamma: The gamma value to use for sDTW (only used if 'distance_type' is a sDTW variant)
        """
        self._metric = metric
        self._distance_type = distance_type
        self._num_classes = num_classes
        self._use_cuda = use_cuda
        self._sdtw_bandwidth = sdtw_bandwidth  # The Sakoe-Chiba band to speed up the processing of sDTW
        self._sdtw_gamma = sdtw_gamma

        self._metric_obj = None
        self._is_metric_symmetric = None  # Is true if the underlying metric has the symmetric property (i.e. d(a, b) == d (b, a) )
        self._initialize_metrics()

    def _initialize_metrics(self):
        """
        Initializes the metric object and properties using the input arguments
        """
        if self._metric == 'sdtw-ed':
            self._metric_obj = SoftDTW(use_cuda=self._use_cuda, gamma=self._sdtw_gamma, bandwidth=self._sdtw_bandwidth)
            self._is_metric_symmetric = True
        elif self._metric == 'sdtw-cos':
            self._metric_obj = SoftDTW(use_cuda=self._use_cuda, gamma=self._sdtw_gamma, bandwidth=self._sdtw_bandwidth, dist_func=dist_func_cos)
            self._is_metric_symmetric = True
        elif self._metric == 'cos':
            self._metric_obj = torch.nn.CosineSimilarity(dim=2)
            self._is_metric_symmetric = True
        elif self._metric == 'ed':
            self._metric_obj = torch.nn.PairwiseDistance()
            self._is_metric_symmetric = True
        else:
            raise Exception(F'HausdorffDistance: Unknown metric "{self._metric}"')

    def _pdist_func(self, x, y, mask):
        """
        Computes the distance between 'x' and 'y' based on how this object was initialized while accounting for a mask.

        :param x: input tensor
        :param y: input tensor
        :param mask: the mask tensor
        :return: the computed distance values put into the tensor according to 'mask'. Other values will be infinite
        """
        # Initialize the result to inf
        result = torch.ones(x.shape[0], device=x.device) * np.inf

        # Mask the inputs
        x = x[mask]
        y = y[mask]

        dist = None

        if 'sdtw' in self._metric:
            dist = self._metric_obj(x, y)
        elif self._metric == 'cos':
            a_first = x[:, 0:1, :]
            b_first = y[:, 0:1, :]
            a_vec = x[:, 1:, :] - x[:, :-1, :]
            b_vec = y[:, 1:, :] - y[:, :-1, :]
            a_vec = torch.cat([a_first, a_vec], dim=1)
            b_vec = torch.cat([b_first, b_vec], dim=1)
            cos_simil = self._metric_obj(a_vec, b_vec)
            cos_simil[torch.isnan(cos_simil)] = -1
            dist = (1 - cos_simil).sum(dim=1)
        elif self._metric == 'ed':
            dist = self._metric_obj(x.view(x.shape[0], -1), y.view(y.shape[0], -1))

        else:
            raise Exception(F'HausdorffDistance: Unknown metric "{self._metric}"')

        # Set the output given the mask
        result[mask] = dist
        return result

    def _compute(self, a, b, labels):
        """
        Computes the specified Hausdorff distance between 'a' and 'b' while ensuring that only rows with matching labels
        are used for the computation.

        :param a: a batch of input
        :param b: a batch of input
        :param labels: the labels corresponding to each element in the batch
        :return: the computed Hausdorff distance. The result is summed over the batch.
        """
        bs = labels.shape[0]

        cart_prod = torch.cartesian_prod(labels, labels)
        cond_pred = (cart_prod[:, 0] == cart_prod[:, 1])

        # Compute d(a, b) which is the distance between every point of a to B
        lhs = torch.repeat_interleave(a, bs, dim=0)
        rhs = b.repeat(bs, 1, 1)
        dists = self._pdist_func(lhs, rhs, cond_pred)
        # with dists.view(bs, -1), every row corresponds to one example in the batch, while every column is the distance
        # to that example, considering the label. If the labels don't match, value will be inf, if they match, then
        # the actual cost value will be there
        ab = dists.view(bs, -1).min(dim=1)[0]

        # Now compute d(b, a) which is the distance between every point of b to A
        # However...
        # If the underlying metric is symmetric, we can reuse what we just computed
        if self._is_metric_symmetric:
            ba = dists.view(bs, -1).min(dim=0)[0]
        else:
            lhs = torch.repeat_interleave(b, bs, dim=0)
            rhs = a.repeat(bs, 1, 1)
            ba = self._pdist_func(lhs, rhs, cond_pred)
            ba = ba.view(bs, -1).min(dim=1)[0]

        # After the above:
        #   ab (BSx1) is the minimum distance from a to b (considering the labels)
        #   ba (BSx1) is the minimum distance from b to a (considering the labels)
        # In other words, every row in ab is the distance of the closest element in b to that row
        # that has a matching class label

        # Do the computation, while ensuring "class awareness"
        result = None

        # Now find the correct distance, while taking the class label into account
        mask = cond_pred.view(bs, -1).float()  # Older versions of PyTorch will complain if this is bool
        # Make ab and ba the same shape as mask so that we can apply the mask
        ab = ab.view(bs, -1).expand_as(mask)
        ba = ba.view(bs, -1).expand_as(mask)
        # After this, ab[mask] would give you BSxBS matrix, every column is either 0 or some value.
        # The value in every row is: "what is the distance between me and the closest element in b that has the same label as me?"
        # So, if you take the mean() across each column, you'll have the "per-class average distance from a to b".

        if self._distance_type == 'hd':
            vals = torch.stack([masked_max(ab, mask, dim=0),
                                    masked_max(ba, mask, dim=0)])
            result, _ = vals.max(dim=0)
        elif self._distance_type == 'mhd':
            vals = torch.stack([masked_mean(ab, mask, dim=0),
                                masked_mean(ba, mask, dim=0)])
            result, _ = vals.max(dim=0)
        elif self._distance_type == 'ahd':
            vals = torch.stack([masked_mean(ab, mask, dim=0),
                                masked_mean(ba, mask, dim=0)])
            result = vals.mean(dim=0)
        else:
            raise Exception(F'HausdorffDistance: Unknown distance type "{self._distance_type}" (mode is class_aware)')

        return result.sum()

    def __call__(self, a, b, labels):
        """
        Computes the specified Hausdorff distance between 'a' and 'b' while ensuring that only rows with matching labels
        are used for the computation.

        :param a: a batch of input
        :param b: a batch of input
        :param labels: the labels corresponding to each element in the batch
        :return: the computed Hausdorff distance
        """
        return self._compute(a, b, labels)
