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
import unittest

from models.hausdorff_distance import HausdorffDistance


# ----------------------------------------------------------------------------------------------------------------------
class TestAverageHausdorffDistance(unittest.TestCase):
    """
    Unit tests for testing the vectorized implementation of "label-aware" average Hausdorff distance.
    """

    def test_ed(self):
        """
        Tests the vectorized implementation of "label-aware" average Hausdorff distance with Euclidean distance.
        """
        batch_size, seq_len, n_dims, n_classes = 100, 64, 32, 5
        a = torch.rand((batch_size, seq_len, n_dims), dtype=torch.float32)
        b = torch.rand((batch_size, seq_len, n_dims), dtype=torch.float32)
        labels = torch.randint(n_classes, (batch_size,), dtype=torch.int64)

        pdist_func = lambda x, y: torch.norm(x - y)
        agg_func = lambda x, y: (x + y) / 2
        # Compute the value via for loops
        truth = TestAverageHausdorffDistance._calc_average_hausdorff_distance(a, b, labels,
                                                                              pdist_func=pdist_func,
                                                                              hd_func=torch.mean,
                                                                              agg_func=agg_func)

        # Compute using the fast vectorized implementation
        hd = HausdorffDistance('ed', 'ahd', n_classes, use_cuda=False)
        obtained = hd(a, b, labels)

        # Compare the two values
        self.assertTrue(torch.isclose(truth, obtained),
                        F"Obtained and true values are different ({obtained.item()} vs {truth.item()})")

    def test_cos(self):
        """
        Tests the vectorized implementation of "label-aware" average Hausdorff distance with cosine dissimilarity distance.
        """

        batch_size, seq_len, n_dims, n_classes = 100, 64, 32, 5
        a = torch.rand((batch_size, seq_len, n_dims), dtype=torch.float32)
        b = torch.rand((batch_size, seq_len, n_dims), dtype=torch.float32)
        labels = torch.randint(n_classes, (batch_size,), dtype=torch.int64)

        def pdist_func(x, y):
            """
            Distance function to compute the cosine dissimilarity between tensors 'x' and 'y'

            :param x: input tensor
            :param y: input tensor
            :return: the cosine dissimilarity between the two input tensors
            """
            a_first = x[:, 0:1, :]
            b_first = y[:, 0:1, :]
            a_vec = x[:, 1:, :] - x[:, :-1, :]
            b_vec = y[:, 1:, :] - y[:, :-1, :]
            a_vec = torch.cat([a_first, a_vec], dim=1)
            b_vec = torch.cat([b_first, b_vec], dim=1)
            cos_simil = torch.nn.CosineSimilarity(dim=2)(a_vec, b_vec)
            cos_simil[torch.isnan(cos_simil)] = -1
            return (1 - cos_simil).sum(dim=1)

        agg_func = lambda x, y: (x + y) / 2
        # Compute the value via for loops
        truth = TestAverageHausdorffDistance._calc_average_hausdorff_distance(a, b, labels,
                                                                              pdist_func=pdist_func,
                                                                              hd_func=torch.mean,
                                                                              agg_func=agg_func)

        # Compute using the fast vectorized implementation
        hd = HausdorffDistance('cos', 'ahd', n_classes, use_cuda=False)
        obtained = hd(a, b, labels)

        # Compare the two values
        self.assertTrue(torch.isclose(truth, obtained),
                        F"Obtained and true values are different ({obtained.item()} vs {truth.item()})")

    @staticmethod
    def _calc_average_hausdorff_distance(a, b, labels, pdist_func, hd_func, agg_func):
        """
        Computes variants of Hausdorff distance between two point sets while accounting for the labels.
        This function is used to verify our vectorized implementation against.

        :param a: the first point set (batch_size x seq_len x n_dims)
        :param b: the second point set (batch_size x seq_len x n_dims)
        :param labels: the labels for every row in the batch
        :param pdist_func: the point distance function to use
        :param hd_func: the function to use for the directed Hausdorff distance
        :param agg_func: the aggregation function to use for the final distance value
        :return: the sum of the computed distances for the batch
        """

        a_to_b = []
        b_to_a = []
        matching_rows = []  # Stores the indices for those points that have the same labels across different rows in the batch
        bs = a.shape[0]

        # Compute point-wise distances, and store label information
        for i in range(bs):
            matching_indices = [j for j in range(bs) if labels[i] == labels[j]]
            matching_rows.append(matching_indices)

            # For every point in 'a', compute the minimum distance to every point in 'b' that has the same label
            a_to_b.append(min([
                pdist_func(a[i:i+1], b[j:j+1]) for j in matching_indices
            ]))
            # Do the same for every point in 'b'
            b_to_a.append(min([
                pdist_func(b[i:i+1], a[j:j+1]) for j in matching_indices
            ]))

        a_to_b = torch.FloatTensor(a_to_b)
        b_to_a = torch.FloatTensor(b_to_a)

        results = []

        # Aggregate the result for every row, while being "label-aware"
        for matches in matching_rows:
            ab = hd_func(a_to_b[matches])
            ba = hd_func(b_to_a[matches])

            results.append(agg_func(ab, ba))

        return torch.stack(results).sum()


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    unittest.main()
