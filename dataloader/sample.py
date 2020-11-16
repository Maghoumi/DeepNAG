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
import uuid
import numpy as np
from numpy.linalg import norm
import torch


# ----------------------------------------------------------------------------------------------------------------------
class Sample:
    """Defines a single data sample used throughout the system"""

    def __init__(self, x, label, unique_id=None):
        """
        The constructor

        :param x: the sample feature arrau
        :param label: the label of this sample
        :param unique_id: a unique ID for this sample to distinguish it uniquely from other samples
        """
        self.x = Sample._fix_type(x)
        self.y = None  # This will become the label index (set later by the dataset)
        self.label = label  # The label for the entire example
        self.unique_id = unique_id if unique_id is not None else uuid.uuid4()

    def __str__(self):
        return F"Sample:     Label={self.label: <3}   Len={self.x.shape[0]: <4}   ID={self.unique_id}"

    def to_torch(self, resample_n):
        """
        Resamples this sample, and converts to PyTorch tensors.

        :param resample_n: how many points to resample this sample to
        :return: a PyTorch tensor corresponding to this sample
        """
        result = copy.deepcopy(self)
        result.x = torch.from_numpy(self.resample(self.x, resample_n))
        result.y = torch.ones((1,), dtype=torch.int64) * self.y
        return result

    @staticmethod
    def _fix_type(arr):
        result = arr

        if result is None:
            return None

        if not isinstance(arr, np.ndarray):
            result = np.asarray(arr)

        if result.dtype == np.float64:
            result = result.astype(np.float32)

        return result

    def resample(self, np_pts, n):
        """
        Resamples to n equidistant points.

        Based on "Gestures without Libraries, Toolkits or Training: A $1 Recognizer for User Interface Prototypes" by Wobbrock et al.
        :param np_pts: numpy matrix of points. Each row is one timestep, columns are features
        :param n: how many points to resample to
        :return: the matrix array of points
        """

        def lerp(this, t, other):
            """
            Linearly interpolate between 'this' point and 'other' point,
            where t=0 is this point and t=1 is the other point.
            """
            return this + t * (other - this)

        # Get the length of the series
        series_len = norm(np_pts[1:, :] - np_pts[:-1, :], axis=1).sum()
        I = series_len / (n - 1)
        points = list(np_pts)
        new_points = [points[0]]
        D = 0.0
        ii = 1

        while ii < len(points):
            d = norm(points[ii - 1] - points[ii])
            if D + d >= I:
                q = lerp(points[ii - 1],
                         (I - D) / d,
                         points[ii])
                new_points += [q]
                points.insert(ii, q)
                D = 0.0
            else:
                D += d
            ii += 1

        while len(new_points) < n:
            new_points += [points[-1]]

        return np.asarray(new_points, dtype=np.float32)
