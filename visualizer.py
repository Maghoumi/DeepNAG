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

import colorsys
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from random import Random
import torch

# Global variable for the animation object. If not used, the animation may not be displayed
# (see https://stackoverflow.com/questions/41625518/matplotlib-funcanimation-isnt-calling-the-passed-function)
ani = None


# ----------------------------------------------------------------------------------------------------------------------
class Visualizer2D(object):
    """Helper class to draw 2D time-series data."""

    def __init__(self, num_classes, resample_n, latent_dim):
        """
        The constructor

        :param num_classes: the number of gesture classes
        :param resample_n: resampling N count
        :param latent_dim: latent space dimensions
        """
        self._num_const_samples_per_class = 16  # How many rows of samples to generate?
        self._rows = self._num_const_samples_per_class + 3  # One additional row for real samples, two additional rows for overlapped versions.
        self._cols = num_classes
        self._idx = None
        self._fig = None
        self._ax = None
        self._random = Random(0)

        # Create what's needed for fake sample generation
        self._const_latent = torch.randn((self._num_const_samples_per_class * num_classes,
                                          resample_n,
                                          latent_dim), dtype=torch.float32)
        # Create the corresponding label vectors
        self._const_latent_labels = torch.LongTensor([c for c in range(num_classes)]).repeat(self._num_const_samples_per_class)
        self.reset()

    @property
    def fig(self):
        self._adjust()
        return self._fig

    def show(self):
        """Show the figure."""
        self._adjust()
        plt.show()

    def reset(self):
        """Reset internal state."""
        self._idx = 0

        if self._fig is not None:
            plt.close(self._fig)

        self._fig, self._ax = plt.subplots(self._rows, self._cols, figsize=(18, 16))
        self._ax = self._ax.flatten()  # For flat indexing

    def visualize(self, model, data_loader):
        """
        Visualize some synthetic samples

        :param model: the model to use
        :param data_loader: the dataloader instance to use
        """

        with torch.no_grad():
            # Transfer to the same device as the model
            self._const_latent = self._const_latent.to(model.device)
            # Generate fake samples
            fakes = model.generate(self._const_latent_labels, self._const_latent, True)
            self.reset()

            real_by_class = {}

            for cls_idx in range(self._cols):
                real_by_class[cls_idx] = [sample.x.cpu() for sample in data_loader.get_samples_of_class('train', cls_idx)]

            # Render the rows of real examples
            for cls_idx in range(self._cols):
                real = self._random.sample(real_by_class[cls_idx], 1)[0]
                self._add_sample(real, label=str(cls_idx), color='red')

            # Render all the fake samples
            for fake in fakes:
                self._add_sample(fake, color='black')

            # Render an overlay of all fake samples
            for cnt, fake in enumerate(fakes):
                if cnt > 0 and cnt % self._cols == 0:
                    self._same_row()
                self._add_sample(fake, color='black', alpha=0.2)

            # Render an overlay of random real samples
            for cnt in range(len(fakes)):
                if cnt > 0 and cnt % self._cols == 0:
                    self._same_row()
                cls_idx = cnt % self._cols
                real = self._random.sample(real_by_class[cls_idx], 1)[0]
                self._add_sample(real, color='red', alpha=0.2)

    def _add_sample(self, pts, label=None, color="black", alpha=1.0):
        """
        Add a sample to the plot.

        :param pts: point vector
        :param label: optional label
        :param color: the drawing color
        :param alpha: the drawing alpha value
        """
        if isinstance(pts, torch.Tensor):
            pts = pts.numpy()

        ax = self._ax[self._idx]
        ax.axis('off')

        if label is not None:
            ax.set_title(label)

        x = pts[:, 0]
        y = -pts[:, 1]
        ax.plot(x, y, color, linewidth=1.7, alpha=alpha)
        # marker at start of stroke
        ax.plot([x[0]], [y[0]], color=color, marker='o', markersize=3, alpha=alpha)

        # increment for next position
        self._idx += 1

    def _same_row(self):
        """Resets the position counter to the beginning of the current row"""
        self._idx = ((self._idx - self._cols) // self._cols) * self._cols

    def _adjust(self):
        """Adjust the figure layout."""
        plt.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98, wspace=0.86, hspace=0.86)


# ----------------------------------------------------------------------------------------------------------------------
class Visualizer3D(object):
    """Helper class to draw 3D skeleton gestures."""

    def __init__(self, num_classes, resample_n, latent_dim, bones, num_joints):
        """
        The constructor

        :param num_classes: the number of gesture classes
        :param resample_n: resampling N count
        :param latent_dim: latent space dimensions
        :param bones: the structure describing the bones/joint connections
        :param num_joints: the number of joints in the skeleton
        """
        self._num_const_samples_per_class = 5
        self._num_classes = num_classes
        self._resample_n = resample_n
        self._latent_dim = latent_dim
        self._bones = bones
        self._num_joints = num_joints

        self._fig = None
        self._samples = []  # The list of samples to visualize
        self._random = Random(0)

        self.reset()

    @property
    def fig(self):
        return self._fig

    def show(self):
        """Show the figure."""
        plt.show()

    def reset(self):
        """Reset internal state."""
        self._samples = []

        if self._fig is not None:
            plt.close(self._fig)

        self._fig = None

    def visualize(self, model, data_loader):
        """
        Visualize some synthetic samples

        :param model: the model to use
        :param data_loader: the dataloader instance to use
        """

        with torch.no_grad():
            for cls_idx in range(self._num_classes):
                self.reset()
                # Get all the real samples of this class
                reals_this_class = [sample.x.cpu().numpy() for sample in data_loader.get_samples_of_class('train', cls_idx)]
                real = model.normalizer.unnormalize_list(self._random.sample(reals_this_class, 1))[0]
                # Add this sample
                self._samples.append(real)
                real_mask = [True]

                # Generate a number of fake samples
                latent = torch.randn((self._num_const_samples_per_class, self._resample_n, self._latent_dim),
                                     dtype=torch.float32, device=model.device)
                labels = torch.LongTensor([cls_idx] * self._num_const_samples_per_class)
                fakes = model.generate(labels, latent, True)

                for i in range(self._num_const_samples_per_class):
                    self._samples.append(fakes[i])
                    real_mask.append(False)

                self.draw_skeleton(annotation=data_loader.data_split.dataset.idx_to_class[cls_idx],
                                   real_mask=real_mask)
                self.show()

    @staticmethod
    def get_unique_colors(num_colors):
        """
        Gets n distinct colors. Taken from https://stackoverflow.com/a/9701141/398316
        Returns a N x 3 array
        """
        colors = []
        for i in np.arange(0., 360., 360. / num_colors):
            hue = i / 360.
            lightness = (50 + np.random.rand() * 10) / 100.
            saturation = (90 + np.random.rand() * 10) / 100.
            colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
        return np.asarray(colors)

    def draw_skeleton(self, annotation='', real_mask=None, body_count=1):
        """
        Draws the skeletons with animation

        :param annotation: optional annotation to show
        :param real_mask: bool mask specifying which samples are real. Real samples are rendered in black
        :param body_count: the number of skeleton bodies each row of samples contain.
        """
        n_samples = len(self._samples)
        num_frames = len(self._samples[0])

        # Subtract the mean value of one gesture from all
        reshaped = self._samples[0].reshape(-1, 3)
        mean = reshaped.mean(axis=0, keepdims=True)
        centered_samples = []

        for sample in self._samples:
            reshaped = sample.reshape(-1, 3)
            reshaped -= mean
            centered_samples.append(reshaped.reshape(sample.shape))

        # Generate colors according to the mask (if provided)
        if real_mask is not None and n_samples > 1:
            # Generate unique colors according to how many fake examples we have
            n_real = sum([1 for flag in real_mask if flag])
            n_unique_colors = n_samples - n_real
            unique_colors = list(Visualizer3D.get_unique_colors(n_unique_colors))
            unique_colors += [np.zeros(3, )] * n_real
        else:
            # Generate more colors if we're drawing more than 1 sample
            unique_colors = []
            if n_samples > 1:
                unique_colors += list(Visualizer3D.get_unique_colors(n_samples - 1))
            unique_colors += [np.zeros(3, )]  # Add the black color to the end

        unique_colors = np.asarray(unique_colors)

        fig, ax = plt.subplots(sharex=True, sharey=True, figsize=(15, 12))
        ax.set_xlim((-1.0, 1.0))
        ax.set_ylim((-1.1, 1))
        fig.tight_layout()
        plt.axis('off')
        plt.text(0, -1.05, annotation, horizontalalignment='center', fontsize=20, fontweight='bold')

        lines = []  # Each "line" is a nested list of "line" objects
        scats = []

        for i in range(n_samples):
            body_lines = []
            body_scats = []
            for body in range(body_count):
                line, = ax.plot([], [], "k", linewidth=2.8)
                scat = ax.scatter([], [], s=70, c=[])
                body_lines += [line]
                body_scats += [scat]
            lines += [body_lines]
            scats += [body_scats]

        if annotation is not None:
            fig.canvas.set_window_title(annotation)

        def update_plot(frame_num, lines, scats):
            """
            The update call on rendering frames

            :param frame_num: frame number
            :param lines: line objects
            :param scats: scatterplot objects
            :return: the objects that changed due to frame update
            """
            x = []
            y = []

            # Go through the list from last to first, so that the first one is rendered on top (in black)
            for idx, sample in enumerate(reversed(centered_samples)):
                row = sample[frame_num]  # The data of the current frame

                # Every row has the points of all bodies in the current frame
                # Need to extract the points of each body, add and draw
                for body in range(body_count):
                    line = lines[idx][body]  # Get the "line" that we'll use for rendering this body
                    scat = scats[idx][body]  # Get the "scat" that we'll use for rendering this body
                    frame = row[body * self._num_joints * 3: (body + 1) * self._num_joints * 3]

                    # Now reshape to rows of xyz
                    frame = frame.reshape(-1, 3)

                    for start, end in self._bones:
                        xx = [frame[start][0], frame[end][0], np.nan]  # NaN is to create discontinuity in the lines
                        yy = [frame[start][1], frame[end][1], np.nan]
                        x += [xx]
                        y += [yy]

                    line.set_data(x, y)

                    # Adjust the transparency value
                    color = np.asarray(list(unique_colors[idx]) + [1.0 if idx == len(centered_samples) - 1 else 0.35])
                    line.set_color(color)

                    # Also render the joints
                    x_scat = [f[0] for f in frame]
                    y_scat = [f[1] for f in frame]
                    scat.set_offsets(np.c_[x_scat, y_scat])
                    scat.set_color(color)

                    x = []
                    y = []

            modified = [item for sublist in lines for item in sublist] + [item for sublist in scats for item in sublist]
            return tuple(modified)

        # Create the animation
        global ani
        ani = animation.FuncAnimation(fig, update_plot, frames=range(int(num_frames)),
                                      fargs=(lines, scats,), interval=20, blit=True)
