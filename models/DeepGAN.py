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
import torch
import torch.nn as nn

try:
    from external.JitGRU.jit_gru import JitGRU
except ImportError as error:
        if 'jit' in error.name.lower():
            print('--------------------------------------------------------------------------------------------')
            print("Could not import 'JitGRU'. Did you forget to initialize this repository's submodules?")
            print("Run 'git submodule update --init --recursive' and try again!")
            print('--------------------------------------------------------------------------------------------')
        raise

from models.model_base import ModelBase
from models.DeepNAG import DeepNAGNet


# ----------------------------------------------------------------------------------------------------------------------
class DeepGAN(ModelBase):
    """The DeepGAN model and the training logic"""

    def __init__(self, num_classes, num_features, opt, device, visualizer=None):
        """
        The constructor

        :param num_classes: the total number of gesture classes
        :param num_features: the dimensionality of each feature vector
        :param opt: an instance of Options
        :param device: the compute device to use
        :param visualizer: an optional visualizer to plot the training progress
        """
        super(DeepGAN, self).__init__(num_classes, num_features, opt, device, visualizer)

        # Instantiate the network and the optimizer
        # DeepGAN uses the same generator as DeepNAG
        self._generator = DeepNAGNet(self._opt.latent_dim,
                                     self._num_features,
                                     self._num_classes)
        # The discriminator model has a similar architecture to the generator
        self._discriminator = DeepGANDiscriminator(self._num_features, self._num_classes)

        self._optimizer = {
            'd': torch.optim.Adam(self._discriminator.parameters(),
                                  lr=self._opt.lr,
                                  betas=(self._opt.beta0, self._opt.beta1)),
            'g': torch.optim.Adam(self._generator.parameters(),
                                  lr=self._opt.lr,
                                  betas=(self._opt.beta0, self._opt.beta1))
         }

        # Metrics
        self.metric_names = [
            'loss_d', 'neg_loss_d', 'loss_d_real', 'loss_d_fake',
            'loss_g',
            'wassersteind_d'
        ]
        self._best_model_which_metric = 'neg_loss_d'
        # Loss-related objects
        self.loss_d = None  # The discriminator's loss
        self.neg_loss_d = None  # The negative discriminator loss. Used to track convergence
        self.loss_d_real = None  # Discriminator's loss on the real data
        self.loss_d_fake = None  # Discriminator's loss on the fake data
        self.wassersteind_d = None  # Wasserstein distance as estimated by the discriminator
        self.loss_g = None  # The generator's loss

        self._to_device()

    def _to_device(self):
        """
        Transfer everything to the correct computation device
        """
        # Put everything in the correct device
        self._generator = torch.nn.DataParallel(self._generator).to(self._device)
        self._discriminator = torch.nn.DataParallel(self._discriminator).to(self._device)

    def _calc_grad_penalty(self, real_data, labels_one_hot, fake_data, lambd):
        """
        Calculates the gradient penalty needed for the WGAN-GP loss function.

        :param real_data: a batch of real samples
        :param labels_one_hot: the one-hot label for the samples
        :param fake_data: a batch of fake samples
        :param lambd: the regularizer value (lambda in WGAN-GP's loss function)
        :return: the calculated gradient penalty
        """
        batch_size = real_data.shape[0]
        alpha = torch.rand(batch_size, 1, 1, device=self.device, requires_grad=True).expand_as(real_data)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        d_interpolates = self._discriminator(interpolates, labels_one_hot)

        gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones_like(d_interpolates),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradients = gradients.view(batch_size, -1)
        result = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambd

        return result

    def _run_one_epoch(self, epoch):
        """
        Runs a single step of training. Note that for DeepGAN, this is not exactly one "epoch". Rather,
        this is a single training step because the discriminator is trained more frequently compared to
        the generator.

        :param epoch: the epoch counter
        """
        critic_iters = self._opt.deepgan_critic_iters
        lambd = self._opt.deepgan_lambda
        batch_size = self._opt.batch_size

        def inf_train_gen():
            """
            Helper function for producing an infinite amount of training data.
            """
            while True:
                for examples, labels, _ in self._data_loader['train']:
                    yield examples, labels

        data = inf_train_gen()

        self._discriminator.train()
        self._generator.train()

        ###########################################
        # Train the discriminator for a few steps #
        ###########################################
        self._discriminator.module.set_freeze(False)  # Make the discriminator trainable

        for iter_d in range(critic_iters):
            examples, labels = next(data)
            examples = examples.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            curr_batch_size = examples.shape[0]
            labels_one_hot = self._to_one_hot(labels)
            # Generate fake examples
            self._generate_new_latent(curr_batch_size)
            fake_examples = self._generator(self._latent, labels_one_hot)

            self._discriminator.zero_grad()

            # Train with real
            self.loss_d_real = self._discriminator(examples, labels_one_hot)
            self.loss_d_real = self.loss_d_real.mean()

            # Train with fake
            self.loss_d_fake = self._discriminator(fake_examples.detach(), labels_one_hot)
            self.loss_d_fake = self.loss_d_fake.mean()

            # Calculate the gradient penalty
            gradient_penalty = self._calc_grad_penalty(examples.data, labels_one_hot, fake_examples.data, lambd)

            self.loss_d = self.loss_d_fake - self.loss_d_real + gradient_penalty
            self.wassersteind_d = self.loss_d_real - self.loss_d_fake
            self.loss_d.backward()
            self._optimizer['d'].step()

        self.neg_loss_d = -self.loss_d  # The negative discriminator loss gives us info about the quality of the model

        ###########################################
        # Train the generator                     #
        ###########################################
        self._discriminator.module.set_freeze(True)  # Freeze the discriminator

        self._generator.zero_grad()
        self._generate_new_latent(batch_size)
        fake_examples = self._generator(self._latent, labels_one_hot)
        self.loss_g = self._discriminator(fake_examples, labels_one_hot)
        self.loss_g = -self.loss_g.mean()
        self.loss_g.backward()
        self._optimizer['g'].step()

        self.bookkeep()

    def _get_state(self):
        """
        Helper function to get the internal state of the model.

        :return: a dictionary containing the internal state of the model.
        """
        return {
            'model': 'DeepGAN',
            'generator': copy.deepcopy(self._generator.state_dict()),
            'discriminator': copy.deepcopy(self._discriminator.state_dict()),
            'optimizer': {
                'g': copy.deepcopy(self._optimizer['g'].state_dict()),
                'd': copy.deepcopy(self._optimizer['d'].state_dict()),
            },
            'normalizer': self._normalizer
        }

    def _load_state(self, state_dict):
        """
        Helper function to load the internal state of a model from a dictionary.

        :param state_dict: the dictionary containing the state to load.
        """
        # Make sure we're loading the correct model
        if state_dict['model'] != 'DeepGAN':
            raise ValueError('The given model to load is not a DeepGAN model!')

        self._generator.load_state_dict(state_dict['generator'])
        self._discriminator.load_state_dict(state_dict['discriminator'])
        self._optimizer['g'].load_state_dict(state_dict['optimizer']['g'])
        self._optimizer['d'].load_state_dict(state_dict['optimizer']['d'])
        self._normalizer = state_dict['normalizer']


# ----------------------------------------------------------------------------------------------------------------------
class DeepGANDiscriminator(nn.Module):
    """
    DeepGAN's discriminator network model. This is a model very similar to uDeepGRU.
    It uses JitGRU as the main recurrent layers. See https://github.com/Maghoumi/JitGRU
    """

    def __init__(self, num_features, num_classes):
        """
        Initializes a new discriminator model.

        :param num_features:  the dimensionality of the input features
        :param num_classes: the total number of sample classes
        """
        super(DeepGANDiscriminator, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        self.gru1 = JitGRU(num_features + num_classes, 512, 1, batch_first=True)
        self.gru2 = JitGRU(512, 256, 1, batch_first=True)
        self.gru3 = JitGRU(256, 128, 1, batch_first=True)

        self.fc = nn.Linear(128, 1)

    def forward(self, x, labels_one_hot):
        """
        The forward pass

        :param x: a batch of input samples (either real or fake)
        :param labels_one_hot: the one-hot labels of each given sample
        :return: an embedding of the input samples used for computing the Wasserstein distance
        """
        x = torch.cat([x, labels_one_hot], dim=2)
        output, _ = self.gru1(x)
        output, _ = self.gru2(output)
        output, _ = self.gru3(output)

        output = self.fc(output[:, -1, :])  # Just need the output for the last time-step
        return output.squeeze()

    def set_freeze(self, freeze):
        """
        Sets the freeze state of the model (frozen means no gradients will be computed).

        :param freeze: boolean flag to use as the freeze state.
        """
        for p in self.parameters():
            p.requires_grad = not freeze
