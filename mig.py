# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.    All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mutual Information Gap from the beta-TC-VAE paper.

Based on "Isolating Sources of Disentanglement in Variational Autoencoders"
(https://arxiv.org/pdf/1802.04942.pdf).
"""
import numpy as np
import sklearn


def obtain_representation(observations, representation_function, batch_size):
    """"Obtain representations from observations.

    Args:
        observations: Observations for which we compute the representation.
        representation_function: Function that takes observation as input and
            outputs a representation.
        batch_size: Batch size to compute the representation.
    Returns:
        representations: Codes (num_codes, num_points)-Numpy array.
    """
    representations = None
    num_points = observations.shape[0]
    i = 0
    while i < num_points:
        num_points_iter = min(num_points - i, batch_size)
        current_observations = observations[i:i + num_points_iter]
        if i == 0:
            representations = representation_function(current_observations)
        else:
            representations = np.vstack((representations,
                                                                     representation_function(
                                                                             current_observations)))
        i += num_points_iter
    return np.transpose(representations)


def _histogram_discretize(target, num_bins):
    """Discretization based on histograms."""
    discretized = np.zeros_like(target)
    for i in range(target.shape[0]):
        discretized[i, :] = np.digitize(target[i, :], np.histogram(
                target[i, :], num_bins)[1][:-1])
    return discretized


def discrete_mutual_info(mus, ys):
    """Compute discrete mutual information."""
    num_codes = mus.shape[0]
    num_factors = ys.shape[0]
    m = np.zeros([num_codes, num_factors])
    for i in range(num_codes):
        for j in range(num_factors):
            m[i, j] = sklearn.metrics.mutual_info_score(ys[j, :], mus[i, :])
    return m


def discrete_entropy(ys):
    """Compute discrete mutual information."""
    num_factors = ys.shape[0]
    h = np.zeros(num_factors)
    for j in range(num_factors):
        h[j] = sklearn.metrics.mutual_info_score(ys[j, :], ys[j, :])
    return h


def _compute_mig(mus_train, ys_train):
    """Computes score based on both training and testing codes and factors."""
    score_dict = {}
    discretized_mus = _histogram_discretize(mus_train,num_bins=3)
    m = discrete_mutual_info(discretized_mus, ys_train)
    assert m.shape[0] == mus_train.shape[0]
    assert m.shape[1] == ys_train.shape[0]
    # m is [num_latents, num_factors]
    entropy = discrete_entropy(ys_train)
    sorted_m = np.sort(m, axis=0)[::-1]
    score_dict["discrete_mig"] = np.divide(sorted_m[0, :] - sorted_m[1, :], entropy[:]).tolist()
    return score_dict


def compute_mig_on_fixed_data(observations, labels, representation_function,
                                                            batch_size=100):
    """Computes the MIG scores on the fixed set of observations and labels.

    Args:
        observations: Observations on which to compute the score. Observations have
            shape (num_observations, 64, 64, num_channels).
        labels: Observed factors of variations.
        representation_function: Function that takes observations as input and
            outputs a dim_representation sized representation for each observation.
        batch_size: Batch size used to compute the representation.

    Returns:
        MIG computed on the provided observations and labels.
    """
    mus = obtain_representation(observations, representation_function,
                                                                        batch_size)
    assert labels.shape[1] == observations.shape[0], "Wrong labels shape."
    assert mus.shape[1] == observations.shape[0], "Wrong representation shape."
    return _compute_mig(mus, labels)

if __name__ == '__main__':
    mutr = np.transpose(np.load('train_data.npz')['latents'])
    ytr = np.transpose(np.load('train_data.npz')['gts'])
    muts = np.transpose(np.load('test_data.npz')['latents'])
    yts = np.transpose(np.load('test_data.npz')['gts'])
    print(_compute_mig(muts,yts))
