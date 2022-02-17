# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# Implementation of causal CNNs partly taken and modified from
# https://github.com/locuslab/TCN/blob/master/TCN/tcn.py, originally created
# with the following license.

# MIT License

# Copyright (c) 2018 CMU Locus Lab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import math

import numpy
import sklearn
import sklearn.externals
import sklearn.model_selection
import sklearn.svm
import torch
import torch.utils.data
from tqdm import tqdm


class Dataset(torch.utils.data.Dataset):
    """PyTorch wrapper for a numpy dataset.

    @param dataset Numpy array representing the dataset.
    """

    def __init__(self, dataset, offset):
        self.dataset = dataset
        self.offset = offset

    def __len__(self):
        return numpy.shape(self.dataset)[0]

    def __getitem__(self, index):
        sample = numpy.copy(self.dataset[index]) - self.offset[index]
        T, N, L = sample.shape
        return torch.from_numpy(sample.reshape(T, N * L).transpose().astype(numpy.float32))


class LabelledDataset(torch.utils.data.Dataset):
    """PyTorch wrapper for a numpy dataset and its associated labels.

    @param dataset Numpy array representing the dataset.
    @param labels One-dimensional array of the same length as dataset with non-negative int values.
    """

    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels

    def __len__(self):
        return numpy.shape(self.dataset)[0]

    def __getitem__(self, index):
        return numpy.copy(self.dataset[index]), numpy.copy(self.labels[index])


class TripletLoss(torch.nn.modules.loss._Loss):
    """Triplet loss for representations of time series. Optimized for training sets where all time series have the same
    length.

    Takes as input a tensor as the chosen batch to compute the loss, a PyTorch module as the encoder, a 3D tensor
    (`B`, `C`, `L`) containing the training set, where `B` is the batch size, `C` is the number of channels and `L` is
    the length of the time series, as well as a boolean which, if True, enables to save GPU memory by propagating
    gradients after each loss term, instead of doing it after computing the whole loss.

    The triplets are chosen in the following manner. First the size of the positive and negative samples are randomly
    chosen in the range of lengths of time series in the dataset. The size of the anchor time series is randomly chosen
    with the same length upper bound but the the length of the positive samples as lower bound. An anchor of this length
    is then chosen randomly in the given time series of the train set, and positive samples are randomly chosen among
    subseries of the anchor. Finally, negative samples of the chosen length are randomly chosen in random time series of
    the train set.

    @param compared_length Maximum length of randomly chosen time series. If None, this parameter is ignored.
    @param nb_random_samples Number of negative samples per batch example.
    @param negative_penalty Multiplicative coefficient for the negative sample loss.
    """

    def __init__(self, compared_length, nb_random_samples, negative_penalty):
        super(TripletLoss, self).__init__()
        self.compared_length = compared_length
        if self.compared_length is None:
            self.compared_length = numpy.inf
        self.nb_random_samples = nb_random_samples
        self.negative_penalty = negative_penalty

    def forward(self, batch, encoder, train, offset, save_memory=False):
        batch_size = batch.shape[0]
        train_size = train.shape[0]
        length = min(self.compared_length, train.shape[2])

        # For each batch element, we pick nb_random_samples possible random
        # time series in the training set (choice of batches from where the
        # negative examples will be sampled)
        samples = numpy.random.choice(train_size, size=(self.nb_random_samples, batch_size))
        samples = torch.LongTensor(samples)

        # Choice of length of positive and negative samples
        length_pos_neg = numpy.random.randint(1, high=length + 1)

        # We choose for each batch example a random interval in the time
        # series, which is the 'anchor'
        random_length = numpy.random.randint(length_pos_neg, high=length + 1)  # Length of anchors
        beginning_batches = numpy.random.randint(
            0, high=length - random_length + 1, size=batch_size
        )  # Start of anchors

        # The positive samples are chosen at random in the chosen anchors
        beginning_samples_pos = numpy.random.randint(
            0, high=random_length - length_pos_neg + 1, size=batch_size
        )  # Start of positive samples in the anchors
        # Start of positive samples in the batch examples
        beginning_positive = beginning_batches + beginning_samples_pos
        # End of positive samples in the batch examples
        end_positive = beginning_positive + length_pos_neg

        # We randomly choose nb_random_samples potential negative samples for
        # each batch example
        beginning_samples_neg = numpy.random.randint(
            0, high=length - length_pos_neg + 1, size=(self.nb_random_samples, batch_size)
        )

        representation = encoder(
            torch.cat(
                [
                    batch[j : j + 1, :, beginning_batches[j] : beginning_batches[j] + random_length]
                    for j in range(batch_size)
                ]
            )
        )  # Anchors representations

        positive_representation = encoder(
            torch.cat(
                [batch[j : j + 1, :, end_positive[j] - length_pos_neg : end_positive[j]] for j in range(batch_size)]
            )
        )  # Positive samples representations

        size_representation = representation.shape[1]
        # Positive loss: -logsigmoid of dot product between anchor and positive
        # representations
        loss = -torch.mean(
            torch.nn.functional.logsigmoid(
                torch.bmm(
                    representation.view(batch_size, 1, size_representation),
                    positive_representation.view(batch_size, size_representation, 1),
                )
            )
        )

        # If required, backward through the first computed term of the loss and
        # free from the graph everything related to the positive sample
        if save_memory:
            loss.backward(retain_graph=True)
            loss = 0
            del positive_representation
            torch.cuda.empty_cache()

        multiplicative_ratio = self.negative_penalty / self.nb_random_samples
        for i in range(self.nb_random_samples):
            # Negative loss: -logsigmoid of minus the dot product between
            # anchor and negative representations
            negative_representation = encoder(
                torch.cat(
                    [
                        torch.from_numpy(
                            (
                                train[samples[i, j] : samples[i, j] + 1][
                                    :, beginning_samples_neg[i, j] : beginning_samples_neg[i, j] + length_pos_neg, :, :
                                ]
                                - offset[samples[i, j] : samples[i, j] + 1, None, None]
                            )
                            .copy()
                            .astype(numpy.float32)
                            .reshape(1, length_pos_neg, -1)
                            .transpose(0, 2, 1)
                        ).to(loss.device)
                        for j in range(batch_size)
                    ]
                )
            )
            loss += multiplicative_ratio * -torch.mean(
                torch.nn.functional.logsigmoid(
                    -torch.bmm(
                        representation.view(batch_size, 1, size_representation),
                        negative_representation.view(batch_size, size_representation, 1),
                    )
                )
            )
            # If required, backward through the first computed term of the loss
            # and free from the graph everything related to the negative sample
            # Leaves the last backward pass to the training procedure
            if save_memory and i != self.nb_random_samples - 1:
                loss.backward(retain_graph=True)
                loss = 0
                del negative_representation
                torch.cuda.empty_cache()

        return loss


class TripletLossVaryingLength(torch.nn.modules.loss._Loss):
    """Triplet loss for representations of time series where the training set features time series with unequal lengths.

    Takes as input a tensor as the chosen batch to compute the loss, a PyTorch module as the encoder, a 3D tensor
    (`B`, `C`, `L`) containing the training set, where `B` is the batch size, `C` is the number of channels and `L` is
    the maximum length of the time series (NaN values representing the end of a shorter time series), as well as a
    boolean which, if True, enables to save GPU memory by propagating gradients after each loss term, instead of doing
    it after computing the whole loss.

    The triplets are chosen in the following manner. First the sizes of positive and negative samples are randomly
    chosen in the range of lengths of time series in the dataset. The size of the anchor time series is randomly chosen
    with the same length upper bound but the the length of the positive samples as lower bound. An anchor of this length
    is then chosen randomly in the given time series of the train set, and positive samples are randomly chosen among
    subseries of the anchor. Finally, negative samples of the chosen length are randomly chosen in random time series of
    the train set.

    @param compared_length Maximum length of randomly chosen time series. If None, this parameter is ignored.
    @param nb_random_samples Number of negative samples per batch example.
    @param negative_penalty Multiplicative coefficient for the negative sample loss.
    """

    def __init__(self, compared_length, nb_random_samples, negative_penalty):
        super(TripletLossVaryingLength, self).__init__()
        self.compared_length = compared_length
        if self.compared_length is None:
            self.compared_length = numpy.inf
        self.nb_random_samples = nb_random_samples
        self.negative_penalty = negative_penalty

    def forward(self, batch, encoder, train, offset, save_memory=False):
        batch_size = batch.shape[0]
        train_size = train.shape[0]
        max_length = train.shape[2]

        # For each batch element, we pick nb_random_samples possible random
        # time series in the training set (choice of batches from where the
        # negative examples will be sampled)
        samples = numpy.random.choice(train_size, size=(self.nb_random_samples, batch_size))
        samples = torch.LongTensor(samples)

        # Computation of the lengths of the relevant time series
        with torch.no_grad():
            lengths_batch = max_length - torch.sum(torch.isnan(batch[:, 0]), 1).data.cpu().numpy()
            lengths_samples = numpy.empty((self.nb_random_samples, batch_size), dtype=int)
            for i in range(self.nb_random_samples):
                lengths_samples[i] = max_length - torch.sum(torch.isnan(train[samples[i], 0]), 1).data.cpu().numpy()

        # Choice of lengths of positive and negative samples
        lengths_pos = numpy.empty(batch_size, dtype=int)
        lengths_neg = numpy.empty((self.nb_random_samples, batch_size), dtype=int)
        for j in range(batch_size):
            lengths_pos[j] = numpy.random.randint(1, high=min(self.compared_length, lengths_batch[j]) + 1)
            for i in range(self.nb_random_samples):
                lengths_neg[i, j] = numpy.random.randint(1, high=min(self.compared_length, lengths_samples[i, j]) + 1)

        # We choose for each batch example a random interval in the time
        # series, which is the 'anchor'
        random_length = numpy.array(
            [
                numpy.random.randint(lengths_pos[j], high=min(self.compared_length, lengths_batch[j]) + 1)
                for j in range(batch_size)
            ]
        )  # Length of anchors
        beginning_batches = numpy.array(
            [numpy.random.randint(0, high=lengths_batch[j] - random_length[j] + 1) for j in range(batch_size)]
        )  # Start of anchors

        # The positive samples are chosen at random in the chosen anchors
        # Start of positive samples in the anchors
        beginning_samples_pos = numpy.array(
            [numpy.random.randint(0, high=random_length[j] - lengths_pos[j] + 1) for j in range(batch_size)]
        )
        # Start of positive samples in the batch examples
        beginning_positive = beginning_batches + beginning_samples_pos
        # End of positive samples in the batch examples
        end_positive = beginning_positive + lengths_pos

        # We randomly choose nb_random_samples potential negative samples for
        # each batch example
        beginning_samples_neg = numpy.array(
            [
                [numpy.random.randint(0, high=lengths_samples[i, j] - lengths_neg[i, j] + 1) for j in range(batch_size)]
                for i in range(self.nb_random_samples)
            ]
        )

        representation = torch.cat(
            [
                encoder(batch[j : j + 1, :, beginning_batches[j] : beginning_batches[j] + random_length[j]])
                for j in range(batch_size)
            ]
        )  # Anchors representations

        positive_representation = torch.cat(
            [
                encoder(batch[j : j + 1, :, end_positive[j] - lengths_pos[j] : end_positive[j]])
                for j in range(batch_size)
            ]
        )  # Positive samples representations

        size_representation = representation.shape[1]
        # Positive loss: -logsigmoid of dot product between anchor and positive
        # representations
        loss = -torch.mean(
            torch.nn.functional.logsigmoid(
                torch.bmm(
                    representation.view(batch_size, 1, size_representation),
                    positive_representation.view(batch_size, size_representation, 1),
                )
            )
        )

        # If required, backward through the first computed term of the loss and
        # free from the graph everything related to the positive sample
        if save_memory:
            loss.backward(retain_graph=True)
            loss = 0
            del positive_representation
            torch.cuda.empty_cache()

        multiplicative_ratio = self.negative_penalty / self.nb_random_samples
        for i in range(self.nb_random_samples):
            # Negative loss: -logsigmoid of minus the dot product between
            # anchor and negative representations
            negative_representation = torch.cat(
                [
                    encoder(
                        torch.from_numpy(
                            (
                                train[samples[i, j] : samples[i, j] + 1][
                                    :,
                                    beginning_samples_neg[i, j] : beginning_samples_neg[i, j] + lengths_neg[i, j],
                                    :,
                                    :,
                                ]
                                - offset[samples[i, j] : samples[i, j] + 1, None, None]
                            )
                            .copy()
                            .astype(numpy.float32)
                            .reshape(1, lengths_neg[i, j], -1)
                            .transpose(0, 2, 1)
                        ).to(loss.device)
                    )
                    for j in range(batch_size)
                ]
            )
            loss += multiplicative_ratio * -torch.mean(
                torch.nn.functional.logsigmoid(
                    -torch.bmm(
                        representation.view(batch_size, 1, size_representation),
                        negative_representation.view(batch_size, size_representation, 1),
                    )
                )
            )
            # If required, backward through the first computed term of the loss
            # and free from the graph everything related to the negative sample
            # Leaves the last backward pass to the training procedure
            if save_memory and i != self.nb_random_samples - 1:
                loss.backward(retain_graph=True)
                loss = 0
                del negative_representation
                torch.cuda.empty_cache()

        return loss


class TimeSeriesEncoderClassifier(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    """ "Virtual" class to wrap an encoder of time series as a PyTorch module and a SVM classifier with RBF kernel on
    top of its computed representations in a scikit-learn class.

    All inheriting classes should implement the get_params and set_params methods, as in the recommendations of
    scikit-learn.

    @param compared_length Maximum length of randomly chosen time series. If None, this parameter is ignored.
    @param nb_random_samples Number of randomly chosen intervals to select the final negative sample in the loss.
    @param negative_penalty Multiplicative coefficient for the negative sample loss.
    @param batch_size Batch size used during the training of the encoder.
    @param nb_steps Number of optimization steps to perform for the training of the encoder.
    @param lr learning rate of the Adam optimizer used to train the encoder.
    @param penalty Penalty term for the SVM classifier. If None and if the number of samples is high enough, performs a
           hyperparameter search to find a suitable constant.
    @param early_stopping Enables, if not None, early stopping heuristic for the training of the representations, based
           on the final score. Representations are still learned unsupervisedly in this case. If the number of samples
           per class is no more than 10, disables this heuristic. If not None, accepts an integer representing the
           patience of the early stopping strategy.
    @param encoder Encoder PyTorch module.
    @param params Dictionaries of the parameters of the encoder.
    @param in_channels Number of input channels of the time series.
    @param cuda Transfers, if True, all computations to the GPU.
    @param gpu GPU index to use, if CUDA is enabled.
    """

    def __init__(
        self,
        compared_length,
        nb_random_samples,
        negative_penalty,
        batch_size,
        nb_steps,
        lr,
        penalty,
        early_stopping,
        encoder,
        params,
        in_channels,
        out_channels,
        cuda=False,
        gpu=0,
    ):
        self.architecture = ""
        self.cuda = cuda
        self.gpu = gpu
        self.batch_size = batch_size
        self.nb_steps = nb_steps
        self.lr = lr
        self.penalty = penalty
        self.early_stopping = early_stopping
        self.encoder = encoder
        self.params = params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.loss = TripletLoss(compared_length, nb_random_samples, negative_penalty)
        self.loss_varying = TripletLossVaryingLength(compared_length, nb_random_samples, negative_penalty)
        self.classifier = sklearn.svm.SVC()
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)

    def save_encoder(self, prefix_file):
        """Saves the encoder and the SVM classifier.

        @param prefix_file Path and prefix of the file where the models should
               be saved (at '$(prefix_file)_$(architecture)_encoder.pth').
        """
        torch.save(self.encoder.state_dict(), prefix_file + "_" + self.architecture + "_encoder.pth")

    def save(self, prefix_file):
        """Saves the encoder and the SVM classifier.

        @param prefix_file Path and prefix of the file where the models should
               be saved (at '$(prefix_file)_$(architecture)_classifier.pkl' and
               '$(prefix_file)_$(architecture)_encoder.pth').
        """
        self.save_encoder(prefix_file)
        sklearn.externals.joblib.dump(self.classifier, prefix_file + "_" + self.architecture + "_classifier.pkl")

    def load_encoder(self, prefix_file):
        """Loads an encoder.

        @param prefix_file Path and prefix of the file where the model should
               be loaded (at '$(prefix_file)_$(architecture)_encoder.pth').
        """
        if self.cuda:
            self.encoder.load_state_dict(
                torch.load(
                    prefix_file + "_" + self.architecture + "_encoder.pth",
                    map_location=lambda storage, loc: storage.cuda(self.gpu),
                )
            )
        else:
            self.encoder.load_state_dict(
                torch.load(
                    prefix_file + "_" + self.architecture + "_encoder.pth", map_location=lambda storage, loc: storage
                )
            )

    def load(self, prefix_file):
        """Loads an encoder and an SVM classifier.

        @param prefix_file Path and prefix of the file where the models should
               be loaded (at '$(prefix_file)_$(architecture)_classifier.pkl'
               and '$(prefix_file)_$(architecture)_encoder.pth').
        """
        self.load_encoder(prefix_file)
        self.classifier = sklearn.externals.joblib.load(prefix_file + "_" + self.architecture + "_classifier.pkl")

    def fit_classifier(self, features, y):
        """Trains the classifier using precomputed features. Uses an SVM classifier with RBF kernel.

        @param features Computed features of the training set.
        @param y Training labels.
        """
        nb_classes = numpy.shape(numpy.unique(y, return_counts=True)[1])[0]
        train_size = numpy.shape(features)[0]
        # To use a 1-NN classifier, no need for model selection, simply
        # replace the code by the following:
        # import sklearn.neighbors
        # self.classifier = sklearn.neighbors.KNeighborsClassifier(
        #     n_neighbors=1
        # )
        # return self.classifier.fit(features, y)
        self.classifier = sklearn.svm.SVC(
            C=1 / self.penalty if self.penalty is not None and self.penalty > 0 else numpy.inf, gamma="scale"
        )
        if train_size // nb_classes < 5 or train_size < 50 or self.penalty is not None:
            return self.classifier.fit(features, y)
        else:
            grid_search = sklearn.model_selection.GridSearchCV(
                self.classifier,
                {
                    "C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, numpy.inf],
                    "kernel": ["rbf"],
                    "degree": [3],
                    "gamma": ["scale"],
                    "coef0": [0],
                    "shrinking": [True],
                    "probability": [False],
                    "tol": [0.001],
                    "cache_size": [200],
                    "class_weight": [None],
                    "verbose": [False],
                    "max_iter": [10000000],
                    "decision_function_shape": ["ovr"],
                    "random_state": [None],
                },
                cv=5,
                iid=False,
                n_jobs=5,
            )
            if train_size <= 10000:
                grid_search.fit(features, y)
            else:
                # If the training set is too large, subsample 10000 train
                # examples
                split = sklearn.model_selection.train_test_split(
                    features, y, train_size=10000, random_state=0, stratify=y
                )
                grid_search.fit(split[0], split[2])
            self.classifier = grid_search.best_estimator_
            return self.classifier

    def fit_encoder(self, X, XO, y=None, save_memory=False, verbose=False):
        """Trains the encoder unsupervisedly using the given training data.

        @param X Training set.
        @param y Training labels, used only for early stopping, if enabled. If
               None, disables early stopping in the method.
        @param save_memory If True, enables to save GPU memory by propagating
               gradients after each loss term of the encoder loss, instead of
               doing it after computing the whole loss.
        @param verbose Enables, if True, to monitor which epoch is running in
               the encoder training.
        """
        # Check if the given time series have unequal lengths
        varying = bool(numpy.isnan(numpy.sum(X)))
        print("varying", varying)

        if y is not None:
            nb_classes = numpy.shape(numpy.unique(y, return_counts=True)[1])[0]
            train_size = numpy.shape(X)[0]
            ratio = train_size // nb_classes

        train_torch_dataset = Dataset(X, XO)
        train_generator = torch.utils.data.DataLoader(train_torch_dataset, batch_size=self.batch_size, shuffle=True)

        max_score = 0
        epochs = 0  # Number of performed epochs
        count = 0  # Count of number of epochs without improvement
        # Will be true if, by enabling epoch_selection, a model was selected
        # using cross-validation
        found_best = False

        # Encoder training
        losses = []
        pbar = tqdm(range(self.nb_steps))
        for i in pbar:
            if verbose:
                pbar.write(
                    f"Epoch: {epochs + 1} \t Steps: {i}/{self.nb_steps} \t Recent Average Loss: {numpy.mean(losses[-100:]) if len(losses) > 0 else -1}"
                )
            for batch in train_generator:
                if self.cuda:
                    batch = batch.cuda(self.gpu)
                self.optimizer.zero_grad()
                if not varying:
                    loss = self.loss(batch, self.encoder, X, XO, save_memory=save_memory)
                else:
                    loss = self.loss_varying(batch, self.encoder, X, XO, save_memory=save_memory)
                loss.backward()
                self.optimizer.step()
                if verbose:
                    losses.append(loss.item())
            epochs += 1
            # Early stopping strategy
            if self.early_stopping is not None and y is not None and (ratio >= 5 and train_size >= 50):
                # Computes the best regularization parameters
                features = self.encode(X)
                self.classifier = self.fit_classifier(features, y)
                # Cross validation score
                score = numpy.mean(
                    sklearn.model_selection.cross_val_score(self.classifier, features, y=y, cv=5, n_jobs=5)
                )
                count += 1
                # If the model is better than the previous one, update
                if score > max_score:
                    count = 0
                    found_best = True
                    max_score = score
                    best_encoder = type(self.encoder)(**self.params)
                    if self.cuda:
                        best_encoder.cuda(self.gpu)
                    best_encoder.load_state_dict(self.encoder.state_dict())
            if count == self.early_stopping:
                break

        # If a better model was found, use it
        if found_best:
            self.encoder = best_encoder

        return self.encoder

    def fit(self, X, y, save_memory=False, verbose=False):
        """Trains sequentially the encoder unsupervisedly and then the classifier using the given labels over the
        learned features.

        @param X Training set.
        @param y Training labels.
        @param save_memory If True, enables to save GPU memory by propagating
               gradients after each loss term of the encoder loss, instead of
               doing it after computing the whole loss.
        @param verbose Enables, if True, to monitor which epoch is running in
               the encoder training.
        """
        # Fitting encoder
        self.encoder = self.fit_encoder(X, y=y, save_memory=save_memory, verbose=verbose)

        # SVM classifier training
        features = self.encode(X)
        self.classifier = self.fit_classifier(features, y)

        return self

    def encode(self, X, batch_size=50):
        """Outputs the representations associated to the input by the encoder.

        @param X Testing set.
        @param batch_size Size of batches used for splitting the test data to
               avoid out of memory errors when using CUDA. Ignored if the
               testing set contains time series of unequal lengths.
        """
        # Check if the given time series have unequal lengths
        varying = bool(numpy.isnan(numpy.sum(X)))

        test = Dataset(X)
        test_generator = torch.utils.data.DataLoader(test, batch_size=batch_size if not varying else 1)
        features = numpy.zeros((numpy.shape(X)[0], self.out_channels))
        self.encoder = self.encoder.eval()

        count = 0
        with torch.no_grad():
            if not varying:
                for batch in test_generator:
                    if self.cuda:
                        batch = batch.cuda(self.gpu)
                    features[count * batch_size : (count + 1) * batch_size] = self.encoder(batch).cpu()
                    count += 1
            else:
                for batch in test_generator:
                    if self.cuda:
                        batch = batch.cuda(self.gpu)
                    length = batch.shape[2] - torch.sum(torch.isnan(batch[0, 0])).data.cpu().numpy()
                    features[count : count + 1] = self.encoder(batch[:, :, :length]).cpu()
                    count += 1

        self.encoder = self.encoder.train()
        return features

    def encode_window(self, X, window, batch_size=50, window_batch_size=10000):
        """Outputs the representations associated to the input by the encoder, for each subseries of the input of the
        given size (sliding window representations).

        @param X Testing set.
        @param window Size of the sliding window.
        @param batch_size Size of batches used for splitting the test data to avoid out of memory errors when using CUDA
        @param window_batch_size Size of batches of windows to compute in a run of encode, to save RAM.
        """
        features = numpy.empty((numpy.shape(X)[0], self.out_channels, numpy.shape(X)[2] - window + 1))
        masking = numpy.empty((min(window_batch_size, numpy.shape(X)[2] - window + 1), numpy.shape(X)[1], window))
        for b in range(numpy.shape(X)[0]):
            for i in range(math.ceil((numpy.shape(X)[2] - window + 1) / window_batch_size)):
                for j in range(i * window_batch_size, min((i + 1) * window_batch_size, numpy.shape(X)[2] - window + 1)):
                    j0 = j - i * window_batch_size
                    masking[j0, :, :] = X[b, :, j : j + window]
                features[b, :, i * window_batch_size : (i + 1) * window_batch_size] = numpy.swapaxes(
                    self.encode(masking[: j0 + 1], batch_size=batch_size), 0, 1
                )
        return features

    def predict(self, X, batch_size=50):
        """Outputs the class predictions for the given test data.

        @param X Testing set.
        @param batch_size Size of batches used for splitting the test data to avoid out of memory errors when using
               CUDA. Ignored if the testing set contains time series of unequal lengths.
        """
        features = self.encode(X, batch_size=batch_size)
        return self.classifier.predict(features)

    def score(self, X, y, batch_size=50):
        """Outputs accuracy of the SVM classifier on the given testing data.

        @param X Testing set.
        @param y Testing labels.
        @param batch_size Size of batches used for splitting the test data to avoid out of memory errors when using
               CUDA. Ignored if the testing set contains time series of unequal lengths.
        """
        features = self.encode(X, batch_size=batch_size)
        return self.classifier.score(features, y)


class Chomp1d(torch.nn.Module):
    """Removes the last elements of a time series.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the batch size, `C` is the number of input
    channels, and `L` is the length of the input. Outputs a three-dimensional tensor (`B`, `C`, `L - s`) where `s` is
    the number of elements to remove.

    @param chomp_size Number of elements to remove.
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size]


class SqueezeChannels(torch.nn.Module):
    """Squeezes, in a three-dimensional tensor, the third dimension."""

    def __init__(self):
        super(SqueezeChannels, self).__init__()

    def forward(self, x):
        return x.squeeze(2)


class CausalConvolutionBlock(torch.nn.Module):
    """Causal convolution block, composed sequentially of two causal convolutions (with leaky ReLU activation
    functions), and a parallel residual connection.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the batch size, `C` is the number of input
    channels, and `L` is the length of the input. Outputs a three-dimensional tensor (`B`, `C`, `L`).

    @param in_channels Number of input channels.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    @param dilation Dilation parameter of non-residual convolutions.
    @param final Disables, if True, the last activation function.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super(CausalConvolutionBlock, self).__init__()

        # Computes left padding so that the applied convolutions are causal
        padding = (kernel_size - 1) * dilation

        # First causal convolution
        conv1 = torch.nn.utils.weight_norm(
            torch.nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        )
        # The truncation makes the convolution causal
        chomp1 = Chomp1d(padding)
        relu1 = torch.nn.LeakyReLU()

        # Second causal convolution
        conv2 = torch.nn.utils.weight_norm(
            torch.nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        )
        chomp2 = Chomp1d(padding)
        relu2 = torch.nn.LeakyReLU()

        # Causal network
        self.causal = torch.nn.Sequential(conv1, chomp1, relu1, conv2, chomp2, relu2)

        # Residual connection
        self.upordownsample = torch.nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

        # Final activation function
        self.relu = torch.nn.LeakyReLU() if final else None

    def forward(self, x):
        out_causal = self.causal(x)
        res = x if self.upordownsample is None else self.upordownsample(x)
        if self.relu is None:
            return out_causal + res
        else:
            return self.relu(out_causal + res)


class CausalCNN(torch.nn.Module):
    """Causal CNN, composed of a sequence of causal convolution blocks.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the batch size, `C` is the number of input
    channels, and `L` is the length of the input. Outputs a three-dimensional tensor (`B`, `C_out`, `L`).

    @param in_channels Number of input channels.
    @param channels Number of channels processed in the network and of output channels.
    @param depth Depth of the network.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    """

    def __init__(self, in_channels, channels, depth, out_channels, kernel_size):
        super(CausalCNN, self).__init__()

        layers = []  # List of causal convolution blocks
        dilation_size = 1  # Initial dilation size

        for i in range(depth):
            in_channels_block = in_channels if i == 0 else channels
            layers += [CausalConvolutionBlock(in_channels_block, channels, kernel_size, dilation_size)]
            dilation_size *= 2  # Doubles the dilation size at each step

        # Last layer
        layers += [CausalConvolutionBlock(channels, out_channels, kernel_size, dilation_size)]

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class CausalCNNEncoder(torch.nn.Module):
    """Encoder of a time series using a causal CNN: the computed representation is the output of a fully connected layer
    applied to the output of an adaptive max pooling layer applied on top of the causal CNN, which reduces the length of
    the time series to a fixed size.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the batch size, `C` is the number of input
    channels, and `L` is the length of the input. Outputs a three-dimensional tensor (`B`, `C`).

    @param in_channels Number of input channels.
    @param channels Number of channels manipulated in the causal CNN.
    @param depth Depth of the causal CNN.
    @param reduced_size Fixed length to which the output time series of the causal CNN is reduced.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    """

    def __init__(self, in_channels, channels, depth, reduced_size, out_channels, kernel_size):
        super(CausalCNNEncoder, self).__init__()
        causal_cnn = CausalCNN(in_channels, channels, depth, reduced_size, kernel_size)
        reduce_size = torch.nn.AdaptiveMaxPool1d(1)
        squeeze = SqueezeChannels()  # Squeezes the third dimension (time)
        linear = torch.nn.Linear(reduced_size, out_channels)
        self.network = torch.nn.Sequential(causal_cnn, reduce_size, squeeze, linear)

    def forward(self, x):
        return self.network(x)


class CausalCNNEncoderClassifier(TimeSeriesEncoderClassifier):
    """Wraps a causal CNN encoder of time series as a PyTorch module and a SVM classifier on top of its computed
    representations in a scikit-learn class.

    @param compared_length Maximum length of randomly chosen time series. If None, this parameter is ignored.
    @param nb_random_samples Number of randomly chosen intervals to select the final negative sample in the loss.
    @param negative_penalty Multiplicative coefficient for the negative sample loss.
    @param batch_size Batch size used during the training of the encoder.
    @param nb_steps Number of optimization steps to perform for the training of the encoder.
    @param lr learning rate of the Adam optimizer used to train the encoder.
    @param penalty Penalty term for the SVM classifier. If None and if the number of samples is high enough, performs a
           hyperparameter search to find a suitable constant.
    @param early_stopping Enables, if not None, early stopping heuristic for the training of the representations, based
           on the final score. Representations are still learned unsupervisedly in this case. If the number of samples
           per class is no more than 10, disables this heuristic. If not None, accepts an integer representing the
           patience of the early stopping strategy.
    @param channels Number of channels manipulated in the causal CNN.
    @param depth Depth of the causal CNN.
    @param reduced_size Fixed length to which the output time series of the causal CNN is reduced.
    @param out_channels Number of features in the final output.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    @param in_channels Number of input channels of the time series.
    @param cuda Transfers, if True, all computations to the GPU.
    @param gpu GPU index to use, if CUDA is enabled.
    """

    def __init__(
        self,
        compared_length=50,
        nb_random_samples=10,
        negative_penalty=1,
        batch_size=1,
        nb_steps=2000,
        lr=0.001,
        penalty=1,
        early_stopping=None,
        channels=10,
        depth=1,
        reduced_size=10,
        out_channels=10,
        kernel_size=4,
        in_channels=1,
        cuda=False,
        gpu=0,
    ):
        super(CausalCNNEncoderClassifier, self).__init__(
            compared_length,
            nb_random_samples,
            negative_penalty,
            batch_size,
            nb_steps,
            lr,
            penalty,
            early_stopping,
            self.__create_encoder(in_channels, channels, depth, reduced_size, out_channels, kernel_size, cuda, gpu),
            self.__encoder_params(in_channels, channels, depth, reduced_size, out_channels, kernel_size),
            in_channels,
            out_channels,
            cuda,
            gpu,
        )
        self.architecture = "CausalCNN"
        self.channels = channels
        self.depth = depth
        self.reduced_size = reduced_size
        self.kernel_size = kernel_size

    def __create_encoder(self, in_channels, channels, depth, reduced_size, out_channels, kernel_size, cuda, gpu):
        encoder = CausalCNNEncoder(in_channels, channels, depth, reduced_size, out_channels, kernel_size)
        if cuda:
            encoder.cuda(gpu)
        return encoder

    def __encoder_params(self, in_channels, channels, depth, reduced_size, out_channels, kernel_size):
        return {
            "in_channels": in_channels,
            "channels": channels,
            "depth": depth,
            "reduced_size": reduced_size,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
        }

    def encode_sequence(self, X, batch_size=50):
        """Outputs the representations associated to the input by the encoder, from the start of the time series to each
        time step (i.e., the evolution of the representations of the input time series with repect to time steps).

        Takes advantage of the causal CNN (before the max pooling), which ensures that its output at time step i only
        depends on time step i and previous time steps.

        @param X Testing set.
        @param batch_size Size of batches used for splitting the test data to avoid out of memory errors when using
        CUDA. Ignored if the testing set contains time series of unequal lengths.
        """
        # Check if the given time series have unequal lengths
        varying = bool(numpy.isnan(numpy.sum(X)))

        test = Dataset(X)
        test_generator = torch.utils.data.DataLoader(
            test, batch_size=batch_size if not varying else 1, num_workers=batch_size
        )
        length = numpy.shape(X)[2]
        features = numpy.full((numpy.shape(X)[0], self.out_channels, length), numpy.nan)
        self.encoder = self.encoder.eval()

        causal_cnn = self.encoder.network[0]
        linear = self.encoder.network[3]

        count = 0
        with torch.no_grad():
            if not varying:
                for batch in test_generator:
                    if self.cuda:
                        batch = batch.cuda(self.gpu)
                    # First applies the causal CNN
                    output_causal_cnn = causal_cnn(batch)
                    after_pool = torch.empty(output_causal_cnn.shape, dtype=torch.float)
                    if self.cuda:
                        after_pool = after_pool.cuda(self.gpu)
                    after_pool[:, :, 0] = output_causal_cnn[:, :, 0]
                    # Then for each time step, computes the output of the max
                    # pooling layer
                    for i in range(1, length):
                        after_pool[:, :, i] = torch.max(
                            torch.cat([after_pool[:, :, i - 1 : i], output_causal_cnn[:, :, i : i + 1]], dim=2), dim=2
                        )[0]
                    features[count * batch_size : (count + 1) * batch_size, :, :] = torch.transpose(
                        linear(torch.transpose(after_pool, 1, 2)), 1, 2
                    )
                    count += 1
            else:
                for batch in test_generator:
                    if self.cuda:
                        batch = batch.cuda(self.gpu)
                    length = batch.shape[2] - torch.sum(torch.isnan(batch[0, 0])).data.cpu().numpy()
                    output_causal_cnn = causal_cnn(batch)
                    after_pool = torch.empty(output_causal_cnn.shape, dtype=torch.float)
                    if self.cuda:
                        after_pool = after_pool.cuda(self.gpu)
                    after_pool[:, :, 0] = output_causal_cnn[:, :, 0]
                    for i in range(1, length):
                        after_pool[:, :, i] = torch.max(
                            torch.cat([after_pool[:, :, i - 1 : i], output_causal_cnn[:, :, i : i + 1]], dim=2), dim=2
                        )[0]
                    features[count : count + 1, :, :] = torch.transpose(linear(torch.transpose(after_pool, 1, 2)), 1, 2)
                    count += 1

        self.encoder = self.encoder.train()
        return features

    def get_params(self, deep=True):
        return {
            "compared_length": self.loss.compared_length,
            "nb_random_samples": self.loss.nb_random_samples,
            "negative_penalty": self.loss.negative_penalty,
            "batch_size": self.batch_size,
            "nb_steps": self.nb_steps,
            "lr": self.lr,
            "penalty": self.penalty,
            "early_stopping": self.early_stopping,
            "channels": self.channels,
            "depth": self.depth,
            "reduced_size": self.reduced_size,
            "kernel_size": self.kernel_size,
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "cuda": self.cuda,
            "gpu": self.gpu,
        }

    def set_params(
        self,
        compared_length,
        nb_random_samples,
        negative_penalty,
        batch_size,
        nb_steps,
        lr,
        penalty,
        early_stopping,
        channels,
        depth,
        reduced_size,
        out_channels,
        kernel_size,
        in_channels,
        cuda,
        gpu,
    ):
        self.__init__(
            compared_length,
            nb_random_samples,
            negative_penalty,
            batch_size,
            nb_steps,
            lr,
            penalty,
            early_stopping,
            channels,
            depth,
            reduced_size,
            out_channels,
            kernel_size,
            in_channels,
            cuda,
            gpu,
        )
        return self
