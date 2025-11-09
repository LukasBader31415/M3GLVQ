"""
Implements median generalized vector quantization, inspired by the paper

Nebel, D., Hammer, B., Frohberg, K., & Villmann, T. (2015). Median variants
of learning vector quantization for learning of dissimilarity data.
Neurocomputing, 169, 295-305. doi:10.1016/j.neucom.2014.12.096
"""

# Copyright (C) 2019
# Benjamin Paaßen
# AG Machine Learning
# Bielefeld University

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import proto_dist_ml.rng as rng

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2019, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.0.1'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

_ERR_CUTOFF = 1E-5

class MGLVQ(BaseEstimator, ClassifierMixin):
    """ A median generalized learning vector quantization model aims at
    identifying a subset of data points, called prototypes, which maximize
    the accuracy of a one-nearest neighbor classifier based on only this
    subset of data points. This makes such a model quite efficient because
    we only need the distances to the prototypes for classification.

    Attributes
    ----------
    K: int
        The number of prototypes per class to be learned.
    T: int (optional, default=50)
        The number of epochs in training.
    phi: function handle (optional, default=identity)
        A squashing function to post-process each error term. Defaults
        to the identity.
    track_path: bool (optional, default=False)
        If True, stores prototype index vectors over accepted updates in
        self._w_history.
    _m: int
        The number of training data points. This is not set by the user
        but during fit().
    _w: array_like
        A K*L dimensional index vector storing for each prototype the
        index in the training data. This is not set by the user but during
        fit().
    _y: array_like
        A K*L dimensional label vector. This is not set by the user but during
        fit().
    _loss: array_like
        The GLVQ loss during the last training run. This is not set by the
        user but during fit().

    """

    # NEU: track_path-Flag ergänzen
    def __init__(self, K, T=50, phi=None, track_path=False):
        self.K = K
        self.T = T
        self.track_path = track_path
        self.phi = (lambda mus: mus) if phi is None else phi

    # NEU: History-Init (nachdem self._w gesetzt ist aufrufen!)
    def _init_history(self):
        if getattr(self, "track_path", False):
            # Startzustand mitschneiden
            self._w_history = [self._w.copy()]

    def fit(self, D, y):
        """ Fits a median generalized learning vector quantization model to
        the given distance matrix.
        """
        if(self.K == 1):
            return self._fit_single(D, y)
        # check that D is a square matrix
        if(len(D.shape) != 2):
            raise ValueError('Input is not a matrix!')
        if(D.shape[0] != D.shape[1]):
            raise ValueError('Input matrix is not square!')
        self._m = D.shape[0]
        # set up the model arrays
        unique_labels = np.unique(y)
        L = len(unique_labels)

        # use class-wise relational neural gas to get a good initialization
        if(not hasattr(self, 'prevent_initialization') or not self.prevent_initialization):
            self._w = np.zeros(self.K * L, dtype=int)
            self._y = np.zeros(self.K * L)
            for l in range(L):
                self._y[l*self.K:(l+1)*self.K] = unique_labels[l]
                inClass_l  = np.where(unique_labels[l] == y)[0]
                D_l   = np.square(D[inClass_l, :][:, inClass_l])
                rng_l = rng.RNG(self.K)
                rng_l.fit(D_l, is_squared = True)
                # compute the prototype-to-data distances
                Dp_l = rng_l._Alpha.dot(D_l) + np.expand_dims(rng_l._z, 1)
                # get the closest data point to each prototype
                closest = np.argmin(Dp_l, axis=1)
                # store those as initial prototypes
                self._w[l*self.K:(l+1)*self.K] = inClass_l[closest]
            # History erst starten, wenn self._w vollständig gesetzt ist
            self._init_history()
        else:
            # Prototypen bereits vorhanden -> trotzdem History starten
            self._init_history()

        # then, compute the initial GLVQ loss. We also compute the closest
        # and second-closest prototype for each data point, because we need
        # these concepts during optimization
        self._loss = []
        closest_plus  = np.zeros(self._m, dtype=int)
        sndclosest_plus  = np.zeros(self._m, dtype=int)
        closest_minus = np.zeros(self._m, dtype=int)
        sndclosest_minus = np.zeros(self._m, dtype=int)
        for l in range(L):
            inClass_l  = np.where(unique_labels[l] == y)[0]
            # get the distances to prototypes with the same label
            inClass_w_l = np.where(unique_labels[l] == self._y)[0]
            Dp = D[inClass_l, :][:, self._w[inClass_w_l]]
            # get the closest and 2ndclosest prototype with the same label
            idxs = np.argpartition(Dp, 1, axis=1)
            closest_plus[inClass_l]    = inClass_w_l[idxs[:,0]]
            sndclosest_plus[inClass_l] = inClass_w_l[idxs[:,1]]
            # get the distances to prototypes with a different label
            outClass_w_l = np.where(unique_labels[l] != self._y)[0]
            Dm = D[inClass_l, :][:, self._w[outClass_w_l]]
            # get the closest and 2ndclosest prototype with a different label
            idxs = np.argpartition(Dm, 1, axis=1)
            closest_minus[inClass_l] = outClass_w_l[idxs[:, 0]]
            sndclosest_minus[inClass_l] = outClass_w_l[idxs[:, 1]]
        # for every data point, retrieve the distances to the positive and
        # negative prototype
        dp = D[np.arange(self._m), self._w[closest_plus]]
        dm = D[np.arange(self._m), self._w[closest_minus]]
        # and compute the loss terms
        mus  = self.phi((dp - dm) / (dp + dm + 1E-5))
        self._loss.append(np.sum(mus))

        # start the actual optimization
        proto_losses  = np.zeros(len(self._w))
        for t in range(self.T):
            # rank the prototypes according to their loss contributions
            for k in range(len(self._w)):
                proto_losses[k] = np.sum(dp[closest_plus == k]) - np.sum(dm[closest_minus == k])
            # iterate over all prototypes, from highest to lowest
            for k in np.argsort(-proto_losses):
                inClass_k  = np.where(y == self._y[k])[0]
                outClass_k = np.where(y != self._y[k])[0]
                receptive_field_plus_k  = np.where(closest_plus == k)[0]
                receptive_field_minus_k = np.where(closest_minus == k)[0]
                # iterate over all data points in ks receptive field
                best_delta_loss = 0.
                for i in receptive_field_plus_k:
                    if(i == self._w[k]):
                        continue
                    # If set change self._w[k] = i, four kinds of changes could occur
                    # First, we have to consider data points in the same class
                    still_closest = D[receptive_field_plus_k, i] <= D[receptive_field_plus_k, self._w[sndclosest_plus[receptive_field_plus_k]]]
                    changed_plus  = np.unique(np.concatenate([inClass_k[D[inClass_k, i] < dp[inClass_k]], receptive_field_plus_k[still_closest]]))
                    # Second, same class but now closer to another prototype
                    changed_plus2 = receptive_field_plus_k[np.logical_not(still_closest)]
                    # Third, points in a different class that now have i as closest negative
                    still_closest = D[receptive_field_minus_k, i] <= D[receptive_field_minus_k, self._w[sndclosest_minus[receptive_field_minus_k]]]
                    changed_minus = np.unique(np.concatenate([outClass_k[D[outClass_k, i] < dm[outClass_k]], receptive_field_minus_k[still_closest]]))
                    # Fourth, different class that had old k as closest negative but now another
                    changed_minus2 = receptive_field_minus_k[np.logical_not(still_closest)]

                    # compute how the loss for all these cases would change
                    delta_loss = 0.

                    dp_new = D[changed_plus, i]
                    mus_new = self.phi((dp_new - dm[changed_plus]) / (dp_new + dm[changed_plus] + 1E-5))
                    delta_loss += np.sum(mus_new - mus[changed_plus])

                    dp_new = D[changed_plus2, self._w[sndclosest_plus[changed_plus2]]]
                    mus_new = self.phi((dp_new - dm[changed_plus2]) / (dp_new + dm[changed_plus2] + 1E-5))
                    delta_loss += np.sum(mus_new - mus[changed_plus2])

                    dm_new = D[changed_minus, i]
                    mus_new = self.phi((dp[changed_minus] - dm_new) / (dp[changed_minus] + dm_new + 1E-5))
                    delta_loss += np.sum(mus_new - mus[changed_minus])

                    dm_new = D[changed_minus2, self._w[sndclosest_minus[changed_minus2]]]
                    mus_new = self.phi((dp[changed_minus2] - dm_new) / (dp[changed_minus2] + dm_new + 1E-5))
                    delta_loss += np.sum(mus_new - mus[changed_minus2])

                    # check if we have obtained a new best option
                    if(delta_loss < best_delta_loss):
                        best_delta_loss = delta_loss
                        best_i = i
                        best_changed_plus = changed_plus
                        best_changed_plus2 = changed_plus2
                        best_changed_minus = changed_minus
                        best_changed_minus2 = changed_minus2
                # if we have found an option that is better than nothing, take it
                if(best_delta_loss < 0.):
                    self._w[k] = best_i
                    # NEU: History fortschreiben
                    if getattr(self, "track_path", False):
                        self._w_history.append(self._w.copy())
                    # Update caches
                    closest_plus[best_changed_plus] = k
                    dp[best_changed_plus] = D[best_changed_plus, best_i]
                    mus[best_changed_plus] = self.phi((dp[best_changed_plus] - dm[best_changed_plus]) / (dp[best_changed_plus] + dm[best_changed_plus] + 1E-5))
                    # new second-closest positive
                    w_inClass = np.where(self._y == self._y[k])[0]
                    idxs = np.argpartition(D[best_changed_plus, :][:, self._w[w_inClass]], 1, axis=1)
                    sndclosest_plus[best_changed_plus] = w_inClass[idxs[:, 1]]

                    closest_plus[best_changed_plus2] = sndclosest_plus[best_changed_plus2]
                    dp[best_changed_plus2] = D[best_changed_plus2, self._w[closest_plus[best_changed_plus2]]]
                    mus[best_changed_plus2] = self.phi((dp[best_changed_plus2] - dm[best_changed_plus2]) / (dp[best_changed_plus2] + dm[best_changed_plus2] + 1E-5))
                    idxs = np.argpartition(D[best_changed_plus2, :][:, self._w[w_inClass]], 1, axis=1)
                    sndclosest_plus[best_changed_plus2] = w_inClass[idxs[:, 1]]

                    closest_minus[best_changed_minus] = k
                    dm[best_changed_minus] = D[best_changed_minus, best_i]
                    mus[best_changed_minus] = self.phi((dp[best_changed_minus] - dm[best_changed_minus]) / (dp[best_changed_minus] + dm[best_changed_minus] + 1E-5))
                    for l in range(L):
                        inClass_l = best_changed_minus[np.where(y[best_changed_minus] == unique_labels[l])[0]]
                        w_outClass = np.where(self._y != unique_labels[l])[0]
                        idxs = np.argpartition(D[inClass_l, :][:, self._w[w_outClass]], 1, axis=1)
                        sndclosest_minus[inClass_l] = w_outClass[idxs[:, 1]]

                    closest_minus[best_changed_minus2] = sndclosest_minus[best_changed_minus2]
                    dm[best_changed_minus2] = D[best_changed_minus2, self._w[closest_minus[best_changed_minus2]]]
                    mus[best_changed_minus2] = self.phi((dp[best_changed_minus2] - dm[best_changed_minus2]) / (dp[best_changed_minus2] + dm[best_changed_minus2] + 1E-5))
                    idxs = np.argpartition(D[best_changed_minus2, :][:, self._w[w_outClass]], 1, axis=1)
                    sndclosest_minus[best_changed_minus2] = w_outClass[idxs[:, 1]]
                    for l in range(L):
                        inClass_l = best_changed_minus2[np.where(y[best_changed_minus2] == unique_labels[l])[0]]
                        w_outClass = np.where(self._y != unique_labels[l])[0]
                        idxs = np.argpartition(D[inClass_l, :][:, self._w[w_outClass]], 1, axis=1)
                        sndclosest_minus[inClass_l] = w_outClass[idxs[:, 1]]

                    # update the loss
                    expected_new_loss = self._loss[-1] + best_delta_loss
                    actual_new_loss = np.sum(mus)
                    if(np.abs(expected_new_loss - actual_new_loss) / self._m > 0.01):
                        raise ValueError('Internal error: Loss changed other than expected! Expected %g but got %d!' % (expected_new_loss, actual_new_loss))
                    self._loss.append(actual_new_loss)
                    break
            # abort if improvement is tiny
            if(best_delta_loss >= -_ERR_CUTOFF):
                break

        return self

    def _fit_single(self, D, y):
        """ Dedicated version for single prototype per class, multi-class (>2) """
        if(self.K > 1):
            raise ValueError('This method is only intended for training with a single prototype, but got K = %d' % self.K)
        if(len(D.shape) != 2):
            raise ValueError('Input is not a matrix!')
        if(D.shape[0] != D.shape[1]):
            raise ValueError('Input matrix is not square!')
        self._m = D.shape[0]
        unique_labels = np.unique(y)
        L = len(unique_labels)
        if(L <= 2):
            return self._fit_single_binary(D, y)

        if(not hasattr(self, 'prevent_initialization') or not self.prevent_initialization):
            self._w = np.zeros(self.K * L, dtype=int)
            self._y = np.zeros(self.K * L)
            for l in range(L):
                self._y[l] = unique_labels[l]
                inClass_l  = np.where(unique_labels[l] == y)[0]
                D_l = np.square(D[inClass_l, :][:, inClass_l])
                self._w[l] = inClass_l[np.argmin(np.sum(D_l, axis=0))]
            self._init_history()
        else:
            self._init_history()

        # closest positive is fixed by the only prototype per class
        closest_plus = np.zeros(self._m, dtype=int)
        for l in range(L):
            if(self._y[l] != unique_labels[l]):
                raise ValueError('expected the %dth prototype to have label %s, but had %s' % (l, str(unique_labels[l]), str(self._y[l])))
            inClass_l  = np.where(unique_labels[l] == y)[0]
            closest_plus[inClass_l] = l

        self._loss = []
        closest_minus = np.zeros(self._m, dtype=int)
        sndclosest_minus = np.zeros(self._m, dtype=int)
        for l in range(L):
            inClass_l  = np.where(unique_labels[l] == y)[0]
            outClass_w_l = np.where(unique_labels[l] != self._y)[0]
            Dm = D[inClass_l, :][:, self._w[outClass_w_l]]
            idxs = np.argpartition(Dm, 1, axis=1)
            closest_minus[inClass_l] = outClass_w_l[idxs[:, 0]]
            sndclosest_minus[inClass_l] = outClass_w_l[idxs[:, 1]]

        dp = D[np.arange(self._m), self._w[closest_plus]]
        dm = D[np.arange(self._m), self._w[closest_minus]]
        mus  = self.phi((dp - dm) / (dp + dm + 1E-5))
        self._loss.append(np.sum(mus))

        proto_losses  = np.zeros(len(self._w))
        for t in range(self.T):
            for k in range(len(self._w)):
                proto_losses[k] = np.sum(dp[closest_plus == k]) - np.sum(dm[closest_minus == k])
            for k in np.argsort(-proto_losses):
                inClass_k  = np.where(y == self._y[k])[0]
                outClass_k = np.where(y != self._y[k])[0]
                receptive_field_plus_k  = np.where(closest_plus == k)[0]
                receptive_field_minus_k = np.where(closest_minus == k)[0]
                best_delta_loss = 0.
                for i in receptive_field_plus_k:
                    if(i == self._w[k]):
                        continue
                    still_closest = D[receptive_field_minus_k, i] <= D[receptive_field_minus_k, self._w[sndclosest_minus[receptive_field_minus_k]]]
                    changed_minus = np.unique(np.concatenate([outClass_k[D[outClass_k, i] < dm[outClass_k]], receptive_field_minus_k[still_closest]]))
                    changed_minus2 = receptive_field_minus_k[np.logical_not(still_closest)]

                    delta_loss = 0.
                    dp_new = D[receptive_field_plus_k, i]
                    mus_new = self.phi((dp_new - dm[receptive_field_plus_k]) / (dp_new + dm[receptive_field_plus_k] + 1E-5))
                    delta_loss += np.sum(mus_new - mus[receptive_field_plus_k])

                    dm_new = D[changed_minus, i]
                    mus_new = self.phi((dp[changed_minus] - dm_new) / (dp[changed_minus] + dm_new + 1E-5))
                    delta_loss += np.sum(mus_new - mus[changed_minus])

                    dm_new = D[changed_minus2, self._w[sndclosest_minus[changed_minus2]]]
                    mus_new = self.phi((dp[changed_minus2] - dm_new) / (dp[changed_minus2] + dm_new + 1E-5))
                    delta_loss += np.sum(mus_new - mus[changed_minus2])

                    if(delta_loss < best_delta_loss):
                        best_delta_loss = delta_loss
                        best_i = i
                        best_changed_minus = changed_minus
                        best_changed_minus2 = changed_minus2
                if(best_delta_loss < 0.):
                    self._w[k] = best_i
                    if getattr(self, "track_path", False):
                        self._w_history.append(self._w.copy())

                    dp[receptive_field_plus_k] = D[receptive_field_plus_k, best_i]
                    mus[receptive_field_plus_k] = self.phi((dp[receptive_field_plus_k] - dm[receptive_field_plus_k]) / (dp[receptive_field_plus_k] + dm[receptive_field_plus_k] + 1E-5))

                    closest_minus[best_changed_minus] = k
                    dm[best_changed_minus] = D[best_changed_minus, best_i]
                    mus[best_changed_minus] = self.phi((dp[best_changed_minus] - dm[best_changed_minus]) / (dp[best_changed_minus] + dm[best_changed_minus] + 1E-5))
                    for l in range(L):
                        inClass_l = best_changed_minus[np.where(y[best_changed_minus] == unique_labels[l])[0]]
                        w_outClass = np.where(self._y != unique_labels[l])[0]
                        idxs = np.argpartition(D[inClass_l, :][:, self._w[w_outClass]], 1, axis=1)
                        sndclosest_minus[inClass_l] = w_outClass[idxs[:, 1]]

                    closest_minus[best_changed_minus2] = sndclosest_minus[best_changed_minus2]
                    dm[best_changed_minus2] = D[best_changed_minus2, self._w[closest_minus[best_changed_minus2]]]
                    mus[best_changed_minus2] = self.phi((dp[best_changed_minus2] - dm[best_changed_minus2]) / (dp[best_changed_minus2] + dm[best_changed_minus2] + 1E-5))
                    idxs = np.argpartition(D[best_changed_minus2, :][:, self._w[w_outClass]], 1, axis=1)
                    sndclosest_minus[best_changed_minus2] = w_outClass[idxs[:, 1]]
                    for l in range(L):
                        inClass_l = best_changed_minus2[np.where(y[best_changed_minus2] == unique_labels[l])[0]]
                        w_outClass = np.where(self._y != unique_labels[l])[0]
                        idxs = np.argpartition(D[inClass_l, :][:, self._w[w_outClass]], 1, axis=1)
                        sndclosest_minus[inClass_l] = w_outClass[idxs[:, 1]]

                    expected_new_loss = self._loss[-1] + best_delta_loss
                    actual_new_loss = np.sum(mus)
                    if(np.abs(expected_new_loss - actual_new_loss) / self._m > 0.01):
                        raise ValueError('Internal error: Loss changed other than expected! Expected %g but got %d!' % (expected_new_loss, actual_new_loss))
                    self._loss.append(actual_new_loss)
                    break
            if(best_delta_loss >= -_ERR_CUTOFF):
                break

        return self

    def _fit_single_binary(self, D, y):
        """ Dedicated version for binary classification with a single prototype per class """
        if(self.K > 1):
            raise ValueError('This method is only intended for training with a single prototype, but got K = %d' % self.K)
        if(len(D.shape) != 2):
            raise ValueError('Input is not a matrix!')
        if(D.shape[0] != D.shape[1]):
            raise ValueError('Input matrix is not square!')
        self._m = D.shape[0]
        unique_labels = np.unique(y)
        L = len(unique_labels)
        if(L > 2):
            raise ValueError('This method is only intended for binary classification problems, but got %d labels' % L)

        if(not hasattr(self, 'prevent_initialization') or not self.prevent_initialization):
            self._w = np.zeros(self.K * L, dtype=int)
            self._y = np.zeros(self.K * L)
            for l in range(L):
                self._y[l] = unique_labels[l]
                inClass_l  = np.where(unique_labels[l] == y)[0]
                D_l = np.square(D[inClass_l, :][:, inClass_l])
                self._w[l] = inClass_l[np.argmin(np.sum(D_l, axis=0))]
            self._init_history()
        else:
            self._init_history()

        closest_plus = np.zeros(self._m, dtype=int)
        closest_minus = np.zeros(self._m, dtype=int)
        for l in range(L):
            if(self._y[l] != unique_labels[l]):
                raise ValueError('expected the %dth prototype to have label %s, but had %s' % (l, str(unique_labels[l]), str(self._y[l])))
            inClass_l  = np.where(unique_labels[l] == y)[0]
            closest_plus[inClass_l] = l
            closest_minus[inClass_l] = 1-l

        self._loss = []
        dp = D[np.arange(self._m), self._w[closest_plus]]
        dm = D[np.arange(self._m), self._w[closest_minus]]
        mus  = self.phi((dp - dm) / (dp + dm + 1E-5))
        self._loss.append(np.sum(mus))

        proto_losses  = np.zeros(len(self._w))
        for t in range(self.T):
            for k in range(len(self._w)):
                proto_losses[k] = np.sum(dp[closest_plus == k]) - np.sum(dm[closest_minus == k])
            for k in np.argsort(-proto_losses):
                inClass_k  = np.where(y == self._y[k])[0]
                outClass_k = np.where(y != self._y[k])[0]
                receptive_field_plus_k  = np.where(closest_plus == k)[0]
                receptive_field_minus_k = np.where(closest_minus == k)[0]
                best_delta_loss = 0.
                for i in receptive_field_plus_k:
                    if(i == self._w[k]):
                        continue
                    delta_loss = 0.

                    dp_new = D[receptive_field_plus_k, i]
                    mus_new = self.phi((dp_new - dm[receptive_field_plus_k]) / (dp_new + dm[receptive_field_plus_k] + 1E-5))
                    delta_loss += np.sum(mus_new - mus[receptive_field_plus_k])

                    dm_new = D[receptive_field_minus_k, i]
                    mus_new = self.phi((dp[receptive_field_minus_k] - dm_new) / (dp[receptive_field_minus_k] + dm_new + 1E-5))
                    delta_loss += np.sum(mus_new - mus[receptive_field_minus_k])

                    if(delta_loss < best_delta_loss):
                        best_delta_loss = delta_loss
                        best_i = i

                if(best_delta_loss < 0.):
                    self._w[k] = best_i
                    if getattr(self, "track_path", False):
                        self._w_history.append(self._w.copy())

                    dp[receptive_field_plus_k] = D[receptive_field_plus_k, best_i]
                    mus[receptive_field_plus_k] = self.phi((dp[receptive_field_plus_k] - dm[receptive_field_plus_k]) / (dp[receptive_field_plus_k] + dm[receptive_field_plus_k] + 1E-5))
                    dm[receptive_field_minus_k] = D[receptive_field_minus_k, best_i]
                    mus[receptive_field_minus_k] = self.phi((dp[receptive_field_minus_k] - dm[receptive_field_minus_k]) / (dp[receptive_field_minus_k] + dm[receptive_field_minus_k] + 1E-5))

                    expected_new_loss = self._loss[-1] + best_delta_loss
                    actual_new_loss = np.sum(mus)
                    if(np.abs(expected_new_loss - actual_new_loss) / self._m > 0.01):
                        raise ValueError('Internal error: Loss changed other than expected! Expected %g but got %d!' % (expected_new_loss, actual_new_loss))
                    self._loss.append(actual_new_loss)
                    break
            if(best_delta_loss >= -_ERR_CUTOFF):
                break

        return self

    def predict(self, D):
        """ Predicts the label for the given test data, represented by their
        distances either to all training data points or only to the prototypes.
        """
        # check the dimensionality
        n = D.shape[0]
        if(D.shape[1] == self._m):
            D = D[:, self._w]
        # find the closest prototype for each data point
        closest = np.argmin(D, axis=1)
        # return the labels of these prototypes
        return self._y[closest]
