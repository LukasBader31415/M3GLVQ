"""
Implements median generalized vector quantization for more then one distance matrix at a time, inspired by the paper

Nebel, D., Hammer, B., Frohberg, K., & Villmann, T. (2015). Median variants
of learning vector quantization for learning of dissimilarity data.
Neurocomputing, 169, 295-305. doi:10.1016/j.neucom.2014.12.096

and 

Learning vector quantization for heterogeneous structured data.
D Zühlke, FM Schleif, T Geweniger, S Haase, T Villmann
ESANN, 271-276
"""

# Copyright (C) 2019, 2025
# Benjamin Paaßen & Dietlind Zühlke
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

class VMGLVQ(BaseEstimator, ClassifierMixin):
    """ A median generalized learning vector quantization model aims at
    identifying a subset of data points, called prototypes, which maximize
    the accuracy of a one-nearest neighbor classifier based on only this
    subet of data points. This makes such a model quite efficient because
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
    _m: int
        The number of training data points. This is not set by the user
        but during fit().
    _w: array_like
        A K dimensional index vector storing for each prototype the
        index in the training data. This is not set by the user but during
        fit().
    _y: array_like
        A K dimensional label vector. This is not set by the user but during
        fit().
    _loss: array_like
        The GLVQ loss during the last training run. This is not set by the
        user but during fit().
    _V: int
        Number of distance matrices. This is not set by the user but during fit().
    _vWeights: array_like
        A V dimensional vector storing the weights of each distance matrix

    """
    def __init__(self, K, T = 50, phi = None):
        self.K = K
        self.T = T
        if(phi is None):
            self.phi = lambda mus : mus
        else:
            self.phi = phi

    def fit(self, DL, y):
        """ Fits a median generalized learning vector quantization model to
        the given set of distance matrices.

        In more detail, the training iterates the following steps:

        1. re-compute the vbGLVQ error term for each data point, which is
        given as self.phi[(d^j_i^+ - d^j_i^-) / (d^j_i^+ + d^j_i^-)], where
        d^j_i^+ is the distance to the closest prototype with the same label
        and d^j_i^- is the distance to the closest prototype with another
        label.
        2. rank the prototypes according to their contributions to the overall
        loss.
        3. Start with the highest ranking prototype, iterate over all data
        points to which it is the closest with the same label, and check
        whether we could improve the error by switching the prototype
        location to that data point.

        Note that step 3 can, at worst, take m² steps because for each
        potential new prototype location we need to check the change in error
        for each other data point. However, in practice step 3 will take much
        fewer computations because we can greedily accept the first change that
        improves our loss.

        Parameters
        ----------
        DL: List of array_like structures
            A list of m x m matrices of pairwise distances. Note that we have no
            preconditions for this matrices. It may even be asymmetric.
        y: array_like
            A m dimensional label vector for the data points.

        Returns
        -------
        class proto_dist_ml.vmglvq.VMGLVQ
            self

        """
            
        if(self.K == 1):
            return self._fit_single(DL, y)
        self._V = len(DL)
        for v in range(self._V):
            # check that D is a square matrix
            if(len(DL[v].shape) != 2):
                raise ValueError('Input is not a matrix!')
            if(DL[v].shape[0] != DL[v].shape[1]):
                raise ValueError('Input matrix is not square!')
            self._m = DL[v].shape[0]
        self._vWeights=np.ones(self._V)
        self._vWeights=self._vWeights/sum(self._vWeights)

        # set up the model arrays
        unique_labels = np.unique(y)
        L = len(unique_labels)
        
        # use class-wise relational neural gas to get a good initialization
        if(not hasattr(self, 'prevent_initialization') or not self.prevent_initialization):
            D=np.zeros((self._m, self._m))
            for i in range(self._m):
                for j in range(self._m):
                    D[i, j] = np.dot(np.array([m[i, j] for m in DL]),self._vWeights)
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
            del inClass_l
            del D_l
            del rng_l
        
        # then, compute the initial GLVQ loss. We also compute the closest
        # and second-closest prototype for each data point, because we need
        # these concepts during optimization
        self._loss = []
        closest_plus  = np.zeros(self._m, dtype=int)
        sndclosest_plus  = np.zeros(self._m, dtype=int)
        closest_minus = np.zeros(self._m, dtype=int)
        sndclosest_minus = np.zeros(self._m, dtype=int)
        for l in range(L):
            overall_dist=np.zeros((self._m, self._m))
            for i in range(self._m):
                for j in range(self._m):
                    overall_dist[i, j] = np.dot(np.array([m[i, j] for m in DL]),self._vWeights)
            inClass_l  = np.where(unique_labels[l] == y)[0]
            # get the distances to prototypes with the same label
            inClass_w_l = np.where(unique_labels[l] == self._y)[0]
            Dp = overall_dist[inClass_l, :][:, self._w[inClass_w_l]]
            # get the closest and 2ndclosest prototype with the same label
            idxs = np.argpartition(Dp, 1, axis=1)
            closest_plus[inClass_l]    = inClass_w_l[idxs[:,0]]
            sndclosest_plus[inClass_l] = inClass_w_l[idxs[:,1]]
            # get the distances to prototypes with a different label
            outClass_w_l = np.where(unique_labels[l] != self._y)[0]
            Dm = overall_dist[inClass_l, :][:, self._w[outClass_w_l]]
            # get the closest and 2ndclosest prototype with a different label
            idxs = np.argpartition(Dm, 1, axis=1)
            closest_minus[inClass_l] = outClass_w_l[idxs[:, 0]]
            sndclosest_minus[inClass_l] = outClass_w_l[idxs[:, 1]]
            # for every data point, retrieve the distances to the positive and
            # negative prototype
            dp = overall_dist[np.arange(self._m), self._w[closest_plus]]
            dm = overall_dist[np.arange(self._m), self._w[closest_minus]]
            mus  = self.phi((dp - dm) / (dp + dm + 1E-5))
            # and compute the loss terms
            self._loss.append(np.sum(mus))
            dp_V = []
            dm_V = []
            mu_V = []
            for v in range(self._V):
                dpv = DL[v][np.arange(self._m), self._w[closest_plus]]
                dp_V.append(dpv)
                dmv = DL[v][np.arange(self._m), self._w[closest_minus]]
                dm_V.append(dmv)
                mu= (np.array(dp_V[v])*np.array(dm)*np.array(self._vWeights[v])-np.array(dm_V[v])*np.array(dp)*np.array(self._vWeights[v]))/((np.array(dp)-np.array(dm)+np.array(1E-5))**2)
                mu_V.append(mu)

        # start the actual optimization
        proto_losses  = np.zeros(len(self._w))
        for t in range(self.T):
            overall_dist=np.zeros((self._m, self._m))
            for i in range(self._m):
                for j in range(self._m):
                    overall_dist[i, j] = np.dot(np.array([m[i, j] for m in DL]),self._vWeights)
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
                    # If set change self._w[k] = i, four kinds of changes could
                    # occur
                    # First, we have to consider data points in the same class
                    # that now have i as the closest positive prototype
                    still_closest = overall_dist[receptive_field_plus_k, i] <= overall_dist[receptive_field_plus_k, self._w[sndclosest_plus[receptive_field_plus_k]]]
                    changed_plus  = np.unique(np.concatenate([inClass_k[overall_dist[inClass_k, i] < dp[inClass_k]], receptive_field_plus_k[still_closest]]))
                    # Second, we have to consider data points in the same class
                    # that had self._w[k] as closest positive prototype before
                    # but are now closer to another prototype
                    changed_plus2 = receptive_field_plus_k[np.logical_not(still_closest)]
                    # Third, we have to consider data points in a different class
                    # that now have i as the closest negative prototype
                    still_closest = overall_dist[receptive_field_minus_k, i] <= overall_dist[receptive_field_minus_k, self._w[sndclosest_minus[receptive_field_minus_k]]]
                    changed_minus = np.unique(np.concatenate([outClass_k[overall_dist[outClass_k, i] < dm[outClass_k]], receptive_field_minus_k[still_closest]]))
                    # Fourth, we have to consider data points in a different class
                    # that had self._w[k] as closest negative prototype before
                    # but are now closer to another prototype
                    changed_minus2 = receptive_field_minus_k[np.logical_not(still_closest)]

                    # compute how the loss for all these cases would change
                    delta_loss = 0.

                    dp_new = overall_dist[changed_plus, i]
                    mus_new = self.phi((dp_new - dm[changed_plus]) / (dp_new + dm[changed_plus] + 1E-5))
                    delta_loss += np.sum(mus_new - mus[changed_plus])

                    dp_new = overall_dist[changed_plus2, self._w[sndclosest_plus[changed_plus2]]]
                    mus_new = self.phi((dp_new - dm[changed_plus2]) / (dp_new + dm[changed_plus2] + 1E-5))
                    delta_loss += np.sum(mus_new - mus[changed_plus2])

                    dm_new = overall_dist[changed_minus, i]
                    mus_new = self.phi((dp[changed_minus] - dm_new) / (dp[changed_minus] + dm_new + 1E-5))
                    delta_loss += np.sum(mus_new - mus[changed_minus])

                    dm_new = overall_dist[changed_minus2, self._w[sndclosest_minus[changed_minus2]]]
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
                # if we have found an option that is better than nothing, take
                # it greedily and break off the search
                if(best_delta_loss < 0.):
                    self._w[k] = best_i
                    # for all data points in changed_plus, i is the new closest
                    # positive prototype
                    closest_plus[best_changed_plus] = k
                    dp[best_changed_plus] = overall_dist[best_changed_plus, best_i]
                    mus[best_changed_plus] = self.phi((dp[best_changed_plus] - dm[best_changed_plus]) / (dp[best_changed_plus] + dm[best_changed_plus] + 1E-5))
                    for v in range(self._V):
                        dp_V[v][best_changed_plus] = DL[v][best_changed_plus, best_i]
                    # compute the new second-closest prototype
                    w_inClass = np.where(self._y == self._y[k])[0]
                    idxs = np.argpartition(overall_dist[best_changed_plus, :][:, self._w[w_inClass]], 1, axis=1)
                    sndclosest_plus[best_changed_plus] = w_inClass[idxs[:, 1]]
                    # for all data points in changed_plus2, the old second
                    # closest positive prototype is now the closest
                    closest_plus[best_changed_plus2] = sndclosest_plus[best_changed_plus2]
                    dp[best_changed_plus2] = overall_dist[best_changed_plus2, self._w[closest_plus[best_changed_plus2]]]
                    mus[best_changed_plus2] = self.phi((dp[best_changed_plus2] - dm[best_changed_plus2]) / (dp[best_changed_plus2] + dm[best_changed_plus2] + 1E-5))
                    for v in range(self._V):
                        dp_V[v][best_changed_plus2] = DL[v][best_changed_plus2, self._w[closest_plus[best_changed_plus2]]]
                    # compute the new second-closest prototype
                    idxs = np.argpartition(overall_dist[best_changed_plus2, :][:, self._w[w_inClass]], 1, axis=1)
                    sndclosest_plus[best_changed_plus2] = w_inClass[idxs[:, 1]]
                    # for all data points in changes_minus, i is the new
                    # closest negative prototype
                    closest_minus[best_changed_minus] = k
                    dm[best_changed_minus] = overall_dist[best_changed_minus, best_i]
                    mus[best_changed_minus] = self.phi((dp[best_changed_minus] - dm[best_changed_minus]) / (dp[best_changed_minus] + dm[best_changed_minus] + 1E-5))
                    for v in range(self._V):
                        dm_V[v][best_changed_minus] = DL[v][best_changed_minus, best_i]
                    # compute the new second-closest prototype
                    for l in range(L):
                        inClass_l = best_changed_minus[np.where(y[best_changed_minus] == unique_labels[l])[0]]
                        w_outClass = np.where(self._y != unique_labels[l])[0]
                        idxs = np.argpartition(overall_dist[inClass_l, :][:, self._w[w_outClass]], 1, axis=1)
                        sndclosest_minus[inClass_l] = w_outClass[idxs[:, 1]]
                    # for all data points in changes_minus2, the old second
                    # closest negative prototype is now the closest
                    closest_minus[best_changed_minus2] = sndclosest_minus[best_changed_minus2]
                    dm[best_changed_minus2] = overall_dist[best_changed_minus2, self._w[closest_minus[best_changed_minus2]]]
                    mus[best_changed_minus2] = self.phi((dp[best_changed_minus2] - dm[best_changed_minus2]) / (dp[best_changed_minus2] + dm[best_changed_minus2] + 1E-5))
                    for v in range(self._V):
                        dm_V[v][best_changed_minus2] = DL[v][best_changed_minus2, self._w[closest_minus[best_changed_minus2]]]
                    # compute the new second-closest prototype
                    idxs = np.argpartition(overall_dist[best_changed_minus2, :][:, self._w[w_outClass]], 1, axis=1)
                    sndclosest_minus[best_changed_minus2] = w_outClass[idxs[:, 1]]
                    for l in range(L):
                        inClass_l = best_changed_minus2[np.where(y[best_changed_minus2] == unique_labels[l])[0]]
                        w_outClass = np.where(self._y != unique_labels[l])[0]
                        idxs = np.argpartition(overall_dist[inClass_l, :][:, self._w[w_outClass]], 1, axis=1)
                        sndclosest_minus[inClass_l] = w_outClass[idxs[:, 1]]

                    # update the loss
                    expected_new_loss = self._loss[-1] + best_delta_loss
                    actual_new_loss = np.sum(mus)
                    if(np.abs(expected_new_loss - actual_new_loss) / self._m > 0.01):
                        raise ValueError('Internal error: Loss changed other than expected! Expected %g but got %d!' % (expected_new_loss, actual_new_loss))
                    self._loss.append(actual_new_loss)
                    ###########################################
                    # hier Anpassung Gewichte rein
                    
                    for v in range(self._V):
                        mu= (np.array(dp_V[v])*np.array(dm)*np.array(self._vWeights[v])-np.array(dm_V[v])*np.array(dp)*np.array(self._vWeights[v]))/((np.array(dp)-np.array(dm)+np.array(1E-5))**2)
                        delta_v=-np.sum(mu)
                        self._vWeights[v]+=delta_v
                    self._vWeights=self._vWeights/sum(self._vWeights)
                    
                    ###########################################
                    
                    break
            
            # if we could not improve the loss more than error threshold,
            # abort the optimization
            if(best_delta_loss >= -_ERR_CUTOFF):
                break

        return self

    def _fit_single(self, DL, y):
        """ A dedicated version of the optimization process in case we have
        only a single prototype per class, such that no ambiguity for
        the positive prototype exists.

        Parameters
        ----------
        DL: List of array_like structures
            A list of m x m matrices of pairwise distances. Note that we have no
            preconditions for this matrices. It may even be asymmetric.
        y: array_like
            A m dimensional label vector for the data points.

        Returns
        -------
        class proto_dist_ml.vmglvq.VMGLVQ
            self

        """
        if(self.K > 1):
            raise ValueError('This method is only intended for training with a single prototype, but got K = %d' % self.K)
        self._V = len(DL)
        for v in range(len(DL)):
            # check that D is a square matrix
            if(len(DL[v].shape) != 2):
                raise ValueError('Input is not a matrix!')
            if(DL[v].shape[0] != DL[v].shape[1]):
                raise ValueError('Input matrix is not square!')
            self._m = DL[v].shape[0]
        self._vWeights=np.ones(len(DL))
        self._vWeights=self._vWeights/sum(self._vWeights)
        # set up the model arrays
        unique_labels = np.unique(y)
        L = len(unique_labels)
        if(L <= 2):
            # if we have only a binary classification problem, there is
            # no ambiguity in prototype assignment at all and we can
            # use a very simple fit method
            return self._fit_single_binary(DL, y)
        D=np.zeros((self._m, self._m))
        for i in range(self._m):
            for j in range(self._m):
                D[i, j] = np.dot(np.array([m[i, j] for m in DL]),self._vWeights)            
        if(not hasattr(self, 'prevent_initialization') or not self.prevent_initialization):
            # initialize the prototypes as the data points which
            # are closest to the respective class means
            self._w = np.zeros(self.K * L, dtype=int)
            self._y = np.zeros(self.K * L)
            for l in range(L):
                self._y[l] = unique_labels[l]
                inClass_l  = np.where(unique_labels[l] == y)[0]
                D_l = np.square(D[inClass_l, :][:, inClass_l])
                self._w[l] = inClass_l[np.argmin(np.sum(D_l, axis=0))]
            del inClass_l
            del D_l
        # initialize the closest positive for each data point, which
        # necessarily is the only prototype within the class
        closest_plus = np.zeros(self._m, dtype=int)
        for l in range(L):
            if(self._y[l] != unique_labels[l]):
                raise ValueError('expected the %dth prototype to have label %s, but had %s' % (l, str(unique_labels[l]), str(self._y[l])))
            inClass_l  = np.where(unique_labels[l] == y)[0]
            closest_plus[inClass_l] = l

        # then, compute the initial GLVQ loss. We also compute the closest
        # and second-closest prototype for each data point, because we need
        # these concepts during optimization
        self._loss = []
        closest_minus = np.zeros(self._m, dtype=int)
        sndclosest_minus = np.zeros(self._m, dtype=int)
        for l in range(L):
            inClass_l  = np.where(unique_labels[l] == y)[0]
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
        
        dp_V = []
        dm_V = []
        mu_V = []
        for v in range(self._V):
            dpv = DL[v][np.arange(self._m), self._w[closest_plus]]
            dp_V.append(dpv)
            dmv = DL[v][np.arange(self._m), self._w[closest_minus]]
            dm_V.append(dmv)
            mu= (np.array(dp_V[v])*np.array(dm)*np.array(self._vWeights[v])-np.array(dm_V[v])*np.array(dp)*np.array(self._vWeights[v]))/((np.array(dp)-np.array(dm)+np.array(1E-5))**2)
            mu_V.append(mu)

        # start the actual optimization
        proto_losses  = np.zeros(len(self._w))
        for t in range(self.T):
            D=np.zeros((self._m, self._m))
            for i in range(self._m):
                for j in range(self._m):
                    D[i, j] = np.dot(np.array([m[i, j] for m in DL]),self._vWeights)            
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
                    # If set change self._w[k] = i, all data points in class
                    # self._y[k] have a different dp value. Further, it may
                    # occur that data points in a different class
                    # now have i as the closest negative prototype
                    still_closest = D[receptive_field_minus_k, i] <= D[receptive_field_minus_k, self._w[sndclosest_minus[receptive_field_minus_k]]]
                    changed_minus = np.unique(np.concatenate([outClass_k[D[outClass_k, i] < dm[outClass_k]], receptive_field_minus_k[still_closest]]))
                    # Second, we have to consider data points in a different class
                    # that had self._w[k] as closest negative prototype before
                    # but are now closer to another prototype
                    changed_minus2 = receptive_field_minus_k[np.logical_not(still_closest)]

                    # compute how the loss for all these cases would change
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

                    # check if we have obtained a new best option
                    if(delta_loss < best_delta_loss):
                        best_delta_loss = delta_loss
                        best_i = i
                        best_changed_minus = changed_minus
                        best_changed_minus2 = changed_minus2
                # if we have found an option that is better than nothing, take
                # it greedily and break off the search
                if(best_delta_loss < 0.):
                    self._w[k] = best_i
                    # for all data points in the receptive field of k, we need
                    # to update dp
                    dp[receptive_field_plus_k] = D[receptive_field_plus_k, best_i]
                    mus[receptive_field_plus_k] = self.phi((dp[receptive_field_plus_k] - dm[receptive_field_plus_k]) / (dp[receptive_field_plus_k] + dm[receptive_field_plus_k] + 1E-5))
                    for v in range(self._V):
                        dp_V[v][receptive_field_plus_k] = DL[v][receptive_field_plus_k, best_i]
                    # for all data points in changes_minus, i is the new
                    # closest negative prototype
                    closest_minus[best_changed_minus] = k
                    dm[best_changed_minus] = D[best_changed_minus, best_i]
                    mus[best_changed_minus] = self.phi((dp[best_changed_minus] - dm[best_changed_minus]) / (dp[best_changed_minus] + dm[best_changed_minus] + 1E-5))
                    for v in range(self._V):
                        dm_V[v][best_changed_minus] = DL[v][best_changed_minus, best_i]
                    # compute the new second-closest prototype
                    for l in range(L):
                        inClass_l = best_changed_minus[np.where(y[best_changed_minus] == unique_labels[l])[0]]
                        w_outClass = np.where(self._y != unique_labels[l])[0]
                        idxs = np.argpartition(D[inClass_l, :][:, self._w[w_outClass]], 1, axis=1)
                        sndclosest_minus[inClass_l] = w_outClass[idxs[:, 1]]
                    # for all data points in changes_minus2, the old second
                    # closest negative prototype is now the closest
                    closest_minus[best_changed_minus2] = sndclosest_minus[best_changed_minus2]
                    dm[best_changed_minus2] = D[best_changed_minus2, self._w[closest_minus[best_changed_minus2]]]
                    mus[best_changed_minus2] = self.phi((dp[best_changed_minus2] - dm[best_changed_minus2]) / (dp[best_changed_minus2] + dm[best_changed_minus2] + 1E-5))
                    for v in range(self._V):
                        dm_V[v][best_changed_minus2] = DL[v][best_changed_minus2, self._w[closest_minus[best_changed_minus2]]]
                    # compute the new second-closest prototype
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
                    ###########################################
                    # hier Anpassung Gewichte rein
                    
                    for v in range(self._V):
                        mu= (np.array(dp_V[v])*np.array(dm)*np.array(self._vWeights[v])-np.array(dm_V[v])*np.array(dp)*np.array(self._vWeights[v]))/((np.array(dp)-np.array(dm)+np.array(1E-5))**2)
                        delta_v=-np.sum(mu)
                        self._vWeights[v]+=delta_v
                    self._vWeights=self._vWeights/sum(self._vWeights)
                    ###########################################

                    break
            # if we could not improve the loss more than error threshold,
            # abort the optimization
            if(best_delta_loss >= -_ERR_CUTOFF):
                break

        return self


    def _fit_single_binary(self, DL, y):
        """ A dedicated version of the optimization process in case we have
        a binary classification problem with a single prototype per class,
        in which case the prototype assignment is always clear.

        Parameters
        ----------
        DL: List of array_like structures
            A list of m x m matrices of pairwise distances. Note that we have no
            preconditions for this matrices. It may even be asymmetric.
        y: array_like
            A m dimensional label vector for the data points.

        Returns
        -------
        class proto_dist_ml.vmglvq.VMGLVQ
            self

        """
        if(self.K > 1):
            raise ValueError('This method is only intended for training with a single prototype, but got K = %d' % self.K)
        # check that D is a square matrix
        for v in range(len(DL)):
            if(len(DL[v].shape) != 2):
                raise ValueError('Input is not a matrix!')
            if(DL[v].shape[0] != DL[v].shape[1]):
                raise ValueError('Input matrix is not square!')
            self._m = DL[v].shape[0]
        self._V=len(DL)
        self._vWeights=np.ones(len(DL))
        self._vWeights=self._vWeights/sum(self._vWeights)
        # set up the model arrays
        unique_labels = np.unique(y)
        L = len(unique_labels)
        if(L > 2):
            raise ValueError('This method is only intended for binary classification problems, but got %d labels' % L)
        D=np.zeros((self._m, self._m))
        for i in range(self._m):
            for j in range(self._m):
                D[i, j] = np.dot(np.array([m[i, j] for m in DL]),self._vWeights)            
        if(not hasattr(self, 'prevent_initialization') or not self.prevent_initialization):
            # initialize the prototypes as the data points which
            # are closest to the respective class means
            self._w = np.zeros(self.K * L, dtype=int)
            self._y = np.zeros(self.K * L)
            for l in range(L):
                self._y[l] = unique_labels[l]
                inClass_l  = np.where(unique_labels[l] == y)[0]
                D_l = np.square(D[inClass_l, :][:, inClass_l])
                self._w[l] = inClass_l[np.argmin(np.sum(D_l, axis=0))]
            del inClass_l
            del D_l

        # initialize the closest positives for each data point, which
        # necessarily are the only prototype within and without the class
        closest_plus = np.zeros(self._m, dtype=int)
        closest_minus = np.zeros(self._m, dtype=int)
        for l in range(L):
            if(self._y[l] != unique_labels[l]):
                raise ValueError('expected the %dth prototype to have label %s, but had %s' % (l, str(unique_labels[l]), str(self._y[l])))
            inClass_l  = np.where(unique_labels[l] == y)[0]
            closest_plus[inClass_l] = l
            closest_minus[inClass_l] = 1-l

        # then, compute the initial GLVQ loss.
        self._loss = []
        # for every data point, retrieve the distances to the positive and
        # negative prototype
        dp = D[np.arange(self._m), self._w[closest_plus]]
        dm = D[np.arange(self._m), self._w[closest_minus]]
        # and compute the loss terms
        mus  = self.phi((dp - dm) / (dp + dm + 1E-5))
        self._loss.append(np.sum(mus))

        dp_V = []
        dm_V = []
        mu_V = []
        for v in range(self._V):
            dpv = DL[v][np.arange(self._m), self._w[closest_plus]]
            dp_V.append(dpv)
            dmv = DL[v][np.arange(self._m), self._w[closest_minus]]
            dm_V.append(dmv)
            mu= (np.array(dp_V[v])*np.array(dm)*np.array(self._vWeights[v])-np.array(dm_V[v])*np.array(dp)*np.array(self._vWeights[v]))/((np.array(dp)-np.array(dm)+np.array(1E-5))**2)
            mu_V.append(mu)

        # start the actual optimization
        proto_losses  = np.zeros(len(self._w))
        for t in range(self.T):
            D=np.zeros((self._m, self._m))
            for i in range(self._m):
                for j in range(self._m):
                    D[i, j] = np.dot(np.array([m[i, j] for m in DL]),self._vWeights)            
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

                    # If set change self._w[k] = i, all data points in class
                    # self._y[k] have a different dp value and all points in
                    # the opposite class have a different dm value
                    delta_loss = 0.

                    dp_new = D[receptive_field_plus_k, i]
                    mus_new = self.phi((dp_new - dm[receptive_field_plus_k]) / (dp_new + dm[receptive_field_plus_k] + 1E-5))
                    delta_loss += np.sum(mus_new - mus[receptive_field_plus_k])

                    dm_new = D[receptive_field_minus_k, i]
                    mus_new = self.phi((dp[receptive_field_minus_k] - dm_new) / (dp[receptive_field_minus_k] + dm_new + 1E-5))
                    delta_loss += np.sum(mus_new - mus[receptive_field_minus_k])

                    # check if we have obtained a new best option
                    if(delta_loss < best_delta_loss):
                        best_delta_loss = delta_loss
                        best_i = i

                # if we have found an option that is better than nothing, take
                # it greedily and break off the search
                if(best_delta_loss < 0.):
                    self._w[k] = best_i
                    # update dp and dm for all points in the receptive field
                    dp[receptive_field_plus_k] = D[receptive_field_plus_k, best_i]
                    mus[receptive_field_plus_k] = self.phi((dp[receptive_field_plus_k] - dm[receptive_field_plus_k]) / (dp[receptive_field_plus_k] + dm[receptive_field_plus_k] + 1E-5))
                    for v in range(self._V):
                        dp_V[v][receptive_field_plus_k] = DL[v][receptive_field_plus_k, best_i]
 
                    dm[receptive_field_minus_k] = D[receptive_field_minus_k, best_i]
                    mus[receptive_field_minus_k] = self.phi((dp[receptive_field_minus_k] - dm[receptive_field_minus_k]) / (dp[receptive_field_minus_k] + dm[receptive_field_minus_k] + 1E-5))
                    for v in range(self._V):
                        dm_V[v][receptive_field_minus_k] = DL[v][receptive_field_minus_k, best_i]

                    # update the loss
                    expected_new_loss = self._loss[-1] + best_delta_loss
                    actual_new_loss = np.sum(mus)
                    if(np.abs(expected_new_loss - actual_new_loss) / self._m > 0.01):
                        raise ValueError('Internal error: Loss changed other than expected! Expected %g but got %d!' % (expected_new_loss, actual_new_loss))
                    self._loss.append(actual_new_loss)
                    ###########################################
                    # hier Anpassung Gewichte rein
                    
                    for v in range(self._V):
                        mu= (np.array(dp_V[v])*np.array(dm)*np.array(self._vWeights[v])-np.array(dm_V[v])*np.array(dp)*np.array(self._vWeights[v]))/((np.array(dp)-np.array(dm)+np.array(1E-5))**2)
                        delta_v=-np.sum(mu)
                        self._vWeights[v]+=delta_v
                    self._vWeights=self._vWeights/sum(self._vWeights)
                    ###########################################
                    break
            # if we could not improve the loss more than error threshold,
            # abort the optimization
            if(best_delta_loss >= -_ERR_CUTOFF):
                break

        return self

    def predict(self, DL):
        """ Predicts the label for the given test data, represented by their
        distances either to all training data points or only to the prototypes.

        Parameters
        ----------
        D: A n x m matrix of distances from the test to the training
        datapoints OR a n x K matrix of distances from the test datapoints
        to the prototypes

        Returns
        -------
        y: array_like
            a n-dimensional vector containing the predicted labels for each
            test datapoint.

        """
        D=np.zeros((self._m, self._m))
        for i in range(self._m):
            for j in range(self._m):
                D[i, j] = np.dot(np.array([m[i, j] for m in DL]),self._vWeights)            
        # check the dimensionality
        n = D.shape[0]
        if(D.shape[1] == self._m):
            D = D[:, self._w]
        # find the closest prototype for each data point
        closest = np.argmin(D, axis=1)
        # return the labels of these prototypes
        return self._y[closest]
