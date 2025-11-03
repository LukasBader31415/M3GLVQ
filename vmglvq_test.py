#!/usr/bin/python3
"""
Tests the median generalized learning vector quantization implementation
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

import unittest
import numpy as np
from scipy.spatial.distance import cdist
from vmglvq import VMGLVQ

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2019, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.0.0'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

class TestVMGLVQ(unittest.TestCase):
    # def test_vmgvlq1(self):
    #     # create the simplest possible test case of two Gaussian clusters
    #     # with slight overlap and binary classification

    #     K = 2
    #     m = 100
    #     X = np.zeros((K*m, 2))
    #     y = np.zeros(K*m)
    #     W_expected = []
    #     y_W = []
    #     for k in range(K):
    #         # get the location of the mean
    #         theta = k * 2 * np.pi / K
    #         mu = 1. * np.array([np.cos(theta), np.sin(theta)])
    #         W_expected.append(mu)
    #         y_W.append(k)
    #         # generate data
    #         X[k*m:(k+1)*m, :] = np.random.randn(m, 2) + mu
    #         y[k*m:(k+1)*m] = k
    #     W_expected = np.stack(W_expected, axis=0)
    #     y_W = np.stack(y_W, axis=0)
    #     # compute all pairwise Euclidean distances
    #     d=cdist(X, X)
    #     D = [d, d]
    #     # set up a mglvq model
    #     model = VMGLVQ(1)
    #     # train it
    #     model.fit(D, y)
    #     # check the result
    #     W_actual = X[model._w, :]
    #     np.testing.assert_allclose(W_actual, W_expected, atol=1.)
    #     # ensure high classification accuracy
    #     self.assertTrue(model.score(D, y) > 0.8)

    # def test_mgvlq2(self):
    #     # create a rather simple test data set of K Gaussian clusters in a circle,
    #     # with a new label for each of them
    #     K = 6
    #     m = 100
    #     X = np.zeros((K*m, 2))
    #     y = np.zeros(K*m)
    #     W_expected = []
    #     y_W = []
    #     for k in range(K):
    #         # get the location of the mean
    #         theta = k * 2 * np.pi / K
    #         mu = 3. * np.array([np.cos(theta), np.sin(theta)])
    #         W_expected.append(mu)
    #         y_W.append(k)
    #         # generate data
    #         X[k*m:(k+1)*m, :] = np.random.randn(m, 2) + mu
    #         y[k*m:(k+1)*m] = k
    #     W_expected = np.stack(W_expected, axis=0)
    #     y_W = np.stack(y_W, axis=0)
    #     # compute all pairwise Euclidean distances
    #     D = [cdist(X, X)]
    #     # set up a mglvq model
    #     model = VMGLVQ(1)
    #     # train it
    #     model.fit(D, y)
    #     # check the result
    #     W_actual = X[model._w, :]
    #     np.testing.assert_allclose(W_actual, W_expected, atol=1.)
    #     # ensure high classification accuracy
    #     self.assertTrue(model.score(D, y) > 0.8)

    # def test_mgvlq3(self):
    #     # create a rather simple test data set of K Gaussian clusters in a circle,
    #     # with interleaved labels
    #     K = 4
    #     m = 5
    #     X = np.zeros((K*m, 2))
    #     y = np.zeros(K*m)
    #     W_expected = []
    #     y_W = []
    #     for k in range(K):
    #         # get the location of the mean
    #         theta = k * 2 * np.pi / K
    #         mu = 3. * np.array([np.cos(theta), np.sin(theta)])
    #         W_expected.append(mu)
    #         y_W.append(k % 2)
    #         # generate data
    #         X[k*m:(k+1)*m, :] = np.random.randn(m, 2) + mu
    #         y[k*m:(k+1)*m] = k % 2
    #     W_expected = np.stack(W_expected, axis=0)
    #     y_W = np.stack(y_W, axis=0)
    #     # compute all pairwise Euclidean distances
    #     d=cdist(X, X)
    #     D = [d, d]
    #     # set up a mglvq model
    #     model = VMGLVQ(int(K / 2))
    #     # train it
    #     model.fit(D, y)
    #     # check the result
    #     W_actual = X[model._w, :]
    #     for l in range(2):
    #         # consider the prototypes for each label and ensure
    #         # that for each actual prototype there is at least one
    #         # very close expected prototype
    #         D_l = cdist(W_actual[model._y == l, :], W_expected[y_W == l, :])
    #         np.testing.assert_allclose(np.min(D_l, axis=1), np.zeros(len(D_l)), atol=1.)
    #     # ensure high classification accuracy
    #     self.assertTrue(model.score(D, y) > 0.8)

    def test_mgvlq4(self):
        # create a data set with four clusters placed in a square,
        # three of which belong to one class, and one of which belongs
        # to the other class
        m = 100
        X = np.zeros((4*m, 2))
        y = np.zeros(4*m)
        for k in range(4):
            # get the location of the mean
            theta = (k + 0.5) * 2 * np.pi / 4
            mu = 3. * np.array([np.cos(theta), np.sin(theta)])
            # generate data
            X[k*m:(k+1)*m, :] = np.random.randn(m, 2) + mu
            y[k*m:(k+1)*m] = 0 if k == 0 else 1
        # compute all pairwise Euclidean distances
        D = [cdist(X, X)]
        # set up a mglvq model
        model = VMGLVQ(2)
        # train it
        model.fit(D, y)
        # check the result
        self.assertTrue(len(model._loss) > 1)
        W_actual = X[model._w, :]
        # ensure high classification accuracy
        self.assertTrue(model.score(D, y) > 0.8)
        
    def test_vmglvq(self):
        # Basisvektoren
        m1 = np.array([1.0, 1.0, 1.0])
        m2 = np.array([2.0, 1.5, 1.1])
        m3 = np.array([3.0, 2.0, 1.2])

        # Varianten von m1
        m11 = np.array([1.01, 1.01, 1.01])
        m12 = np.array([0.99, 0.98, 0.99])
        m13 = np.array([1.01, 0.99, 1.01])
        m14 = np.array([1.01, 1.01, 0.99])
        m15 = np.array([0.99, 1.01, 0.99])
        m16 = np.array([0.99, 1.01, 1.01])
        m17 = np.array([1.01, 0.99, 0.99])
        m18 = np.array([0.99, 0.99, 1.01])

        # Varianten von m2
        m21 = np.array([2.01, 1.51, 1.11])
        m22 = np.array([2.01, 1.51, 1.09])
        m23 = np.array([2.01, 1.49, 1.11])
        m24 = np.array([2.01, 1.49, 1.09])
        m25 = np.array([1.99, 1.51, 1.11])
        m26 = np.array([1.99, 1.51, 1.09])
        m27 = np.array([1.99, 1.49, 1.11])
        m28 = np.array([1.99, 1.49, 1.09])

        # Varianten von m3
        m31 = np.array([3.01, 2.01, 1.21])
        m32 = np.array([3.01, 2.01, 1.19])
        m33 = np.array([3.01, 1.99, 1.21])
        m34 = np.array([3.01, 1.99, 1.19])
        m35 = np.array([2.99, 2.01, 1.21])
        m36 = np.array([2.99, 2.01, 1.19])
        m37 = np.array([2.99, 1.99, 1.21])
        m38 = np.array([2.99, 1.99, 1.19])
        all_vectors = [
            m11, m12, m13, m14, m15, m16, m17, m18,
            m21, m22, m23, m24, m25, m26, m27, m28
            #m3, m31, m32, m33, m34, m35, m36, m37, m38
        ]

        all_vectors = np.array(all_vectors)

        # --- Euklidische Distanzmatrix ---
        n = len(all_vectors)
        distances = []

        y_W = [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]#,2,2,2,2,2,2,2,2,2]

        for d in range(3):
            dist_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    dist_matrix[i, j] = np.abs(all_vectors[i][d] - all_vectors[j][d])
            distances.append(dist_matrix)
        
        # set up a mglvq model
        model = VMGLVQ(1)
        # train it
        model.fit(distances, np.array(y_W))
        print(model._vWeights)
        print(model._w)
        pass


if __name__ == '__main__':
    unittest.main()
