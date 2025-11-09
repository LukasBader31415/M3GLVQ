# mglvq_uneven.py
"""
Implements median generalized vector quantization with **per-label prototype counts**.

Based on:

Nebel, D., Hammer, B., Frohberg, K., & Villmann, T. (2015). Median variants
of learning vector quantization for learning of dissimilarity data.
Neurocomputing, 169, 295-305. doi:10.1016/j.neucom.2014.12.096
"""

from __future__ import annotations

import warnings
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import proto_dist_ml.rng as rng
from collections.abc import Sequence

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2019, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.1.2-perlabelK'
__maintainer__ = 'Benjamin Paaßen / modified'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

_ERR_CUTOFF = 1E-5


class MGLVQ(BaseEstimator, ClassifierMixin):
    """Median generalized learning vector quantization with *per-label* K.

    K can be:
        - int: same number of prototypes for every label
        - dict: {label: K_label}
        - array-like of length L (aligned to np.unique(y) order)

    Parameters
    ----------
    K : int | dict | sequence
        Prototype configuration.
    T : int
        Number of optimization iterations.
    phi : callable | None
        Transfer on mu-values. Defaults to identity.
    track_path : bool
        If True, store prototype indices over time.
    loss_mismatch_tol : float
        **Relative** Toleranzschwelle (Standard: 0.05 = 5%) für |expected-actual|/m.
    loss_mismatch_warn : bool
        Wenn True, nur Warnung; wenn False, Exception.
    """

    def __init__(self, K, T=50, phi=None, track_path=False,
                 loss_mismatch_tol: float = 0.05,
                 loss_mismatch_warn: bool = True):
        self.K = K
        self.T = T
        self.track_path = track_path
        self.phi = (lambda mus: mus) if phi is None else phi
        self.loss_mismatch_tol = float(loss_mismatch_tol)
        self.loss_mismatch_warn = bool(loss_mismatch_warn)

    # History init (call after self._w is set!)
    def _init_history(self):
        if getattr(self, "track_path", False):
            self._w_history = [self._w.copy()]

    # ---- K resolution helpers ------------------------------------------------
    def _resolve_K(self, y):
        labels = np.unique(y)
        L = len(labels)

        # Normalize K into per-label array matching labels order
        if isinstance(self.K, int):
            K_per_label = np.full(L, int(self.K), dtype=int)
        elif isinstance(self.K, dict):
            K_per_label = np.empty(L, dtype=int)
            for i, lab in enumerate(labels):
                if lab not in self.K:
                    raise ValueError(f"K dict missing entry for label {lab!r}")
                K_per_label[i] = int(self.K[lab])
        elif isinstance(self.K, Sequence):
            if len(self.K) != L:
                raise ValueError(f"K sequence length {len(self.K)} must equal number of labels {L}")
            K_per_label = np.asarray(self.K, dtype=int)
        else:
            raise TypeError("K must be int, dict, or sequence")

        if np.any(K_per_label <= 0):
            bad = K_per_label[K_per_label <= 0]
            raise ValueError(f"All K must be positive; got {bad}")

        self._labels = labels
        self._K_per_label = K_per_label
        self._sumK = int(np.sum(K_per_label))
        self._all_K_one = bool(np.all(K_per_label == 1))
        return labels, K_per_label

    # ---- Fit (general case; handles all K) ----------------------------------
    def fit(self, D, y):
        """Fit to a precomputed *square* distance matrix D (m x m) with labels y."""
        # Input checks
        if len(D.shape) != 2:
            raise ValueError('Input is not a matrix!')
        if D.shape[0] != D.shape[1]:
            raise ValueError('Input matrix is not square!')
        self._m = D.shape[0]

        # Resolve labels and per-label K
        unique_labels, K_per_label = self._resolve_K(y)

        # Optional optimized path when all K==1
        if self._all_K_one:
            return self._fit_allK1(D, y)

        # Initialization
        if (not hasattr(self, 'prevent_initialization')) or (not self.prevent_initialization):
            self._w = np.zeros(self._sumK, dtype=int)
            self._y = np.zeros(self._sumK, dtype=unique_labels.dtype)
            offset = 0
            for l, lab in enumerate(unique_labels):
                Kl = int(K_per_label[l])
                self._y[offset:offset+Kl] = lab
                inClass_l  = np.where(lab == y)[0]
                D_l   = np.square(D[inClass_l, :][:, inClass_l])
                rng_l = rng.RNG(Kl)
                rng_l.fit(D_l, is_squared=True)
                # prototype-to-data distances
                Dp_l = rng_l._Alpha.dot(D_l) + np.expand_dims(rng_l._z, 1)
                closest = np.argmin(Dp_l, axis=1)
                self._w[offset:offset+Kl] = inClass_l[closest]
                offset += Kl
            self._init_history()
        else:
            # assume self._w and self._y provided externally
            if len(self._w) != self._sumK:
                raise ValueError("prevent_initialization set but existing _w length does not match sum(K_per_label)")
            self._init_history()

        # Precompute closest/snd-closest pos/neg prototypes
        self._loss = []
        closest_plus  = np.zeros(self._m, dtype=int)
        sndclosest_plus  = np.zeros(self._m, dtype=int)
        closest_minus = np.zeros(self._m, dtype=int)
        sndclosest_minus = np.zeros(self._m, dtype=int)

        for l, lab in enumerate(unique_labels):
            inClass_l  = np.where(lab == y)[0]

            # positive set: prototypes with same label
            inClass_w_l = np.where(lab == self._y)[0]
            Dp = D[inClass_l, :][:, self._w[inClass_w_l]]
            if Dp.shape[1] > 1:
                idxs = np.argpartition(Dp, 1, axis=1)
                closest_plus[inClass_l]    = inClass_w_l[idxs[:, 0]]
                sndclosest_plus[inClass_l] = inClass_w_l[idxs[:, 1]]
            else:
                only_idx = inClass_w_l[0]
                closest_plus[inClass_l]    = only_idx
                sndclosest_plus[inClass_l] = only_idx

            # negative set: prototypes with different label
            outClass_w_l = np.where(lab != self._y)[0]
            Dm = D[inClass_l, :][:, self._w[outClass_w_l]]
            if Dm.shape[1] > 1:
                idxs = np.argpartition(Dm, 1, axis=1)
                closest_minus[inClass_l]    = outClass_w_l[idxs[:, 0]]
                sndclosest_minus[inClass_l] = outClass_w_l[idxs[:, 1]]
            else:
                only_idx = outClass_w_l[0]
                closest_minus[inClass_l]    = only_idx
                sndclosest_minus[inClass_l] = only_idx

        # Initial loss
        dp = D[np.arange(self._m), self._w[closest_plus]]
        dm = D[np.arange(self._m), self._w[closest_minus]]
        mus  = self.phi((dp - dm) / (dp + dm + 1E-5))
        self._loss.append(np.sum(mus))

        # Optimization
        proto_losses  = np.zeros(len(self._w))
        for t in range(self.T):
            for k in range(len(self._w)):
                proto_losses[k] = np.sum(dp[closest_plus == k]) - np.sum(dm[closest_minus == k])
            best_delta_loss = 0.0
            for k in np.argsort(-proto_losses):
                inClass_k  = np.where(y == self._y[k])[0]
                outClass_k = np.where(y != self._y[k])[0]
                receptive_field_plus_k  = np.where(closest_plus == k)[0]
                receptive_field_minus_k = np.where(closest_minus == k)[0]

                best_i = None
                best_changed_plus = best_changed_plus2 = None
                best_changed_minus = best_changed_minus2 = None
                local_best = 0.0

                for i in receptive_field_plus_k:
                    if i == self._w[k]:
                        continue
                    # same-label updates
                    still_closest = D[receptive_field_plus_k, i] <= D[receptive_field_plus_k, self._w[sndclosest_plus[receptive_field_plus_k]]]
                    changed_plus  = np.unique(np.concatenate([
                        inClass_k[D[inClass_k, i] < dp[inClass_k]],
                        receptive_field_plus_k[still_closest]
                    ]))
                    changed_plus2 = receptive_field_plus_k[np.logical_not(still_closest)]

                    # different-label updates
                    still_closest = D[receptive_field_minus_k, i] <= D[receptive_field_minus_k, self._w[sndclosest_minus[receptive_field_minus_k]]]
                    changed_minus = np.unique(np.concatenate([
                        outClass_k[D[outClass_k, i] < dm[outClass_k]],
                        receptive_field_minus_k[still_closest]
                    ]))
                    changed_minus2 = receptive_field_minus_k[np.logical_not(still_closest)]

                    # loss delta
                    delta_loss = 0.0
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

                    if delta_loss < local_best:
                        local_best = delta_loss
                        best_i = i
                        best_changed_plus = changed_plus
                        best_changed_plus2 = changed_plus2
                        best_changed_minus = changed_minus
                        best_changed_minus2 = changed_minus2

                if local_best < best_delta_loss:
                    best_delta_loss = local_best

                # apply update for k immediately when improving
                if local_best < 0.0 and best_i is not None:
                    self._w[k] = best_i
                    if getattr(self, "track_path", False):
                        self._w_history.append(self._w.copy())

                    # Update caches
                    closest_plus[best_changed_plus] = k
                    dp[best_changed_plus] = D[best_changed_plus, best_i]
                    mus[best_changed_plus] = self.phi((dp[best_changed_plus] - dm[best_changed_plus]) / (dp[best_changed_plus] + dm[best_changed_plus] + 1E-5))
                    # new second-closest positive among same-label prototypes
                    w_inClass = np.where(self._y == self._y[k])[0]
                    if len(w_inClass) > 1:
                        idxs = np.argpartition(D[best_changed_plus, :][:, self._w[w_inClass]], 1, axis=1)
                        sndclosest_plus[best_changed_plus] = w_inClass[idxs[:, 1]]
                    else:
                        sndclosest_plus[best_changed_plus] = w_inClass[0]

                    closest_plus[best_changed_plus2] = sndclosest_plus[best_changed_plus2]
                    dp[best_changed_plus2] = D[best_changed_plus2, self._w[closest_plus[best_changed_plus2]]]
                    mus[best_changed_plus2] = self.phi((dp[best_changed_plus2] - dm[best_changed_plus2]) / (dp[best_changed_plus2] + dm[best_changed_plus2] + 1E-5))
                    if len(w_inClass) > 1:
                        idxs = np.argpartition(D[best_changed_plus2, :][:, self._w[w_inClass]], 1, axis=1)
                        sndclosest_plus[best_changed_plus2] = w_inClass[idxs[:, 1]]
                    else:
                        sndclosest_plus[best_changed_plus2] = w_inClass[0]

                    closest_minus[best_changed_minus] = k
                    dm[best_changed_minus] = D[best_changed_minus, best_i]
                    mus[best_changed_minus] = self.phi((dp[best_changed_minus] - dm[best_changed_minus]) / (dp[best_changed_minus] + dm[best_changed_minus] + 1E-5))
                    for l, lab in enumerate(unique_labels):
                        inClass_l = best_changed_minus[np.where(y[best_changed_minus] == lab)[0]]
                        w_outClass = np.where(self._y != lab)[0]
                        if len(w_outClass) > 1 and len(inClass_l) > 0:
                            idxs = np.argpartition(D[inClass_l, :][:, self._w[w_outClass]], 1, axis=1)
                            sndclosest_minus[inClass_l] = w_outClass[idxs[:, 1]]
                        elif len(w_outClass) == 1 and len(inClass_l) > 0:
                            sndclosest_minus[inClass_l] = w_outClass[0]

                    closest_minus[best_changed_minus2] = sndclosest_minus[best_changed_minus2]
                    dm[best_changed_minus2] = D[best_changed_minus2, self._w[closest_minus[best_changed_minus2]]]
                    mus[best_changed_minus2] = self.phi((dp[best_changed_minus2] - dm[best_changed_minus2]) / (dp[best_changed_minus2] + dm[best_changed_minus2] + 1E-5))
                    # recompute sndclosest_minus for both class groups
                    for l, lab in enumerate(unique_labels):
                        inClass_l = best_changed_minus2[np.where(y[best_changed_minus2] == lab)[0]]
                        w_outClass = np.where(self._y != lab)[0]
                        if len(w_outClass) > 1 and len(inClass_l) > 0:
                            idxs = np.argpartition(D[inClass_l, :][:, self._w[w_outClass]], 1, axis=1)
                            sndclosest_minus[inClass_l] = w_outClass[idxs[:, 1]]
                        elif len(w_outClass) == 1 and len(inClass_l) > 0:
                            sndclosest_minus[inClass_l] = w_outClass[0]

                    expected_new_loss = self._loss[-1] + local_best
                    actual_new_loss = np.sum(mus)
                    rel_mismatch = abs(expected_new_loss - actual_new_loss) / self._m
                    if rel_mismatch > self.loss_mismatch_tol:
                        msg = (f"Loss changed other than expected (rel={rel_mismatch:.3%}). "
                               f"Expected {expected_new_loss:.6f} but got {actual_new_loss:.6f}!")
                        if self.loss_mismatch_warn:
                            warnings.warn("Internal warning: " + msg)
                        else:
                            raise ValueError("Internal error: " + msg)
                    self._loss.append(actual_new_loss)
                    break

            if best_delta_loss >= -_ERR_CUTOFF:
                break
        return self

    # ---- Optimized path when all K == 1 -------------------------------------
    def _fit_allK1(self, D, y):
        """Specialized training when every label has exactly one prototype."""
        if len(D.shape) != 2:
            raise ValueError('Input is not a matrix!')
        if D.shape[0] != D.shape[1]:
            raise ValueError('Input matrix is not square!')
        self._m = D.shape[0]
        unique_labels = np.unique(y)

        if (not hasattr(self, 'prevent_initialization')) or (not self.prevent_initialization):
            self._w = np.zeros(len(unique_labels), dtype=int)
            self._y = np.zeros(len(unique_labels), dtype=unique_labels.dtype)
            for l, lab in enumerate(unique_labels):
                self._y[l] = lab
                inClass_l  = np.where(lab == y)[0]
                D_l = np.square(D[inClass_l, :][:, inClass_l])
                self._w[l] = inClass_l[np.argmin(np.sum(D_l, axis=0))]
            self._init_history()
        else:
            self._init_history()

        # closest positive is fixed
        closest_plus = np.zeros(self._m, dtype=int)
        for l, lab in enumerate(unique_labels):
            if self._y[l] != lab:
                raise ValueError(f'expected the {l}th prototype to have label {lab}, but had {self._y[l]}')
            inClass_l  = np.where(lab == y)[0]
            closest_plus[inClass_l] = l

        self._loss = []
        closest_minus = np.zeros(self._m, dtype=int)
        sndclosest_minus = np.zeros(self._m, dtype=int)
        for l, lab in enumerate(unique_labels):
            inClass_l  = np.where(lab == y)[0]
            outClass_w_l = np.where(lab != self._y)[0]
            Dm = D[inClass_l, :][:, self._w[outClass_w_l]]
            if Dm.shape[1] > 1:
                idxs = np.argpartition(Dm, 1, axis=1)
                closest_minus[inClass_l]    = outClass_w_l[idxs[:, 0]]
                sndclosest_minus[inClass_l] = outClass_w_l[idxs[:, 1]]
            else:
                only_idx = outClass_w_l[0]
                closest_minus[inClass_l]    = only_idx
                sndclosest_minus[inClass_l] = only_idx

        dp = D[np.arange(self._m), self._w[closest_plus]]
        dm = D[np.arange(self._m), self._w[closest_minus]]
        mus  = self.phi((dp - dm) / (dp + dm + 1E-5))
        self._loss.append(np.sum(mus))

        proto_losses  = np.zeros(len(self._w))
        for t in range(self.T):
            for k in range(len(self._w)):
                proto_losses[k] = np.sum(dp[closest_plus == k]) - np.sum(dm[closest_minus == k])
            best_delta_loss = 0.0
            for k in np.argsort(-proto_losses):
                inClass_k  = np.where(y == self._y[k])[0]
                outClass_k = np.where(y != self._y[k])[0]
                receptive_field_plus_k  = np.where(closest_plus == k)[0]
                receptive_field_minus_k = np.where(closest_minus == k)[0]
                local_best = 0.0
                best_i = None
                for i in receptive_field_plus_k:
                    if i == self._w[k]:
                        continue
                    delta_loss = 0.0
                    dp_new = D[receptive_field_plus_k, i]
                    mus_new = self.phi((dp_new - dm[receptive_field_plus_k]) / (dp_new + dm[receptive_field_plus_k] + 1E-5))
                    delta_loss += np.sum(mus_new - mus[receptive_field_plus_k])
                    dm_new = D[receptive_field_minus_k, i]
                    mus_new = self.phi((dp[receptive_field_minus_k] - dm_new) / (dp[receptive_field_minus_k] + dm_new + 1E-5))
                    delta_loss += np.sum(mus_new - mus[receptive_field_minus_k])
                    if delta_loss < local_best:
                        local_best = delta_loss
                        best_i = i
                if local_best < 0.0 and best_i is not None:
                    self._w[k] = best_i
                    if getattr(self, "track_path", False):
                        self._w_history.append(self._w.copy())
                    dp[receptive_field_plus_k] = D[receptive_field_plus_k, best_i]
                    mus[receptive_field_plus_k] = self.phi((dp[receptive_field_plus_k] - dm[receptive_field_plus_k]) / (dp[receptive_field_plus_k] + dm[receptive_field_plus_k] + 1E-5))
                    dm[receptive_field_minus_k] = D[receptive_field_minus_k, best_i]
                    mus[receptive_field_minus_k] = self.phi((dp[receptive_field_minus_k] - dm[receptive_field_minus_k]) / (dp[receptive_field_minus_k] + dm[receptive_field_minus_k] + 1E-5))
                    expected_new_loss = self._loss[-1] + local_best
                    actual_new_loss = np.sum(mus)
                    rel_mismatch = abs(expected_new_loss - actual_new_loss) / self._m
                    if rel_mismatch > self.loss_mismatch_tol:
                        msg = (f"Loss changed other than expected (rel={rel_mismatch:.3%}). "
                               f"Expected {expected_new_loss:.6f} but got {actual_new_loss:.6f}!")
                        if self.loss_mismatch_warn:
                            warnings.warn("Internal warning: " + msg)
                        else:
                            raise ValueError("Internal error: " + msg)
                    self._loss.append(actual_new_loss)
                    break
            if best_delta_loss >= -_ERR_CUTOFF:
                break
        return self

    # ---- Predict -------------------------------------------------------------
    def predict(self, D):
        """Predict labels for a set of items given distances.

        If D has shape (n, m) with m==number of training points, it will be
        reduced to distances to prototypes. If D already has shape (n, sumK)
        with columns ordered as self._w, it is used directly.
        """
        n = D.shape[0]
        if D.shape[1] == self._m:
            D = D[:, self._w]
        closest = np.argmin(D, axis=1)
        return self._y[closest]
