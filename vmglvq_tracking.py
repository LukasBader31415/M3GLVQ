import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from tqdm.auto import tqdm
import proto_dist_ml.rng as rng

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2019, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.0.1'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

_ERR_CUTOFF = 1E-5

class VMGLVQ(BaseEstimator, ClassifierMixin):
    """
    (Docstring unverändert)
    """

    # ALT -> NEU: optionale Tracking-Flags mit Defaults (backwards-kompatibel)
    def __init__(self, K, T=50, phi=None, *, track_path=False, track_vweights=True, track_metrics=False):
        self.K = K
        self.T = T
        if(phi is None):
            self.phi = lambda mus : mus
        else:
            self.phi = phi

        # --- Tracking (minimalinvasiv) ---
        self.track_path = bool(track_path)
        self.track_vweights = bool(track_vweights)
        self.track_metrics = bool(track_metrics)

        self._w_history = None
        self._v_history = None
        self._log = None
        self._snapshot_idx = 0
    # ---------------- Tracking-Helfer ----------------
    def _init_tracking(self):
        self._snapshot_idx = 0
        if self.track_path:
            self._w_history = []
        if self.track_vweights:
            self._v_history = []
        if self.track_metrics:
            self._log = {"loss": []}

    def _snapshot(self, *, w=None, v=None, loss=None):
        # nur das Nötigste, Algorithmus bleibt unberührt
        if self.track_path and w is not None:
            self._w_history.append(np.array(w, copy=True))
        if self.track_vweights and v is not None:
            self._v_history.append(np.array(v, dtype=float, copy=True))
        if self.track_metrics and loss is not None:
            # loss als float sichern
            if self._log is None:
                self._log = {"loss": []}
            self._log["loss"].append(float(loss))
        self._snapshot_idx += 1

    # ---------------- Fit (allgemein) ----------------
    def fit(self, DL, y):
        # Tracking zu Beginn vorbereiten
        self._init_tracking()

        if(self.K == 1):
            return self._fit_single(DL, y)

        self._V = len(DL)
        for v in range(self._V):
            if(len(DL[v].shape) != 2):
                raise ValueError('Input is not a matrix!')
            if(DL[v].shape[0] != DL[v].shape[1]):
                raise ValueError('Input matrix is not square!')
            self._m = DL[v].shape[0]
        self._vWeights=np.ones(self._V)
        self._vWeights=self._vWeights/sum(self._vWeights)

        unique_labels = np.unique(y)
        L = len(unique_labels)

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
                Dp_l = rng_l._Alpha.dot(D_l) + np.expand_dims(rng_l._z, 1)
                closest = np.argmin(Dp_l, axis=1)
                self._w[l*self.K:(l+1)*self.K] = inClass_l[closest]
            del inClass_l
            del D_l
            del rng_l

        # Initiale Loss-Berechnung (wie gehabt)
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
            inClass_w_l = np.where(unique_labels[l] == self._y)[0]
            Dp = overall_dist[inClass_l, :][:, self._w[inClass_w_l]]
            idxs = np.argpartition(Dp, 1, axis=1)
            closest_plus[inClass_l]    = inClass_w_l[idxs[:,0]]
            sndclosest_plus[inClass_l] = inClass_w_l[idxs[:,1]]
            outClass_w_l = np.where(unique_labels[l] != self._y)[0]
            Dm = overall_dist[inClass_l, :][:, self._w[outClass_w_l]]
            idxs = np.argpartition(Dm, 1, axis=1)
            closest_minus[inClass_l] = outClass_w_l[idxs[:, 0]]
            sndclosest_minus[inClass_l] = outClass_w_l[idxs[:, 1]]
            dp = overall_dist[np.arange(self._m), self._w[closest_plus]]
            dm = overall_dist[np.arange(self._m), self._w[closest_minus]]
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

        # --------- Tracking: initialer Snapshot nach erster Loss-Berechnung
        self._snapshot(w=self._w, v=self._vWeights, loss=self._loss[-1])

        # start the actual optimization (unverändert, nur Snapshots ergänzt)
        proto_losses  = np.zeros(len(self._w))
        for t in range(self.T):
            overall_dist=np.zeros((self._m, self._m))
            for i in range(self._m):
                for j in range(self._m):
                    overall_dist[i, j] = np.dot(np.array([m[i, j] for m in DL]),self._vWeights)
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
                    still_closest = overall_dist[receptive_field_plus_k, i] <= overall_dist[receptive_field_plus_k, self._w[sndclosest_plus[receptive_field_plus_k]]]
                    changed_plus  = np.unique(np.concatenate([inClass_k[overall_dist[inClass_k, i] < dp[inClass_k]], receptive_field_plus_k[still_closest]]))
                    changed_plus2 = receptive_field_plus_k[np.logical_not(still_closest)]
                    still_closest = overall_dist[receptive_field_minus_k, i] <= overall_dist[receptive_field_minus_k, self._w[sndclosest_minus[receptive_field_minus_k]]]
                    changed_minus = np.unique(np.concatenate([outClass_k[overall_dist[outClass_k, i] < dm[outClass_k]], receptive_field_minus_k[still_closest]]))
                    changed_minus2 = receptive_field_minus_k[np.logical_not(still_closest)]

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

                    if(delta_loss < best_delta_loss):
                        best_delta_loss = delta_loss
                        best_i = i
                        best_changed_plus = changed_plus
                        best_changed_plus2 = changed_plus2
                        best_changed_minus = changed_minus
                        best_changed_minus2 = changed_minus2

                if(best_delta_loss < 0.):
                    self._w[k] = best_i
                    closest_plus[best_changed_plus] = k
                    dp[best_changed_plus] = overall_dist[best_changed_plus, best_i]
                    mus[best_changed_plus] = self.phi((dp[best_changed_plus] - dm[best_changed_plus]) / (dp[best_changed_plus] + dm[best_changed_plus] + 1E-5))
                    for v in range(self._V):
                        dp_V[v][best_changed_plus] = DL[v][best_changed_plus, best_i]
                    w_inClass = np.where(self._y == self._y[k])[0]
                    idxs = np.argpartition(overall_dist[best_changed_plus, :][:, self._w[w_inClass]], 1, axis=1)
                    sndclosest_plus[best_changed_plus] = w_inClass[idxs[:, 1]]

                    closest_plus[best_changed_plus2] = sndclosest_plus[best_changed_plus2]
                    dp[best_changed_plus2] = overall_dist[best_changed_plus2, self._w[closest_plus[best_changed_plus2]]]
                    mus[best_changed_plus2] = self.phi((dp[best_changed_plus2] - dm[best_changed_plus2]) / (dp[best_changed_plus2] + dm[best_changed_plus2] + 1E-5))
                    for v in range(self._V):
                        dp_V[v][best_changed_plus2] = DL[v][best_changed_plus2, self._w[closest_plus[best_changed_plus2]]]
                    idxs = np.argpartition(overall_dist[best_changed_plus2, :][:, self._w[w_inClass]], 1, axis=1)
                    sndclosest_plus[best_changed_plus2] = w_inClass[idxs[:, 1]]

                    closest_minus[best_changed_minus] = k
                    dm[best_changed_minus] = overall_dist[best_changed_minus, best_i]
                    mus[best_changed_minus] = self.phi((dp[best_changed_minus] - dm[best_changed_minus]) / (dp[best_changed_minus] + dm[best_changed_minus] + 1E-5))
                    for v in range(self._V):
                        dm_V[v][best_changed_minus] = DL[v][best_changed_minus, best_i]

                    for l in range(L):
                        inClass_l = best_changed_minus[np.where(y[best_changed_minus] == unique_labels[l])[0]]
                        w_outClass = np.where(self._y != unique_labels[l])[0]
                        idxs = np.argpartition(overall_dist[inClass_l, :][:, self._w[w_outClass]], 1, axis=1)
                        sndclosest_minus[inClass_l] = w_outClass[idxs[:, 1]]

                    closest_minus[best_changed_minus2] = sndclosest_minus[best_changed_minus2]
                    dm[best_changed_minus2] = overall_dist[best_changed_minus2, self._w[closest_minus[best_changed_minus2]]]
                    mus[best_changed_minus2] = self.phi((dp[best_changed_minus2] - dm[best_changed_minus2]) / (dp[best_changed_minus2] + dm[best_changed_minus2] + 1E-5))
                    for v in range(self._V):
                        dm_V[v][best_changed_minus2] = DL[v][best_changed_minus2, self._w[closest_minus[best_changed_minus2]]]
                    idxs = np.argpartition(overall_dist[best_changed_minus2, :][:, self._w[w_outClass]], 1, axis=1)
                    sndclosest_minus[best_changed_minus2] = w_outClass[idxs[:, 1]]
                    for l in range(L):
                        inClass_l = best_changed_minus2[np.where(y[best_changed_minus2] == unique_labels[l])[0]]
                        w_outClass = np.where(self._y != unique_labels[l])[0]
                        idxs = np.argpartition(overall_dist[inClass_l, :][:, self._w[w_outClass]], 1, axis=1)
                        sndclosest_minus[inClass_l] = w_outClass[idxs[:, 1]]

                    expected_new_loss = self._loss[-1] + best_delta_loss
                    actual_new_loss = np.sum(mus)
                    rel_err = abs(expected_new_loss - actual_new_loss) / (abs(self._loss[-1]) + 1e-12)
                    if rel_err > 0.05:
                        print(f"[Warnung] Loss-Abweichung: erwartet {expected_new_loss:.3f}, tatsächlich {actual_new_loss:.3f} (rel_err={rel_err:.3%})")

                    self._loss.append(actual_new_loss)

                    # --- Tracking: Snapshot nach JEDEM kombinierten Update ---
                    self._snapshot(w=self._w, v=self._vWeights, loss=self._loss[-1])

                    ###########################################
                    # hier Anpassung Gewichte rein (unverändert)
                    for v in range(self._V):
                        mu= (np.array(dp_V[v])*np.array(dm)*np.array(self._vWeights[v])-np.array(dm_V[v])*np.array(dp)*np.array(self._vWeights[v]))/((np.array(dp)-np.array(dm)+np.array(1E-5))**2)
                        delta_v=-np.sum(mu)
                        self._vWeights[v]+=delta_v
                    self._vWeights=self._vWeights/sum(self._vWeights)
                    ###########################################

                    # --- Tracking: Snapshot nach v-Update (damit man beide sieht)
                    self._snapshot(w=self._w, v=self._vWeights, loss=self._loss[-1])

                    break

            if(best_delta_loss >= -_ERR_CUTOFF):
                # letzter Zustand auch noch ablegen (nützlich fürs Plotten)
                self._snapshot(w=self._w, v=self._vWeights, loss=self._loss[-1])
                break

        return self

    def _fit_single(self, DL, y):
        # Tracking vorbereiten
        self._init_tracking()

        if(self.K > 1):
            raise ValueError('This method is only intended for training with a single prototype, but got K = %d' % self.K)
        self._V = len(DL)
        for v in range(len(DL)):
            if(len(DL[v].shape) != 2):
                raise ValueError('Input is not a matrix!')
            if(DL[v].shape[0] != DL[v].shape[1]):
                raise ValueError('Input matrix is not square!')
            self._m = DL[v].shape[0]
        self._vWeights=np.ones(len(DL))
        self._vWeights=self._vWeights/sum(self._vWeights)

        unique_labels = np.unique(y)
        L = len(unique_labels)
        if(L <= 2):
            return self._fit_single_binary(DL, y)

        D=np.zeros((self._m, self._m))
        for i in range(self._m):
            for j in range(self._m):
                D[i, j] = np.dot(np.array([m[i, j] for m in DL]),self._vWeights)
        if(not hasattr(self, 'prevent_initialization') or not self.prevent_initialization):
            self._w = np.zeros(self.K * L, dtype=int)
            self._y = np.zeros(self.K * L)
            for l in range(L):
                self._y[l] = unique_labels[l]
                inClass_l  = np.where(unique_labels[l] == y)[0]
                D_l = np.square(D[inClass_l, :][:, inClass_l])
                self._w[l] = inClass_l[np.argmin(np.sum(D_l, axis=0))]
            del inClass_l
            del D_l

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

        # Initialer Snapshot
        self._snapshot(w=self._w, v=self._vWeights, loss=self._loss[-1])

        proto_losses  = np.zeros(len(self._w))
        for t in range(self.T):
            D=np.zeros((self._m, self._m))
            for i in range(self._m):
                for j in range(self._m):
                    D[i, j] = np.dot(np.array([m[i, j] for m in DL]),self._vWeights)
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
                    dp[receptive_field_plus_k] = D[receptive_field_plus_k, best_i]
                    mus[receptive_field_plus_k] = self.phi((dp[receptive_field_plus_k] - dm[receptive_field_plus_k]) / (dp[receptive_field_plus_k] + dm[receptive_field_plus_k] + 1E-5))
                    for v in range(self._V):
                        dp_V[v][receptive_field_plus_k] = DL[v][receptive_field_plus_k, best_i]

                    closest_minus[best_changed_minus] = k
                    dm[best_changed_minus] = D[best_changed_minus, best_i]
                    mus[best_changed_minus] = self.phi((dp[best_changed_minus] - dm[best_changed_minus]) / (dp[best_changed_minus] + dm[best_changed_minus] + 1E-5))
                    for v in range(self._V):
                        dm_V[v][best_changed_minus] = DL[v][best_changed_minus, best_i]
                    for l in range(L):
                        inClass_l = best_changed_minus[np.where(y[best_changed_minus] == unique_labels[l])[0]]
                        w_outClass = np.where(self._y != unique_labels[l])[0]
                        idxs = np.argpartition(D[inClass_l, :][:, self._w[w_outClass]], 1, axis=1)
                        sndclosest_minus[inClass_l] = w_outClass[idxs[:, 1]]

                    closest_minus[best_changed_minus2] = sndclosest_minus[best_changed_minus2]
                    dm[best_changed_minus2] = D[best_changed_minus2, self._w[closest_minus[best_changed_minus2]]]
                    mus[best_changed_minus2] = self.phi((dp[best_changed_minus2] - dm[best_changed_minus2]) / (dp[best_changed_minus2] + dm[best_changed_minus2] + 1E-5))
                    for v in range(self._V):
                        dm_V[v][best_changed_minus2] = DL[v][best_changed_minus2, self._w[closest_minus[best_changed_minus2]]]
                    idxs = np.argpartition(D[best_changed_minus2, :][:, self._w[w_outClass]], 1, axis=1)
                    sndclosest_minus[best_changed_minus2] = w_outClass[idxs[:, 1]]
                    for l in range(L):
                        inClass_l = best_changed_minus2[np.where(y[best_changed_minus2] == unique_labels[l])[0]]
                        w_outClass = np.where(self._y != unique_labels[l])[0]
                        idxs = np.argpartition(D[inClass_l, :][:, self._w[w_outClass]], 1, axis=1)
                        sndclosest_minus[inClass_l] = w_outClass[idxs[:, 1]]

                    expected_new_loss = self._loss[-1] + best_delta_loss
                    actual_new_loss = np.sum(mus)
                    rel_err = abs(expected_new_loss - actual_new_loss) / (abs(self._loss[-1]) + 1e-12)
                    if rel_err > 0.05:
                        print(f"[Warnung] Loss-Abweichung: erwartet {expected_new_loss:.3f}, tatsächlich {actual_new_loss:.3f} (rel_err={rel_err:.3%})")

                    self._loss.append(actual_new_loss)

                    # Snapshot nach Prototypen-Update
                    self._snapshot(w=self._w, v=self._vWeights, loss=self._loss[-1])

                    ###########################################
                    # hier Anpassung Gewichte rein (unverändert)
                    for v in range(self._V):
                        mu= (np.array(dp_V[v])*np.array(dm)*np.array(self._vWeights[v])-np.array(dm_V[v])*np.array(dp)*np.array(self._vWeights[v]))/((np.array(dp)-np.array(dm)+np.array(1E-5))**2)
                        delta_v=-np.sum(mu)
                        self._vWeights[v]+=delta_v
                    self._vWeights=self._vWeights/sum(self._vWeights)
                    ###########################################

                    # Snapshot nach v-Update
                    self._snapshot(w=self._w, v=self._vWeights, loss=self._loss[-1])

                    break

            if(best_delta_loss >= -_ERR_CUTOFF):
                self._snapshot(w=self._w, v=self._vWeights, loss=self._loss[-1])
                break

        return self


    def _fit_single_binary(self, DL, y):
        # Tracking vorbereiten
        self._init_tracking()

        if(self.K > 1):
            raise ValueError('This method is only intended for training with a single prototype, but got K = %d' % self.K)
        for v in range(len(DL)):
            if(len(DL[v].shape) != 2):
                raise ValueError('Input is not a matrix!')
            if(DL[v].shape[0] != DL[v].shape[1]):
                raise ValueError('Input matrix is not square!')
            self._m = DL[v].shape[0]
        self._V=len(DL)
        self._vWeights=np.ones(len(DL))
        self._vWeights=self._vWeights/sum(self._vWeights)

        unique_labels = np.unique(y)
        L = len(unique_labels)
        if(L > 2):
            raise ValueError('This method is only intended for binary classification problems, but got %d labels' % L)
        D=np.zeros((self._m, self._m))
        for i in range(self._m):
            for j in range(self._m):
                D[i, j] = np.dot(np.array([m[i, j] for m in DL]),self._vWeights)
        if(not hasattr(self, 'prevent_initialization') or not self.prevent_initialization):
            self._w = np.zeros(self.K * L, dtype=int)
            self._y = np.zeros(self.K * L)
            for l in range(L):
                self._y[l] = unique_labels[l]
                inClass_l  = np.where(unique_labels[l] == y)[0]
                D_l = np.square(D[inClass_l, :][:, inClass_l])
                self._w[l] = inClass_l[np.argmin(np.sum(D_l, axis=0))]
            del inClass_l
            del D_l

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

        # Initialer Snapshot
        self._snapshot(w=self._w, v=self._vWeights, loss=self._loss[-1])

        proto_losses  = np.zeros(len(self._w))
        for t in range(self.T):
            D=np.zeros((self._m, self._m))
            for i in range(self._m):
                for j in range(self._m):
                    D[i, j] = np.dot(np.array([m[i, j] for m in DL]),self._vWeights)
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
                    dp[receptive_field_plus_k] = D[receptive_field_plus_k, best_i]
                    mus[receptive_field_plus_k] = self.phi((dp[receptive_field_plus_k] - dm[receptive_field_plus_k]) / (dp[receptive_field_plus_k] + dm[receptive_field_plus_k] + 1E-5))
                    for v in range(self._V):
                        dp_V[v][receptive_field_plus_k] = DL[v][receptive_field_plus_k, best_i]

                    dm[receptive_field_minus_k] = D[receptive_field_minus_k, best_i]
                    mus[receptive_field_minus_k] = self.phi((dp[receptive_field_minus_k] - dm[receptive_field_minus_k]) / (dp[receptive_field_minus_k] + dm[receptive_field_minus_k] + 1E-5))
                    for v in range(self._V):
                        dm_V[v][receptive_field_minus_k] = DL[v][receptive_field_minus_k, best_i]

                    expected_new_loss = self._loss[-1] + best_delta_loss
                    actual_new_loss = np.sum(mus)
                    rel_err = abs(expected_new_loss - actual_new_loss) / (abs(self._loss[-1]) + 1e-12)
                    if rel_err > 0.05:
                        print(f"[Warnung] Loss-Abweichung: erwartet {expected_new_loss:.3f}, tatsächlich {actual_new_loss:.3f} (rel_err={rel_err:.3%})")

                    self._loss.append(actual_new_loss)

                    # Snapshot nach Prototypen-Update
                    self._snapshot(w=self._w, v=self._vWeights, loss=self._loss[-1])

                    ###########################################
                    # hier Anpassung Gewichte rein (unverändert)
                    for v in range(self._V):
                        mu= (np.array(dp_V[v])*np.array(dm)*np.array(self._vWeights[v])-np.array(dm_V[v])*np.array(dp)*np.array(self._vWeights[v]))/((np.array(dp)-np.array(dm)+np.array(1E-5))**2)
                        delta_v=-np.sum(mu)
                        self._vWeights[v]+=delta_v
                    self._vWeights=self._vWeights/sum(self._vWeights)
                    ###########################################

                    # Snapshot nach v-Update
                    self._snapshot(w=self._w, v=self._vWeights, loss=self._loss[-1])

                    break

            if(best_delta_loss >= -_ERR_CUTOFF):
                self._snapshot(w=self._w, v=self._vWeights, loss=self._loss[-1])
                break

        return self

    # ---------------- Inferenz & Auslese ----------------
    def predict(self, DL):
        D=np.zeros((self._m, self._m))
        for i in range(self._m):
            for j in range(self._m):
                D[i, j] = np.dot(np.array([m[i, j] for m in DL]),self._vWeights)
        n = D.shape[0]
        if(D.shape[1] == self._m):
            D = D[:, self._w]
        closest = np.argmin(D, axis=1)
        return self._y[closest]

    def get_prototype_path(self):
        """(steps, K)-Array mit Prototyp-Indizes (oder None)."""
        if self._w_history is None:
            return None
        return np.vstack(self._w_history) if len(self._w_history) else np.empty((0,))

    def get_vweight_path(self):
        """(steps, V)-Array mit Distanz-Gewichten (oder None)."""
        if self._v_history is None:
            return None
        return np.vstack(self._v_history) if len(self._v_history) else np.empty((0,))

    def get_training_log(self):
        """dict mit Verlaufsdaten; aktuell nur 'loss' (minimalinvasiv)."""
        if self._log is None:
            return None
        return {"loss": np.array(self._log["loss"], dtype=float)}
