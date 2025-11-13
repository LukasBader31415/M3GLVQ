import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import proto_dist_ml.rng as rng

__author__ = 'Benjamin Paaßen'
__copyright__ = 'Copyright 2019, Benjamin Paaßen'
__license__ = 'GPLv3'
__version__ = '1.0.1'
__maintainer__ = 'Benjamin Paaßen'
__email__  = 'bpaassen@techfak.uni-bielefeld.de'

_ERR_CUTOFF = 1e-5


def _project_simplex(v):
    """Projiziere v auf das Simplex {w >= 0, sum(w)=1}. Duchi et al., 2008."""
    v = np.asarray(v, dtype=float)
    if v.ndim != 1:
        raise ValueError("Simplex projection expects 1D vector.")
    n = v.size
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    w = np.maximum(v - theta, 0.0)
    return w


class VMGLVQ(BaseEstimator, ClassifierMixin):
    """
    VMGLVQ: Median Generalized LVQ für mehrere Distanzmatrizen (DL = [D^(v)]).
    - Kombiniert Distanzen: D* = Σ_v w_v D^(v) mit lernbaren Gewichten w_v (w >= 0, Σ w = 1).
    - Vektorisierte Implementierung (keine O(V·m²)-Doppelschleifen).
    - Unterstützt K>1 sowie K==1 (multi-class/binary) mit spezialisierten Pfaden.
    - Optionales Tracking: Prototypenpfad, Gewichte, Loss-Historie.

    Parameter
    ---------
    K : int | dict[label -> int] | sequence[int]
        Anzahl Prototypen je Label (int = uniform für alle Labels).
    T : int
        Epochen.
    phi : callable or None
        Squashing-Funktion auf die GLVQ µ-Werte (default: Identität).
    track_path : bool
    track_vweights : bool
    track_metrics : bool
    eta : float
        Lernrate für die Matrixgewichte.
    v_init : array-like or None
        Initiale Gewichte der V Matrizen; wird auf das Simplex projiziert.
    """

    def __init__(
        self,
        K,
        T=50,
        phi=None,
        *,
        track_path=False,
        track_vweights=True,
        track_metrics=False,
        eta=1.0,
        v_init=None
    ):
        self.K = K
        self.T = T
        self.phi = (lambda mus: mus) if phi is None else phi

        self.track_path = bool(track_path)
        self.track_vweights = bool(track_vweights)
        self.track_metrics = bool(track_metrics)
        self.eta = float(eta)
        self.v_init = None if v_init is None else np.asarray(v_init, dtype=float)

        self._w_history = None
        self._v_history = None
        self._log = None
        self._snapshot_idx = 0

    # ---------- Tracking ----------
    def _init_tracking(self):
        self._snapshot_idx = 0
        self._w_history = [] if self.track_path else None
        self._v_history = [] if self.track_vweights else None
        self._log = {"loss": []} if self.track_metrics else None

    def _snapshot(self, *, w=None, v=None, loss=None):
        if self.track_path and w is not None:
            self._w_history.append(np.array(w, copy=True))
        if self.track_vweights and v is not None:
            self._v_history.append(np.array(v, dtype=float, copy=True))
        if self.track_metrics and loss is not None:
            self._log["loss"].append(float(loss))
        self._snapshot_idx += 1

    # ---------- Utilities ----------
    def _resolve_K_per_label(self, unique_labels):
        """
        self.K:
          - int -> gleiche Anzahl je Label
          - dict {label: k_l}
          - sequence[int] in Reihenfolge unique_labels
        Rückgabe: K_per (L,) als int-Array
        """
        L = len(unique_labels)
        if isinstance(self.K, (int, np.integer)):
            if self.K < 1:
                raise ValueError("K must be >= 1")
            return np.full(L, int(self.K), dtype=int)

        if isinstance(self.K, dict):
            K_per = np.empty(L, dtype=int)
            for i, lab in enumerate(unique_labels):
                if lab not in self.K:
                    raise ValueError(f"K missing for label {lab!r}")
                K_per[i] = int(self.K[lab])
            if np.any(K_per < 1):
                raise ValueError("All K_l must be >= 1")
            return K_per

        try:
            K_seq = np.asarray(self.K, dtype=int)
            if K_seq.shape != (L,):
                raise ValueError(f"K must have length {L} (one per label); got shape {K_seq.shape}")
            if np.any(K_seq < 1):
                raise ValueError("All K_l must be >= 1")
            return K_seq
        except Exception as e:
            raise ValueError("Unsupported K format. Use int, dict {label:k}, or sequence length=L") from e

    def _stack_DL(self, DL):
        """DL → M mit Shape (V, m, m) und Basisgrößen setzen; v_init respektieren."""
        M = np.stack(DL, axis=0)  # (V, m, m)
        self._V = M.shape[0]
        self._m = M.shape[1]

        # Gewichte initialisieren / übernehmen
        if self.v_init is not None:
            if self.v_init.shape != (self._V,):
                raise ValueError(f"v_init must have shape ({self._V},), got {self.v_init.shape}")
            self._vWeights = _project_simplex(self.v_init)
        elif not hasattr(self, "_vWeights") or self._vWeights is None or self._vWeights.size != self._V:
            self._vWeights = np.ones(self._V) / self._V

        # Optionaler Momentum-Puffer
        if not hasattr(self, "_v_m") or self._v_m is None or self._v_m.size != self._V:
            self._v_m = np.zeros(self._V, dtype=float)

        return M

    def _overall(self, M):
        """Gewichtete Summe über V (m, m)."""
        return np.tensordot(self._vWeights, M, axes=(0, 0))

    def _weights_update(self, dp, dm, dp_V, dm_V):
        """
        Vektorisierte mu_V- und Gewichtsaktualisierung + Simplex-Projektion.
        dp, dm: (m,)
        dp_V, dm_V: (V, m)
        """
        den  = (dp - dm + 1e-5) ** 2                         # (m,)
        num  = dp_V * dm - dm_V * dp                         # (V, m)
        grad = -(num * self._vWeights[:, None] / den).sum(axis=1)  # (V,)

        # Stabilisierung: Normierung (macht eta interpretierbar)
        g_norm = np.linalg.norm(grad, 1) + 1e-12
        grad /= g_norm

        # (optional) Momentum:
        # self._v_m = 0.9 * self._v_m + 0.1 * grad
        # step_vec = self._v_m
        step_vec = grad

        # Schritt + Simplex-Projektion
        self._vWeights = _project_simplex(self._vWeights + self.eta * step_vec)

    # ---------- Fit: allgemeiner Pfad (beliebige K_l) ----------
    def fit(self, DL, y):
        self._init_tracking()

        # Stack + Basis
        M = self._stack_DL(DL)                  # (V, m, m)
        unique_labels = np.unique(y)
        L = len(unique_labels)

        # K per label auflösen
        K_per = self._resolve_K_per_label(unique_labels)     # (L,)
        total_K = int(K_per.sum())

        # y/w-Container für alle Prototypen
        self._y = np.repeat(unique_labels, K_per)            # (total_K,)
        self._w = np.zeros(total_K, dtype=int)               # (total_K,)

        # Wenn alle K_l == 1 → spezieller Single-Pfad
        if np.all(K_per == 1):
            return self._fit_single(DL, y)

        # RNG-Initialisierung je Label mit jeweiligem K_l
        if (not hasattr(self, 'prevent_initialization')) or (not self.prevent_initialization):
            overall = self._overall(M)          # (m, m)
            offset = 0
            for l, lab in enumerate(unique_labels):
                k_l = int(K_per[l])
                idx_w = np.arange(offset, offset + k_l)      # Positionen in self._w, self._y gehört bereits zum Label
                inClass = np.where(y == lab)[0]
                D_l = np.square(overall[inClass, :][:, inClass])

                rng_l = rng.RNG(k_l)
                rng_l.fit(D_l, is_squared=True)
                Dp_l = rng_l._Alpha.dot(D_l) + np.expand_dims(rng_l._z, 1)
                closest = np.argmin(Dp_l, axis=1)
                self._w[idx_w] = inClass[closest]
                offset += k_l

        # Initiale closest ±
        closest_plus  = np.zeros(self._m, dtype=int)
        sndclosest_plus  = np.zeros(self._m, dtype=int)
        closest_minus = np.zeros(self._m, dtype=int)
        sndclosest_minus = np.zeros(self._m, dtype=int)

        overall = self._overall(M)
        # pro Label: positive/negative Prototypen finden
        for l, lab in enumerate(unique_labels):
            inClass = np.where(y == lab)[0]
            in_w    = np.where(self._y == lab)[0]
            out_w   = np.where(self._y != lab)[0]

            Dp = overall[inClass, :][:, self._w[in_w]]
            idx = np.argpartition(Dp, 1, axis=1)
            closest_plus[inClass]    = in_w[idx[:, 0]]
            sndclosest_plus[inClass] = in_w[idx[:, 1 if len(in_w) > 1 else 0]]

            Dm = overall[inClass, :][:, self._w[out_w]]
            idx = np.argpartition(Dm, 1, axis=1)
            closest_minus[inClass]    = out_w[idx[:, 0]]
            sndclosest_minus[inClass] = out_w[idx[:, 1 if len(out_w) > 1 else 0]]

        # Initialer Loss
        rows = np.arange(self._m)
        dp  = overall[rows, self._w[closest_plus]]
        dm  = overall[rows, self._w[closest_minus]]
        mus = self.phi((dp - dm) / (dp + dm + 1e-5))
        self._loss = [float(mus.sum())]

        # dp_V / dm_V (V, m)
        dp_V = M[:, rows, self._w[closest_plus]]
        dm_V = M[:, rows, self._w[closest_minus]]

        # Tracking
        self._snapshot(w=self._w, v=self._vWeights, loss=self._loss[-1])

        # Optimierung
        proto_losses = np.zeros(len(self._w))
        for _ in range(self.T):
            overall = self._overall(M)

            # Prototypen ranken
            for k in range(len(self._w)):
                proto_losses[k] = np.sum(dp[closest_plus == k]) - np.sum(dm[closest_minus == k])

            improved = False
            best_delta_global = 0.0

            for k in np.argsort(-proto_losses):
                lab_k = self._y[k]
                inClass_k  = np.where(y == lab_k)[0]
                outClass_k = np.where(y != lab_k)[0]
                rf_plus  = np.where(closest_plus == k)[0]
                rf_minus = np.where(closest_minus == k)[0]

                best_delta = 0.0
                best = None

                for i in rf_plus:
                    if i == self._w[k]:
                        continue

                    still_p = overall[rf_plus, i] <= overall[rf_plus, self._w[sndclosest_plus[rf_plus]]]
                    changed_plus  = np.unique(np.concatenate([
                        inClass_k[overall[inClass_k, i] < dp[inClass_k]],
                        rf_plus[still_p]
                    ]))
                    changed_plus2 = rf_plus[~still_p]

                    still_m = overall[rf_minus, i] <= overall[rf_minus, self._w[sndclosest_minus[rf_minus]]]
                    changed_minus  = np.unique(np.concatenate([
                        outClass_k[overall[outClass_k, i] < dm[outClass_k]],
                        rf_minus[still_m]
                    ]))
                    changed_minus2 = rf_minus[~still_m]

                    delta = 0.0
                    dp_new = overall[changed_plus, i]
                    mus_new = self.phi((dp_new - dm[changed_plus]) / (dp_new + dm[changed_plus] + 1e-5))
                    delta += np.sum(mus_new - mus[changed_plus])

                    dp_new = overall[changed_plus2, self._w[sndclosest_plus[changed_plus2]]]
                    mus_new = self.phi((dp_new - dm[changed_plus2]) / (dp_new + dm[changed_plus2] + 1e-5))
                    delta += np.sum(mus_new - mus[changed_plus2])

                    dm_new = overall[changed_minus, i]
                    mus_new = self.phi((dp[changed_minus] - dm_new) / (dp[changed_minus] + dm_new + 1e-5))
                    delta += np.sum(mus_new - mus[changed_minus])

                    dm_new = overall[changed_minus2, self._w[sndclosest_minus[changed_minus2]]]
                    mus_new = self.phi((dp[changed_minus2] - dm_new) / (dp[changed_minus2] + dm_new + 1e-5))
                    delta += np.sum(mus_new - mus[changed_minus2])

                    if delta < best_delta:
                        best_delta = delta
                        best = (i, changed_plus, changed_plus2, changed_minus, changed_minus2)

                if best is None:
                    continue

                # akzeptieren (greedy)
                best_i, c_p, c_p2, c_m, c_m2 = best
                self._w[k] = best_i
                improved = True
                best_delta_global = best_delta

                # positive updates
                closest_plus[c_p] = k
                dp[c_p] = overall[c_p, best_i]
                mus[c_p] = self.phi((dp[c_p] - dm[c_p]) / (dp[c_p] + dm[c_p] + 1e-5))
                dp_V[:, c_p] = M[:, c_p, best_i]

                w_in = np.where(self._y == self._y[k])[0]
                idx = np.argpartition(overall[c_p, :][:, self._w[w_in]], 1, axis=1)
                if len(w_in) > 1:
                    sndclosest_plus[c_p] = w_in[idx[:, 1]]

                closest_plus[c_p2] = sndclosest_plus[c_p2]
                dp[c_p2] = overall[c_p2, self._w[closest_plus[c_p2]]]
                mus[c_p2] = self.phi((dp[c_p2] - dm[c_p2]) / (dp[c_p2] + dm[c_p2] + 1e-5))
                dp_V[:, c_p2] = M[:, c_p2, self._w[closest_plus[c_p2]]]
                idx = np.argpartition(overall[c_p2, :][:, self._w[w_in]], 1, axis=1)
                if len(w_in) > 1:
                    sndclosest_plus[c_p2] = w_in[idx[:, 1]]

                # negative updates
                closest_minus[c_m] = k
                dm[c_m] = overall[c_m, best_i]
                mus[c_m] = self.phi((dp[c_m] - dm[c_m]) / (dp[c_m] + dm[c_m] + 1e-5))
                dm_V[:, c_m] = M[:, c_m, best_i]

                for l, lab in enumerate(unique_labels):
                    ic = c_m[np.where(y[c_m] == lab)[0]]
                    if ic.size == 0:
                        continue
                    w_out = np.where(self._y != lab)[0]
                    idx = np.argpartition(overall[ic, :][:, self._w[w_out]], 1, axis=1)
                    if len(w_out) > 1:
                        sndclosest_minus[ic] = w_out[idx[:, 1]]
                    else:
                        sndclosest_minus[ic] = w_out[0]

                closest_minus[c_m2] = sndclosest_minus[c_m2]
                dm[c_m2] = overall[c_m2, self._w[closest_minus[c_m2]]]
                mus[c_m2] = self.phi((dp[c_m2] - dm[c_m2]) / (dp[c_m2] + dm[c_m2] + 1e-5))
                dm_V[:, c_m2] = M[:, c_m2, self._w[closest_minus[c_m2]]]
                w_out = np.where(self._y != self._y[k])[0]
                idx = np.argpartition(overall[c_m2, :][:, self._w[w_out]], 1, axis=1)
                if len(w_out) > 1:
                    sndclosest_minus[c_m2] = w_out[idx[:, 1]]

                for l, lab in enumerate(unique_labels):
                    ic = c_m2[np.where(y[c_m2] == lab)[0]]
                    if ic.size == 0:
                        continue
                    w_out = np.where(self._y != lab)[0]
                    idx = np.argpartition(overall[ic, :][:, self._w[w_out]], 1, axis=1)
                    if len(w_out) > 1:
                        sndclosest_minus[ic] = w_out[idx[:, 1]]
                    else:
                        sndclosest_minus[ic] = w_out[0]

                expected_new = self._loss[-1] + best_delta
                actual_new = float(mus.sum())
                rel_err = abs(expected_new - actual_new) / (abs(self._loss[-1]) + 1e-12)
                if rel_err > 0.05:
                    print(f"[Warnung] Loss-Abweichung: erwartet {expected_new:.3f}, tatsächlich {actual_new:.3f} (rel_err={rel_err:.3%})")

                self._loss.append(actual_new)
                self._snapshot(w=self._w, v=self._vWeights, loss=self._loss[-1])

                # Matrixgewichte (vektorisiert + Simplex)
                self._weights_update(dp, dm, dp_V, dm_V)
                self._snapshot(w=self._w, v=self._vWeights, loss=self._loss[-1])

                break  # greedy

            if not improved or best_delta_global >= -_ERR_CUTOFF:
                self._snapshot(w=self._w, v=self._vWeights, loss=self._loss[-1])
                break

        return self

    # ---------- Fit: K == 1 (multi-class) ----------
    def _fit_single(self, DL, y):
        self._init_tracking()

        M = self._stack_DL(DL)
        unique_labels = np.unique(y)
        L = len(unique_labels)

        # K_per == 1 für alle Labels (durch Aufrufer gesichert)
        self._y = np.array(unique_labels, copy=True)
        self._w = np.zeros(L, dtype=int)

        overall = self._overall(M)
        # Prototyp = Medoid je Klasse
        for l, lab in enumerate(unique_labels):
            inClass = np.where(y == lab)[0]
            D_l = np.square(overall[inClass, :][:, inClass])
            self._w[l] = inClass[np.argmin(np.sum(D_l, axis=0))]

        closest_plus = np.zeros(self._m, dtype=int)
        for l, lab in enumerate(unique_labels):
            closest_plus[np.where(y == lab)[0]] = l

        closest_minus = np.zeros(self._m, dtype=int)
        sndclosest_minus = np.zeros(self._m, dtype=int)
        for l, lab in enumerate(unique_labels):
            inClass = np.where(y == lab)[0]
            out_w = np.where(self._y != lab)[0]
            Dm = overall[inClass, :][:, self._w[out_w]]
            idx = np.argpartition(Dm, 1, axis=1)
            closest_minus[inClass] = out_w[idx[:, 0]]
            sndclosest_minus[inClass] = out_w[idx[:, 1 if len(out_w) > 1 else 0]]

        rows = np.arange(self._m)
        dp = overall[rows, self._w[closest_plus]]
        dm = overall[rows, self._w[closest_minus]]
        mus = self.phi((dp - dm) / (dp + dm + 1e-5))
        self._loss = [float(mus.sum())]

        dp_V = M[:, rows, self._w[closest_plus]]
        dm_V = M[:, rows, self._w[closest_minus]]

        self._snapshot(w=self._w, v=self._vWeights, loss=self._loss[-1])

        proto_losses = np.zeros(len(self._w))
        for _ in range(self.T):
            overall = self._overall(M)

            for k in range(len(self._w)):
                proto_losses[k] = np.sum(dp[closest_plus == k]) - np.sum(dm[closest_minus == k])

            improved = False
            best_delta_global = 0.0
            for k in np.argsort(-proto_losses):
                inClass_k  = np.where(y == self._y[k])[0]
                outClass_k = np.where(y != self._y[k])[0]
                rf_plus  = np.where(closest_plus == k)[0]
                rf_minus = np.where(closest_minus == k)[0]

                best_delta = 0.0
                best = None
                for i in rf_plus:
                    if i == self._w[k]:
                        continue

                    still_m = overall[rf_minus, i] <= overall[rf_minus, self._w[sndclosest_minus[rf_minus]]]
                    changed_minus  = np.unique(np.concatenate([outClass_k[overall[outClass_k, i] < dm[outClass_k]],
                                                               rf_minus[still_m]]))
                    changed_minus2 = rf_minus[~still_m]

                    delta = 0.0
                    dp_new = overall[rf_plus, i]
                    mus_new = self.phi((dp_new - dm[rf_plus]) / (dp_new + dm[rf_plus] + 1e-5))
                    delta += np.sum(mus_new - mus[rf_plus])

                    dm_new = overall[changed_minus, i]
                    mus_new = self.phi((dp[changed_minus] - dm_new) / (dp[changed_minus] + dm_new + 1e-5))
                    delta += np.sum(mus_new - mus[changed_minus])

                    dm_new = overall[changed_minus2, self._w[sndclosest_minus[changed_minus2]]]
                    mus_new = self.phi((dp[changed_minus2] - dm_new) / (dp[changed_minus2] + dm_new + 1e-5))
                    delta += np.sum(mus_new - mus[changed_minus2])

                    if delta < best_delta:
                        best_delta = delta
                        best = (i, changed_minus, changed_minus2)

                if best is None:
                    continue

                i_best, c_m, c_m2 = best
                self._w[k] = i_best
                improved = True
                best_delta_global = best_delta

                dp[rf_plus] = overall[rf_plus, i_best]
                mus[rf_plus] = self.phi((dp[rf_plus] - dm[rf_plus]) / (dp[rf_plus] + dm[rf_plus] + 1e-5))
                dp_V[:, rf_plus] = M[:, rf_plus, i_best]

                closest_minus[c_m] = k
                dm[c_m] = overall[c_m, i_best]
                mus[c_m] = self.phi((dp[c_m] - dm[c_m]) / (dp[c_m] + dm[c_m] + 1e-5))
                dm_V[:, c_m] = M[:, c_m, i_best]

                for l, lab in enumerate(unique_labels):
                    ic = c_m[np.where(y[c_m] == lab)[0]]
                    if ic.size == 0:
                        continue
                    out_w = np.where(self._y != lab)[0]
                    idx = np.argpartition(overall[ic, :][:, self._w[out_w]], 1, axis=1)
                    if len(out_w) > 1:
                        sndclosest_minus[ic] = out_w[idx[:, 1]]
                    else:
                        sndclosest_minus[ic] = out_w[0]

                closest_minus[c_m2] = sndclosest_minus[c_m2]
                dm[c_m2] = overall[c_m2, self._w[closest_minus[c_m2]]]
                mus[c_m2] = self.phi((dp[c_m2] - dm[c_m2]) / (dp[c_m2] + dm[c_m2] + 1e-5))
                dm_V[:, c_m2] = M[:, c_m2, self._w[closest_minus[c_m2]]]
                out_w = np.where(self._y != self._y[k])[0]
                idx = np.argpartition(overall[c_m2, :][:, self._w[out_w]], 1, axis=1)
                if len(out_w) > 1:
                    sndclosest_minus[c_m2] = out_w[idx[:, 1]]

                for l, lab in enumerate(unique_labels):
                    ic = c_m2[np.where(y[c_m2] == lab)[0]]
                    if ic.size == 0:
                        continue
                    out_w = np.where(self._y != lab)[0]
                    idx = np.argpartition(overall[ic, :][:, self._w[out_w]], 1, axis=1)
                    if len(out_w) > 1:
                        sndclosest_minus[ic] = out_w[idx[:, 1]]
                    else:
                        sndclosest_minus[ic] = out_w[0]

                expected_new = self._loss[-1] + best_delta
                actual_new = float(mus.sum())
                rel_err = abs(expected_new - actual_new) / (abs(self._loss[-1]) + 1e-12)
                if rel_err > 0.05:
                    print(f"[Warnung] Loss-Abweichung: erwartet {expected_new:.3f}, tatsächlich {actual_new:.3f} (rel_err={rel_err:.3%})")

                self._loss.append(actual_new)
                self._snapshot(w=self._w, v=self._vWeights, loss=self._loss[-1])

                # Matrixgewichte
                self._weights_update(dp, dm, dp_V, dm_V)
                self._snapshot(w=self._w, v=self._vWeights, loss=self._loss[-1])

                break

            if not improved or best_delta_global >= -_ERR_CUTOFF:
                self._snapshot(w=self._w, v=self._vWeights, loss=self._loss[-1])
                break

        return self

    # ---------- Fit: K == 1 (binary) ----------
    def _fit_single_binary(self, DL, y):
        self._init_tracking()

        M = self._stack_DL(DL)
        unique_labels = np.unique(y)
        L = len(unique_labels)
        if L > 2:
            raise ValueError(f"Binary path requires 2 classes, got {L}")

        self._y = np.array(unique_labels, copy=True)
        self._w = np.zeros(L, dtype=int)

        overall = self._overall(M)
        for l, lab in enumerate(unique_labels):
            inClass = np.where(y == lab)[0]
            D_l = np.square(overall[inClass, :][:, inClass])
            self._w[l] = inClass[np.argmin(np.sum(D_l, axis=0))]

        closest_plus  = np.zeros(self._m, dtype=int)
        closest_minus = np.zeros(self._m, dtype=int)
        for l, lab in enumerate(unique_labels):
            inClass = np.where(y == lab)[0]
            closest_plus[inClass]  = l
            closest_minus[inClass] = 1 - l

        rows = np.arange(self._m)
        dp = overall[rows, self._w[closest_plus]]
        dm = overall[rows, self._w[closest_minus]]
        mus = self.phi((dp - dm) / (dp + dm + 1e-5))
        self._loss = [float(mus.sum())]

        dp_V = M[:, rows, self._w[closest_plus]]
        dm_V = M[:, rows, self._w[closest_minus]]

        self._snapshot(w=self._w, v=self._vWeights, loss=self._loss[-1])

        proto_losses = np.zeros(len(self._w))
        for _ in range(self.T):
            overall = self._overall(M)

            for k in range(len(self._w)):
                proto_losses[k] = np.sum(dp[closest_plus == k]) - np.sum(dm[closest_minus == k])

            improved = False
            best_delta_global = 0.0
            for k in np.argsort(-proto_losses):
                rf_plus  = np.where(closest_plus == k)[0]
                rf_minus = np.where(closest_minus == k)[0]

                best_delta = 0.0
                best_i = None
                for i in rf_plus:
                    if i == self._w[k]:
                        continue

                    delta = 0.0
                    dp_new = overall[rf_plus, i]
                    mus_new = self.phi((dp_new - dm[rf_plus]) / (dp_new + dm[rf_plus] + 1e-5))
                    delta += np.sum(mus_new - mus[rf_plus])

                    dm_new = overall[rf_minus, i]
                    mus_new = self.phi((dp[rf_minus] - dm_new) / (dp[rf_minus] + dm_new + 1e-5))
                    delta += np.sum(mus_new - mus[rf_minus])

                    if delta < best_delta:
                        best_delta = delta
                        best_i = i

                if best_i is None:
                    continue

                self._w[k] = best_i
                improved = True
                best_delta_global = best_delta

                dp[rf_plus] = overall[rf_plus, best_i]
                mus[rf_plus] = self.phi((dp[rf_plus] - dm[rf_plus]) / (dp[rf_plus] + dm[rf_plus] + 1e-5))
                dp_V[:, rf_plus] = M[:, rf_plus, best_i]

                dm[rf_minus] = overall[rf_minus, best_i]
                mus[rf_minus] = self.phi((dp[rf_minus] - dm[rf_minus]) / (dp[rf_minus] + dm[rf_minus] + 1e-5))
                dm_V[:, rf_minus] = M[:, rf_minus, best_i]

                expected_new = self._loss[-1] + best_delta
                actual_new = float(mus.sum())
                rel_err = abs(expected_new - actual_new) / (abs(self._loss[-1]) + 1e-12)
                if rel_err > 0.05:
                    print(f"[Warnung] Loss-Abweichung: erwartet {expected_new:.3f}, tatsächlich {actual_new:.3f} (rel_err={rel_err:.3%})")

                self._loss.append(actual_new)
                self._snapshot(w=self._w, v=self._vWeights, loss=self._loss[-1])

                # Gewichte
                self._weights_update(dp, dm, dp_V, dm_V)
                self._snapshot(w=self._w, v=self._vWeights, loss=self._loss[-1])

                break

            if not improved or best_delta_global >= -_ERR_CUTOFF:
                self._snapshot(w=self._w, v=self._vWeights, loss=self._loss[-1])
                break

        return self

    # ---------- Inferenz ----------
    def predict(self, DL):
        """
        DL: Liste von V Testdistanzmatrizen (n x m) ODER (n x total_K) zu Prototypen.
        Gibt Labels (n,) zurück.
        """
        Mtest = np.stack(DL, axis=0)  # (V, n, m) oder (V, n, total_K)
        D = np.tensordot(self._vWeights, Mtest, axes=(0, 0))  # (n, m) oder (n, total_K)
        if D.shape[1] == self._m:
            D = D[:, self._w]  # n x total_K
        closest = np.argmin(D, axis=1)
        return self._y[closest]

    # ---------- Auslesen ----------
    def get_prototype_path(self):
        if self._w_history is None:
            return None
        return np.vstack(self._w_history) if len(self._w_history) else np.empty((0,))

    def get_vweight_path(self):
        if self._v_history is None:
            return None
        return np.vstack(self._v_history) if len(self._v_history) else np.empty((0,))

    def get_training_log(self):
        if self._log is None:
            return None
        return {"loss": np.array(self._log["loss"], dtype=float)}
