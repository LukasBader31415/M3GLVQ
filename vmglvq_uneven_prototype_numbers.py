import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import proto_dist_ml.rng as rng

__author__ = 'Benjamin Paaßen, erweitert'
__license__ = 'GPLv3'
__version__ = '1.3.0-track'  # + Tracking für Prototypen- und Distanzgewichte

_ERR_CUTOFF = 1E-5


class VMGLVQ(BaseEstimator, ClassifierMixin):
    """Median Generalized LVQ mit variabler Prototypenzahl pro Klasse + Tracking.

    Parameter
    ---------
    K : int or None
        Feste Zahl Prototypen *pro Klasse*. Entweder `K` ODER `K_per_class` setzen.
    K_per_class : dict | array-like | None
        Abbildung label -> Anzahl Prototypen. Beispiel: {0:3, 1:1}.
    T : int
        Epochen.
    phi : callable or None
        Fehlerfunktion, Standard: Identität.
    v_init : array-like or None
        Initialgewichte der Distanzmatrizen, wird auf Simplex projiziert.
    eta_v : float
        Lernrate für Gewichtsupdate der Distanzmatrizen.

    Tracking
    --------
    track_path : bool
        Wenn True, speichere _w nach jeder Verbesserung (wie bei deinem MGLVQ).
    track_vweights : bool
        Wenn True, speichere _vWeights bei Initialisierung und nach JEDEM Update.
    track_metrics : bool
        Wenn True, speichere Loss sowie dp/dm (und pro-V-Matrizen) bei Schnappschüssen.

    Auslesen
    --------
    - get_prototype_path() -> (steps, K): int
    - get_vweight_path()   -> (steps, V): float
    - get_training_log()   -> dict mit 'loss', optional 'dp', 'dm', 'dp_V', 'dm_V', 'proto_losses'
    """

    def __init__(
        self,
        K=None,
        K_per_class=None,
        T=50,
        phi=None,
        v_init=None,
        eta_v=1e-2,
        *,
        track_path=False,
        track_vweights=True,
        track_metrics=False,
    ):
        if (K is None) == (K_per_class is None):
            raise ValueError("Entweder K oder K_per_class angeben, aber nicht beides/keines.")
        self.K = K
        self.K_per_class = K_per_class
        self.T = T
        self.v_init = v_init
        self.eta_v = float(eta_v)
        self.phi = (lambda x: x) if phi is None else phi

        # Tracking-Flags
        self.track_path = bool(track_path)
        self.track_vweights = bool(track_vweights)
        self.track_metrics = bool(track_metrics)

        # interne Logs
        self._w_history = None
        self._v_history = None
        self._log = None
        self._snapshot_idx = 0

    # ---------------- Hilfsfunktionen ----------------
    def _normalize_v(self, v):
        v = np.asarray(v, dtype=float)
        v = np.maximum(v, 0.0)
        s = float(np.sum(v))
        if s <= 1e-12:
            v = np.ones_like(v) / len(v)
        else:
            v = v / s
        return v

    def _init_vweights(self, V):
        if self.v_init is None:
            return np.ones(V, dtype=float) / V
        v = np.asarray(self.v_init, dtype=float)
        if v.shape[0] != V:
            raise ValueError(f"v_init Länge {v.shape[0]} passt nicht zu V={V}.")
        return self._normalize_v(v)

    def _resolve_K_per_class(self, unique_labels):
        # Liefert dict label -> k_l und Gesamtzahl
        if self.K_per_class is not None:
            if isinstance(self.K_per_class, dict):
                kmap = {lbl: int(self.K_per_class.get(lbl, 0)) for lbl in unique_labels}
            else:
                arr = np.asarray(self.K_per_class)
                if arr.shape[0] != len(unique_labels):
                    raise ValueError("K_per_class array muss Länge = #Labels haben")
                kmap = {lbl: int(arr[i]) for i, lbl in enumerate(unique_labels)}
        else:
            kmap = {lbl: int(self.K) for lbl in unique_labels}
        if any(k < 0 for k in kmap.values()):
            raise ValueError("Prototypenzahlen müssen >= 0 sein.")
        if all(k == 0 for k in kmap.values()):
            raise ValueError("Mindestens eine Klasse muss >=1 Prototyp haben.")
        return kmap, sum(kmap.values())

    def _mix_dist(self, DL, w):
        # Mischt V Matrizen gemäß Gewichten w (nicht optimiert)
        m = DL[0].shape[0]
        D = np.zeros((m, m))
        for i in range(m):
            # kleine Beschleunigung: spaltenweise Dot-Produkt
            rows = np.array([M[i, :] for M in DL])  # (V, m)
            D[i, :] = rows.T.dot(w)
        return D

    # --------- Tracking-Initialisierung & Snapshot ----------
    def _init_tracking(self):
        self._snapshot_idx = 0
        if self.track_path:
            self._w_history = []
        if self.track_vweights:
            self._v_history = []
        if self.track_metrics:
            self._log = {
                "loss": [],
                # optional Felder werden später gefüllt, wenn Größen bekannt
                "dp": [],
                "dm": [],
                "dp_V": None,  # wird zu Liste[List[np.ndarray]] init. sobald V bekannt
                "dm_V": None,
                "proto_losses": [],
            }

    def _snapshot(self, *, w=None, v=None, loss=None, dp=None, dm=None, dp_V=None, dm_V=None, proto_losses=None):
        """Einen konsistenten Schnappschuss in die Verlaufs-Container schreiben."""
        if self.track_path and w is not None:
            self._w_history.append(np.array(w, copy=True))
        if self.track_vweights and v is not None:
            self._v_history.append(np.array(v, dtype=float, copy=True))
        if self.track_metrics:
            if loss is not None:
                self._log["loss"].append(float(loss))
            if dp is not None:
                self._log["dp"].append(np.array(dp, copy=True))
            if dm is not None:
                self._log["dm"].append(np.array(dm, copy=True))
            if dp_V is not None:
                if self._log["dp_V"] is None:
                    self._log["dp_V"] = [[] for _ in range(len(dp_V))]
                for v_i, arr in enumerate(dp_V):
                    self._log["dp_V"][v_i].append(np.array(arr, copy=True))
            if dm_V is not None:
                if self._log["dm_V"] is None:
                    self._log["dm_V"] = [[] for _ in range(len(dm_V))]
                for v_i, arr in enumerate(dm_V):
                    self._log["dm_V"][v_i].append(np.array(arr, copy=True))
            if proto_losses is not None:
                self._log["proto_losses"].append(np.array(proto_losses, copy=True))
        self._snapshot_idx += 1

    # ---------------- Training (allgemein) ----------------
    def fit(self, DL, y):
        self._V = len(DL)
        for v in range(self._V):
            if DL[v].ndim != 2:
                raise ValueError('Input is not a matrix!')
            if DL[v].shape[0] != DL[v].shape[1]:
                raise ValueError('Input matrix is not square!')
            self._m = DL[v].shape[0]
        y = np.asarray(y)
        unique_labels = np.unique(y)
        kmap, total_K = self._resolve_K_per_class(unique_labels)

        # Tracking vorbereiten
        self._init_tracking()

        # Sonderfall: alle k_l == 1 -> schnelle Single-Logik
        if all(kmap[lbl] == 1 for lbl in unique_labels):
            self.K = 1
            self._vWeights = self._init_vweights(self._V)
            out = self._fit_single_binary(DL, y) if len(unique_labels) <= 2 else self._fit_single_multi(DL, y)
            # initialen Snapshot schreiben
            self._snapshot(w=self._w, v=self._vWeights)
            return out

        # Allgemeiner Fall: variable k_l
        self._vWeights = self._init_vweights(self._V)

        # Prototyp-Container
        self._w = np.zeros(total_K, dtype=int)
        self._y = np.empty(total_K, dtype=unique_labels.dtype)

        # Initialisierung per class-wise RNG, mit Offsets
        offset = 0
        Dmix = self._mix_dist(DL, self._vWeights)
        for lbl in unique_labels:
            k_l = kmap[lbl]
            if k_l == 0:
                continue
            inClass = np.where(y == lbl)[0]
            D_l = np.square(Dmix[inClass, :][:, inClass])
            rng_l = rng.RNG(k_l)
            rng_l.fit(D_l, is_squared=True)
            Dp_l = rng_l._Alpha.dot(D_l) + np.expand_dims(rng_l._z, 1)
            closest = np.argmin(Dp_l, axis=1)
            self._w[offset:offset + k_l] = inClass[closest]
            self._y[offset:offset + k_l] = lbl
            offset += k_l

        # Vorbereitungen für Optimierung
        self._loss = []
        m = self._m
        Ktot = len(self._w)
        closest_plus = np.zeros(m, dtype=int)
        sndclosest_plus = np.zeros(m, dtype=int)
        closest_minus = np.zeros(m, dtype=int)
        sndclosest_minus = np.zeros(m, dtype=int)

        def recompute_neighborhoods(Dmix_local):
            for l_i, lbl in enumerate(unique_labels):
                inClass_idx = np.where(y == lbl)[0]
                w_in = np.where(self._y == lbl)[0]
                if len(w_in) == 0:
                    continue
                Dp = Dmix_local[inClass_idx, :][:, self._w[w_in]]
                kth = 1 if len(w_in) > 1 else 0
                idxs = np.argpartition(Dp, kth, axis=1)
                closest_plus[inClass_idx] = w_in[idxs[:, 0]]
                sndclosest_plus[inClass_idx] = w_in[idxs[:, kth]]
                w_out = np.where(self._y != lbl)[0]
                Dm = Dmix_local[inClass_idx, :][:, self._w[w_out]]
                kthm = 1 if len(w_out) > 1 else 0
                idxs = np.argpartition(Dm, kthm, axis=1)
                closest_minus[inClass_idx] = w_out[idxs[:, 0]]
                sndclosest_minus[inClass_idx] = w_out[idxs[:, kthm]]

        Dmix = self._mix_dist(DL, self._vWeights)
        recompute_neighborhoods(Dmix)
        dp = Dmix[np.arange(m), self._w[closest_plus]]
        dm = Dmix[np.arange(m), self._w[closest_minus]]
        mus = self.phi((dp - dm) / (dp + dm + 1E-5))
        self._loss.append(np.sum(mus))

        dp_V = [DL[v][np.arange(m), self._w[closest_plus]].copy() for v in range(self._V)]
        dm_V = [DL[v][np.arange(m), self._w[closest_minus]].copy() for v in range(self._V)]

        proto_losses = np.zeros(Ktot)

        # Initialer Snapshot (t=0)
        self._snapshot(
            w=self._w, v=self._vWeights, loss=self._loss[-1],
            dp=dp, dm=dm, dp_V=dp_V, dm_V=dm_V, proto_losses=proto_losses
        )

        for t in range(self.T):
            Dmix = self._mix_dist(DL, self._vWeights)
            for k in range(Ktot):
                proto_losses[k] = np.sum(dp[closest_plus == k]) - np.sum(dm[closest_minus == k])

            any_update = False

            for k in np.argsort(-proto_losses):
                lbl_k = self._y[k]
                inClass_k = np.where(y == lbl_k)[0]
                outClass_k = np.where(y != lbl_k)[0]
                rf_plus = np.where(closest_plus == k)[0]
                rf_minus = np.where(closest_minus == k)[0]

                best_delta_loss = 0.0
                best_i = None
                # Platzhalter für Sets
                best_changed_plus = best_changed_plus2 = np.array([], dtype=int)
                best_changed_minus = best_changed_minus2 = np.array([], dtype=int)

                for i_pt in rf_plus:
                    if i_pt == self._w[k]:
                        continue
                    still_closest = Dmix[rf_plus, i_pt] <= Dmix[rf_plus, self._w[sndclosest_plus[rf_plus]]]
                    changed_plus = np.unique(np.concatenate([
                        inClass_k[Dmix[inClass_k, i_pt] < dp[inClass_k]],
                        rf_plus[still_closest]
                    ]))
                    changed_plus2 = rf_plus[~still_closest]
                    still_closest_m = Dmix[rf_minus, i_pt] <= Dmix[rf_minus, self._w[sndclosest_minus[rf_minus]]]
                    changed_minus = np.unique(np.concatenate([
                        outClass_k[Dmix[outClass_k, i_pt] < dm[outClass_k]],
                        rf_minus[still_closest_m]
                    ]))
                    changed_minus2 = rf_minus[~still_closest_m]

                    delta = 0.0
                    if changed_plus.size:
                        dp_new = Dmix[changed_plus, i_pt]
                        mus_new = self.phi((dp_new - dm[changed_plus]) / (dp_new + dm[changed_plus] + 1E-5))
                        delta += np.sum(mus_new - mus[changed_plus])
                    if changed_plus2.size:
                        dp_new = Dmix[changed_plus2, self._w[sndclosest_plus[changed_plus2]]]
                        mus_new = self.phi((dp_new - dm[changed_plus2]) / (dp_new + dm[changed_plus2] + 1E-5))
                        delta += np.sum(mus_new - mus[changed_plus2])
                    if changed_minus.size:
                        dm_new = Dmix[changed_minus, i_pt]
                        mus_new = self.phi((dp[changed_minus] - dm_new) / (dp[changed_minus] + dm_new + 1E-5))
                        delta += np.sum(mus_new - mus[changed_minus])
                    if changed_minus2.size:
                        dm_new = Dmix[changed_minus2, self._w[sndclosest_minus[changed_minus2]]]
                        mus_new = self.phi((dp[changed_minus2] - dm_new) / (dp[changed_minus2] + dm[changed_minus2] + 1E-5))
                        delta += np.sum(mus_new - mus[changed_minus2])

                    if delta < best_delta_loss:
                        best_delta_loss = delta
                        best_i = i_pt
                        best_changed_plus = changed_plus
                        best_changed_plus2 = changed_plus2
                        best_changed_minus = changed_minus
                        best_changed_minus2 = changed_minus2

                if best_delta_loss < 0.0 and best_i is not None:
                    any_update = True
                    # --- Prototyp-Update ---
                    self._w[k] = best_i
                    if self.track_path:
                        self._snapshot(w=self._w)  # nur _w schreiben

                    if best_changed_plus.size:
                        closest_plus[best_changed_plus] = k
                        dp[best_changed_plus] = Dmix[best_changed_plus, best_i]
                        mus[best_changed_plus] = self.phi((dp[best_changed_plus] - dm[best_changed_plus]) / (dp[best_changed_plus] + dm[best_changed_plus] + 1E-5))
                        for v in range(self._V):
                            dp_V[v][best_changed_plus] = DL[v][best_changed_plus, best_i]
                        w_in = np.where(self._y == lbl_k)[0]
                        idxs = np.argpartition(Dmix[best_changed_plus, :][:, self._w[w_in]], 1 if len(w_in) > 1 else 0, axis=1)
                        sndclosest_plus[best_changed_plus] = w_in[idxs[:, 1 if len(w_in) > 1 else 0]]
                    if best_changed_plus2.size:
                        closest_plus[best_changed_plus2] = sndclosest_plus[best_changed_plus2]
                        dp[best_changed_plus2] = Dmix[best_changed_plus2, self._w[closest_plus[best_changed_plus2]]]
                        mus[best_changed_plus2] = self.phi((dp[best_changed_plus2] - dm[best_changed_plus2]) / (dp[best_changed_plus2] + dm[best_changed_plus2] + 1E-5))
                        for v in range(self._V):
                            dp_V[v][best_changed_plus2] = DL[v][best_changed_plus2, self._w[closest_plus[best_changed_plus2]]]
                        w_in = np.where(self._y == lbl_k)[0]
                        idxs = np.argpartition(Dmix[best_changed_plus2, :][:, self._w[w_in]], 1 if len(w_in) > 1 else 0, axis=1)
                        sndclosest_plus[best_changed_plus2] = w_in[idxs[:, 1 if len(w_in) > 1 else 0]]
                    if best_changed_minus.size:
                        closest_minus[best_changed_minus] = k
                        dm[best_changed_minus] = Dmix[best_changed_minus, best_i]
                        mus[best_changed_minus] = self.phi((dp[best_changed_minus] - dm[best_changed_minus]) / (dp[best_changed_minus] + dm[best_changed_minus] + 1E-5))
                        for v in range(self._V):
                            dm_V[v][best_changed_minus] = DL[v][best_changed_minus, best_i]
                        for lbl in unique_labels:
                            inClass_l = best_changed_minus[y[best_changed_minus] == lbl]
                            w_out = np.where(self._y != lbl)[0]
                            if inClass_l.size and w_out.size:
                                idxs = np.argpartition(Dmix[inClass_l, :][:, self._w[w_out]], 1 if len(w_out) > 1 else 0, axis=1)
                                sndclosest_minus[inClass_l] = w_out[idxs[:, 1 if len(w_out) > 1 else 0]]
                    if best_changed_minus2.size:
                        closest_minus[best_changed_minus2] = sndclosest_minus[best_changed_minus2]
                        dm[best_changed_minus2] = Dmix[best_changed_minus2, self._w[closest_minus[best_changed_minus2]]]
                        mus[best_changed_minus2] = self.phi((dp[best_changed_minus2] - dm[best_changed_minus2]) / (dp[best_changed_minus2] + dm[best_changed_minus2] + 1E-5))
                        for v in range(self._V):
                            dm_V[v][best_changed_minus2] = DL[v][best_changed_minus2, self._w[closest_minus[best_changed_minus2]]]
                        w_out = np.where(self._y != lbl_k)[0]
                        if w_out.size:
                            kth = 1 if w_out.size > 1 else 0
                            sub = Dmix[best_changed_minus2, :][:, self._w[w_out]]
                            idxs = np.argpartition(sub, kth, axis=1)
                            sndclosest_minus[best_changed_minus2] = w_out[idxs[:, kth]]
                        for lbl in unique_labels:
                            inClass_l = best_changed_minus2[y[best_changed_minus2] == lbl]
                            w_out = np.where(self._y != lbl)[0]
                            if inClass_l.size and w_out.size:
                                idxs = np.argpartition(Dmix[inClass_l, :][:, self._w[w_out]], 1 if len(w_out) > 1 else 0, axis=1)
                                sndclosest_minus[inClass_l] = w_out[idxs[:, 1 if len(w_out) > 1 else 0]]

                    expected_new = self._loss[-1] + best_delta_loss
                    actual_new = np.sum(mus)
                    rel_err = abs(expected_new - actual_new) / (abs(self._loss[-1]) + 1e-12)
                    if rel_err > 0.05:
                        print(f"[Warnung] Loss-Abweichung: erwartet {expected_new:.3f}, tatsächlich {actual_new:.3f} (rel_err={rel_err:.3%})")
                    self._loss.append(actual_new)

                    # --- Gewichtsupdate (Matrizen) ---
                    for v in range(self._V):
                        mu_comp = (np.array(dp_V[v]) * np.array(dm) * self._vWeights[v] -
                                   np.array(dm_V[v]) * np.array(dp) * self._vWeights[v]) / (
                                      (np.array(dp) - np.array(dm) + 1E-5)**2)
                        delta_v = -np.sum(mu_comp)
                        self._vWeights[v] += self.eta_v * delta_v
                    self._vWeights = self._normalize_v(self._vWeights)

                    # Vollständiger Snapshot NACH dem kombinierten Update
                    self._snapshot(
                        w=self._w,
                        v=self._vWeights,
                        loss=self._loss[-1],
                        dp=dp, dm=dm,
                        dp_V=dp_V, dm_V=dm_V,
                        proto_losses=proto_losses
                    )
                    break  # greedy: nach erster Verbesserung weiter
            if not any_update or best_delta_loss >= -_ERR_CUTOFF:
                # trotzdem letzten Zustand protokollieren (nützlich fürs Plotten)
                self._snapshot(
                    w=self._w, v=self._vWeights, loss=self._loss[-1],
                    dp=dp, dm=dm, dp_V=dp_V, dm_V=dm_V, proto_losses=proto_losses
                )
                break

        return self

    # ---------------- Spezielle Single-Fälle (für K_l == 1 überall) ----------------
    def _fit_single_multi(self, DL, y):
        self._V = len(DL)
        self._vWeights = self._init_vweights(self._V)
        y = np.asarray(y)
        unique_labels = np.unique(y)
        L = len(unique_labels)
        m = DL[0].shape[0]
        Dmix = self._mix_dist(DL, self._vWeights)
        self._w = np.zeros(L, dtype=int)
        self._y = unique_labels.copy()
        for l, lbl in enumerate(unique_labels):
            inClass = np.where(y == lbl)[0]
            D_l = np.square(Dmix[inClass, :][:, inClass])
            self._w[l] = inClass[np.argmin(np.sum(D_l, axis=0))]
        # Snapshot initial
        self._snapshot(w=self._w, v=self._vWeights)
        return self

    def _fit_single_binary(self, DL, y):
        self._V = len(DL)
        y = np.asarray(y)
        unique_labels = np.unique(y)
        if len(unique_labels) != 2:
            raise ValueError("_fit_single_binary erfordert genau 2 Klassen.")
        self._vWeights = self._init_vweights(self._V)
        Dmix = self._mix_dist(DL, self._vWeights)
        self._w = np.zeros(2, dtype=int)
        self._y = unique_labels.copy()
        for l, lbl in enumerate(unique_labels):
            inClass = np.where(y == lbl)[0]
            D_l = np.square(Dmix[inClass, :][:, inClass])
            self._w[l] = inClass[np.argmin(np.sum(D_l, axis=0))]
        # Snapshot initial
        self._snapshot(w=self._w, v=self._vWeights)
        return self

    # ---------------- Inferenz ----------------
    def predict(self, DL):
        Dmix = self._mix_dist(DL, self._vWeights)
        if Dmix.shape[1] == self._m:
            D = Dmix[:, self._w]
        else:
            D = Dmix
        closest = np.argmin(D, axis=1)
        return self._y[closest]

    # ---------------- Auslese-APIs ----------------
    def get_prototype_path(self):
        """Gibt (steps, K)-Array mit Prototyp-Indizes zurück (oder None)."""
        if self._w_history is None:
            return None
        return np.vstack(self._w_history) if len(self._w_history) else np.empty((0,))

    def get_vweight_path(self):
        """Gibt (steps, V)-Array mit Distanz-Gewichten zurück (oder None)."""
        if self._v_history is None:
            return None
        return np.vstack(self._v_history) if len(self._v_history) else np.empty((0,))

    def get_training_log(self):
        """Gibt dict mit Verlaufsdaten zurück (Loss, dp/dm, …), falls mitgeloggt."""
        if self._log is None:
            return None
        # Kopien, damit außerhalb nichts mutiert wird
        out = {
            "loss": np.array(self._log["loss"], dtype=float),
            "dp": [a.copy() for a in self._log["dp"]],
            "dm": [a.copy() for a in self._log["dm"]],
            "proto_losses": [a.copy() for a in self._log["proto_losses"]],
            "eta_v": float(self.eta_v),  # Lernrate ist konstant, aber für die Doku praktisch
        }
        if self._log["dp_V"] is not None:
            out["dp_V"] = [[a.copy() for a in per_v] for per_v in self._log["dp_V"]]
        if self._log["dm_V"] is not None:
            out["dm_V"] = [[a.copy() for a in per_v] for per_v in self._log["dm_V"]]
        return out
