
# -*- coding: utf-8 -*-
"""
preparation.py
--------------
Hilfsfunktionen zur Vorbereitung von Distanz-/Ähnlichkeitsmatrizen für weitere
Analysen und (spätere) grafische Auswertungen.

Rubrik: Preparation
"""

from typing import Dict, Tuple, List, Union
import pandas as pd
import numpy as np


# === Preparation ============================================================

def align_square_matrix_to_df_order(
    D_df: pd.DataFrame,
    df: pd.DataFrame,
    id_col: str,
    return_numpy: bool = False
):
    """
    Bringt die quadratische Matrix `D_df` (Index=IDs, Columns=IDs) exakt in die
    Reihenfolge von `df[id_col]`. Achtet auf Typkonsistenz (cast zu str) und
    prüft auf fehlende IDs.

    Parameters
    ----------
    D_df : pd.DataFrame
        Quadratische Matrix mit identischen ID-Mengen in Index und Spalten.
    df : pd.DataFrame
        DataFrame, dessen Spalte `id_col` die gewünschte Reihenfolge definiert.
    id_col : str
        Spaltenname in `df`, der die IDs enthält.
    return_numpy : bool, optional (default: False)
        Wenn True, wird ein NumPy-Array zurückgegeben, sonst ein DataFrame.

    Returns
    -------
    Union[pd.DataFrame, np.ndarray]
        Neu sortierte Matrix (DataFrame oder ndarray).
    """
    # alles zu str, damit "1000013" == "1000013"
    D_df = D_df.copy()
    D_df.index = D_df.index.astype(str)
    D_df.columns = D_df.columns.astype(str)
    df_ids = df[id_col].astype(str)

    # Konsistenzchecks
    if set(D_df.index) != set(D_df.columns):
        raise ValueError("Index und Columns von D_df enthalten unterschiedliche ID-Mengen.")
    missing = set(df_ids) - set(D_df.index)
    if missing:
        ex = ", ".join(list(missing)[:5])
        more = " ..." if len(missing) > 5 else ""
        raise ValueError(f"IDs aus df fehlen in D_df: {ex}{more}")

    # Reindex in df-Reihenfolge
    D_sorted = D_df.loc[df_ids, df_ids]

    # Optional: schnelle Symmetrieprüfung (bei Distanzen üblich)
    if not np.allclose(D_sorted.values, D_sorted.values.T, equal_nan=True):
        print("Warnung: D_sorted ist nicht exakt symmetrisch.")

    # Optional: NaN-Check
    if D_sorted.isna().any().any():
        nmiss = int(D_sorted.isna().sum().sum())
        raise ValueError(f"Matrix enthält {nmiss} NaN-Werte nach dem Reindex.")

    return D_sorted.values if return_numpy else D_sorted


def align_matrices_to_df(
    mats: Dict[str, pd.DataFrame],
    df: pd.DataFrame,
    id_col: str = "CustomerCode",
    set_diag_zero: bool = True,
    set_globals: bool = False,
    verbose: bool = True,
    matrices_as_numpy: bool = True,   # << Nur für MATRIZEN steuern
) -> Tuple[Dict[str, Union[np.ndarray, pd.DataFrame]], pd.DataFrame, List]:
    """
    1) KGN (Index∩Spalten über alle Matrizen).
    2) df auf KGN kürzen und Reihenfolge (id_col) mit erstem Vorkommen beibehalten.
    3) Alle Matrizen exakt auf diese Reihenfolge reindizieren.
    4) NUR für die MATRIZEN wählbar: NumPy-Arrays ODER DataFrames (per matrices_as_numpy).
       df_cut bleibt DataFrame, ordered_ids bleibt Liste.

    Rückgabe:
      aligned : dict[str, ndarray|DataFrame]  # Typ je nach matrices_as_numpy
      df_cut  : DataFrame
      ordered_ids : list

    Wenn set_globals=True:
      - Für jede Matrix 'name': globals()['{name}_p_aligned'] = (ndarray ODER DataFrame)
        (also z. B. D_naics_sub_p_aligned im gewünschten Typ)
    """
    if not mats:
        raise ValueError("mats ist leer.")
    if id_col not in df.columns:
        raise KeyError(f"Spalte '{id_col}' nicht im DataFrame.")

    # --- Schritt 1: KGN bestimmen ---
    idx_sets  = [set(M.index)   for M in mats.values()]
    col_sets  = [set(M.columns) for M in mats.values()]
    common_ids = set.intersection(*idx_sets) & set.intersection(*col_sets)
    if verbose:
        print(f"[1] KGN (Schnittmenge über alle Matrizen): {len(common_ids)} IDs")

    # --- Schritt 2: df kürzen + Reihenfolge (Duplikate: erstes Vorkommen) ---
    df_cut = df[df[id_col].isin(common_ids)].copy()
    
    seen = set()
    ordered_ids: List = [x for x in df_cut[id_col].tolist() if not (x in seen or seen.add(x))]
    if verbose:
        print(f"[2] DF gekürzt auf KGN: {len(df_cut)} Zeilen | eindeutige IDs in DF-Reihenfolge: {len(ordered_ids)}")

    # --- Schritt 3: Reindizieren ---
    aligned: Dict[str, Union[np.ndarray, pd.DataFrame]] = {}
    n = len(ordered_ids)
    for name, M in mats.items():
        A_df = M.reindex(index=ordered_ids, columns=ordered_ids)
        if set_diag_zero and n > 0:
            A_df.values[range(n), range(n)] = 0.0

        if matrices_as_numpy:
            A = A_df.to_numpy(copy=True)           # nur Array behalten
        else:
            A = A_df                                # DataFrame behalten

        aligned[name] = A

        if set_globals:
            globals()[f"{name}_p_aligned"] = A      # z. B. D_naics_sub_p_aligned

        if verbose:
            shape = A.shape
            typ = "ndarray" if matrices_as_numpy else "DataFrame"
            print(f"[3] {name}_p_aligned -> {shape} ({typ})")

    try:
        if df_cut.shape[1] >= 2:
            df_cut_slim = df_cut[df_cut.columns[1]]  # zweite Spalte
        else:
            df_cut_slim = df_cut.iloc[:, 0]
        if verbose:
            print(f"[4] df_cut -> Series mit Name '{df_cut_slim.name}' und Länge {len(df_cut_slim)}")
    except Exception:
        df_cut_slim = df_cut  # Fallback: unverändert

    return aligned, df_cut_slim, ordered_ids
    #return aligned, df_cut, ordered_ids


# === Hinweise/Beispiele zur Verwendung (nicht ausführen, nur Referenz) ======
# Einzelmatrix auf DF-Reihenfolge bringen:
# y_df = y_df_0201_optimized
# D_sorted = align_square_matrix_to_df_order(D_weighted_sub_p, y_df, id_col="CustomerCode", return_numpy=False)
# display(len(y_df))
# display(D_sorted.shape)
#
# Mehrere Matrizen gemeinsam ausrichten:
# mats = {
#     "D_weighted_sub": D_weighted_sub_p,
#     "D_naics_sub":    D_naics_sub_p,
#     "D_hs_sub":       D_hs_sub_p,
#     "D_am_sub":       D_am_sub_p,
#     "D_geo_sub":      D_geo_sub_p,
# }
# y_df = y_df_0201_optimized
# aligned_np, y_df_cut, ordered_ids = align_matrices_to_df(
#     mats, y_df, id_col="CustomerCode",
#     matrices_as_numpy=True, set_globals=True
# )
#
# Spätere Sektion: Grafische Auswertungen (folgt in anderer Datei/Sektion).


# === Visualization / Grafische Auswertung ===================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.manifold import TSNE, MDS
from umap import UMAP
import inspect
from pathlib import Path

# ===== feste Farben =====
COL_DATA_0 = 'tab:blue'
COL_DATA_1 = 'tab:orange'
COL_PROTO_0 = 'red'     # negative Prototypen (Label 0)
COL_PROTO_1 = 'green'   # positive Prototypen (Label 1)

# Trajektorien jetzt je nach Label gefärbt
COL_TRAJ_NEG = 'red'
COL_TRAJ_POS = 'green'

COL_MARGIN  = 'black'

# ===== neue Style-Parameter (feiner) =====
S_DATA        = 10    # zuvor 15
ALPHA_DATA    = 0.75

S_PROTO       = 140   # zuvor 180
EDGE_LW       = 0.8   # zuvor 1.2

LW_TRAJ       = 1.2   # zuvor 2.0
S_TRAJ_START  = 40    # zuvor 60
S_TRAJ_END    = 110   # zuvor 140
ALPHA_TRAJ    = 0.85

LW_MARGIN     = 1.2   # zuvor 2.0
LW_LEVELS     = 0.6   # zuvor 1.0
ALPHA_FILL    = 0.06  # zuvor 0.08


# --------- Hilfen ---------
def _filter_kwargs(cls, kwargs):
    """Nur Parameter durchlassen, die der __init__ der Klasse unterstützt."""
    if kwargs is None:
        return {}
    sig = inspect.signature(cls.__init__).parameters
    return {k: v for k, v in kwargs.items() if k in sig}

def _validate_distance_matrix(D: np.ndarray):
    D = np.asarray(D, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("Distance matrix must be square.")
    np.fill_diagonal(D, 0.0)
    if not np.allclose(D, D.T, atol=1e-8, rtol=1e-8):
        raise ValueError("Distance matrix must be symmetric.")
    if (D < -1e-12).any():
        raise ValueError("Distances must be non-negative.")
    return D

def classical_mds_from_distance(D, n_components=2):
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    D2 = D ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    w, V = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1]
    w = w[idx]
    V = V[:, idx]
    L = np.clip(w[:n_components], 0, None)
    Y = V[:, :n_components] * np.sqrt(L)
    return Y

def _check_tsne_perplexity(perplexity, n_samples):
    # Faustregel: 3 * perplexity < n_samples - 1
    if perplexity >= (n_samples - 1) / 3:
        raise ValueError(
            f"t-SNE: 'perplexity'={perplexity} ist zu groß für n={n_samples}. "
            f"Bitte < {(n_samples - 1)/3:.2f} wählen."
        )

# ------ Generic Loader/Saver für 2D-Embeddings ------
# ------ Generic Loader/Saver für 2D-Embeddings (nur .npy) ------
# ------ Generic Loader/Saver für 2D-Embeddings (nur .npy, ohne IDs) ------
def _load_embedding_2d(path, expected_n=None):
    """
    Lädt ein 2D-Embedding aus .npy.
    Optional: expected_n prüft die Zeilenzahl.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Embedding nicht gefunden: {p}")
    if p.suffix != ".npy":
        raise ValueError(f"Nur .npy wird unterstützt, erhalten: {p.suffix}")

    X = np.load(p)
    X = np.asarray(X, float)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(f"Erwartet (n,2), erhalten {X.shape} in '{p}'.")
    if expected_n is not None and X.shape[0] != expected_n:
        raise ValueError(f"n-Mismatch: expected_n={expected_n}, geladen n={X.shape[0]} aus '{p}'.")
    return X

from pathlib import Path
import numpy as np

def _save_embedding_2d(X, path):
    """
    Speichert 2D-Embedding X als .npy (atomar: temp -> rename).
    """
    p = Path(path)
    if p.suffix != ".npy":
        raise ValueError(f"Nur .npy wird unterstützt, erhalten: {p.suffix}")
    p.parent.mkdir(parents=True, exist_ok=True)

    X = np.asarray(X, float)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(f"_save_embedding_2d erwartet (n,2); erhalten {X.shape}.")

    # Temp-Datei MUSS auf .npy enden, damit np.save NICHT nochmal .npy anhängt
    tmp = p.with_name(p.name + ".tmp.npy")
    np.save(tmp, X)        # erzeugt exakt ...tmp.npy
    tmp.replace(p)         # atomar verschieben


# --------- Zeichnen eines Panels ---------
def _draw_panel(ax, X2, y_np, model, title):
    X2   = np.asarray(X2)
    y_np = np.asarray(y_np).ravel()

    # Prototyp-Infos
    widx = np.asarray(getattr(model, "_w"), int)
    yW   = np.asarray(getattr(model, "_y"), int)

    # Trajektorien (_w_history oder get_vweight_path)
    H = None
    if hasattr(model, "_w_history") and model._w_history is not None and len(model._w_history) > 1:
        H = np.asarray(model._w_history, int)
    elif hasattr(model, "get_vweight_path") and callable(model.get_vweight_path):
        try:
            path = model.get_vweight_path()
            if path is not None and len(path) > 1:
                H = np.asarray(path, int)
        except Exception:
            H = None

    # Trajektorien je Prototyp in passender Farbe
    if H is not None and H.ndim == 2 and H.shape[0] > 1:
        for k in range(H.shape[1]):
            path = X2[H[:, k], :]
            col  = COL_TRAJ_POS if (k < len(yW) and yW[k] == 1) else COL_TRAJ_NEG
            ax.plot(path[:, 0], path[:, 1], lw=LW_TRAJ, alpha=ALPHA_TRAJ, color=col)
            ax.scatter(path[0, 0],  path[0, 1],  s=S_TRAJ_START, marker='o', color=col)
            ax.scatter(path[-1, 0], path[-1, 1], s=S_TRAJ_END,   marker='P',
                       edgecolor='k', linewidths=EDGE_LW, color=col)

    # Datenpunkte
    m0 = (y_np == 0); m1 = (y_np == 1)
    if np.any(m0):
        ax.scatter(X2[m0, 0], X2[m0, 1], s=S_DATA, alpha=ALPHA_DATA, label="Label 0", color=COL_DATA_0)
    if np.any(m1):
        ax.scatter(X2[m1, 0], X2[m1, 1], s=S_DATA, alpha=ALPHA_DATA, label="Label 1", color=COL_DATA_1)

    # Prototypen
    W_actual = X2[widx, :]
    if np.any(yW == 0):
        ax.scatter(W_actual[yW == 0, 0], W_actual[yW == 0, 1],
                   s=S_PROTO, marker='P', edgecolor='k', linewidths=EDGE_LW,
                   color=COL_PROTO_0, label="Prototypen (Label 0)")
    if np.any(yW == 1):
        ax.scatter(W_actual[yW == 1, 0], W_actual[yW == 1, 1],
                   s=S_PROTO, marker='P', edgecolor='k', linewidths=EDGE_LW,
                   color=COL_PROTO_1, label="Prototypen (Label 1)")

    # Margin & Füllung (unverändert)
    uniqW = np.unique(yW)
    if len(uniqW) == 2 and W_actual.shape[0] >= 2:
        pad = 0.05 * (X2.max(0) - X2.min(0)).max() + 1e-9
        x_min, x_max = X2[:, 0].min() - pad, X2[:, 0].max() + pad
        y_min, y_max = X2[:, 1].min() - pad, X2[:, 1].max() + pad
        nx = ny = 300
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                             np.linspace(y_min, y_max, ny))
        grid = np.c_[xx.ravel(), yy.ravel()]
        DW = cdist(grid, W_actual, metric='euclidean')
        c0 = np.where(yW == uniqW[0])[0]
        c1 = np.where(yW == uniqW[1])[0]
        if c0.size and c1.size:
            d0 = DW[:, c0].min(1); d1 = DW[:, c1].min(1)
            MU = ((d0 - d1) / (d0 + d1 + 1e-9)).reshape(xx.shape)

            cs = ax.contour(xx, yy, MU, levels=[0.0],
                            linewidths=LW_MARGIN, linestyles='--', colors=[COL_MARGIN])
            if hasattr(cs, "collections") and len(cs.collections) > 0:
                cs.collections[0].set_label("Margin-Grenze (μ=0)")

            ax.contour(xx, yy, MU, levels=[-0.5, -0.25, 0.25, 0.5],
                       linewidths=LW_LEVELS, linestyles=':')
            ax.contourf(xx, yy, MU, levels=np.linspace(-1, 1, 21), alpha=ALPHA_FILL)

    ax.set_title(title)
    ax.set_xlabel(title + " 1")
    ax.set_ylabel(title + " 2")
    ax.legend(loc="best", fontsize=9)

# --------- Hauptfunktion (mit Lade-/Speicher-Option für MDS) ---------
def visualize_from_distance_and_model(
    model,
    D,
    y,
    *,
    tsne_params=None,
    umap_params=None,
    use_classical_mds=True,
    figsize=(18, 6),
    random_state=42,
    # --- MDS I/O ---
    mds_load_path=None,
    mds_save_path=None,
    # --- t-SNE I/O ---
    tsne_load_path=None,
    tsne_save_path=None,
    # --- UMAP I/O ---
    umap_load_path=None,
    umap_save_path=None
):
    # --- Validierung ---
    D = _validate_distance_matrix(D)
    y = np.asarray(y).ravel().astype(int)
    n = D.shape[0]

    # --- MDS ---
    X_mds = None
    if mds_load_path is not None and Path(mds_load_path).exists():
        X_mds = _load_embedding_2d(mds_load_path, expected_n=n)
    if X_mds is None:
        if use_classical_mds:
            X_mds = classical_mds_from_distance(D, n_components=2)
        else:
            mds_params = dict(n_components=2, dissimilarity="precomputed", random_state=random_state)
            mds_params = _filter_kwargs(MDS, mds_params)
            X_mds = MDS(**mds_params).fit_transform(D)
        if mds_save_path is not None:
            _save_embedding_2d(X_mds, mds_save_path)

    # --- t-SNE ---
    X_tsne = None
    if tsne_load_path is not None and Path(tsne_load_path).exists():
        X_tsne = _load_embedding_2d(tsne_load_path, expected_n=n)
    if X_tsne is None:
        tsp = dict(tsne_params or {})
        tsp.setdefault('n_components', 2)
        tsp.setdefault('perplexity', 45)
        tsp.setdefault('learning_rate', 100)
        tsp.setdefault('random_state', random_state)
        tsp.setdefault('n_iter', 1000)
        tsp.setdefault('early_exaggeration', 12.0)
        tsp.setdefault('init', 'random')
        tsp['metric'] = 'precomputed'
        tsp.setdefault('square_distances', True)
        tsp.setdefault('verbose', 0)
        tsp.setdefault('n_jobs', -1)

        _check_tsne_perplexity(tsp['perplexity'], n_samples=n)
        tsp = _filter_kwargs(TSNE, tsp)
        X_tsne = TSNE(**tsp).fit_transform(D)
        if tsne_save_path is not None:
            _save_embedding_2d(X_tsne, tsne_save_path)

    # --- UMAP ---
    X_umap = None
    if umap_load_path is not None and Path(umap_load_path).exists():
        X_umap = _load_embedding_2d(umap_load_path, expected_n=n)
    if X_umap is None:
        ump = dict(umap_params or {})
        ump.setdefault('n_components', 2)
        ump.setdefault('n_neighbors', 40)
        ump.setdefault('min_dist', 0.15)
        ump.setdefault('random_state', random_state)
        ump.setdefault('init', 'random')
        ump['metric'] = 'precomputed'

        ump = _filter_kwargs(UMAP, ump)
        X_umap = UMAP(**ump).fit_transform(D)
        if umap_save_path is not None:
            _save_embedding_2d(X_umap, umap_save_path)

    # --- Plot wie gehabt ---
    fig, (ax_mds, ax_tsne, ax_umap) = plt.subplots(1, 3, figsize=figsize)
    _draw_panel(ax_mds,  X_mds,  y, model, "MDS")
    _draw_panel(ax_tsne, X_tsne, y, model, "t-SNE")
    _draw_panel(ax_umap, X_umap, y, model, "UMAP")
    plt.tight_layout(); plt.show()

    embeddings = {'MDS': X_mds, 't-SNE': X_tsne, 'UMAP': X_umap}
    return fig, (ax_mds, ax_tsne, ax_umap), embeddings



# === Training-Übersicht (Gewichte/Loss) =====================================
import numpy as np
import matplotlib.pyplot as plt

def _moving_average(x, w):
    if w is None or w <= 1:
        return x
    w = int(max(1, w))
    ker = np.ones(w, dtype=float) / w
    ma = np.convolve(x, ker, mode='valid')
    pad = np.full(w-1, np.nan)
    return np.concatenate([pad, ma])

def plot_training_overview(
    model,
    *,
    select_v=None,
    normalize_v=False,
    smooth_v=None,
    downsample_v=None,
    figsize=(12,7),
    show=True,          # zeigt die Grafik
    return_fig=False    # gibt die Figure zurück
):
    # === Daten holen ===
    V_path = None
    if hasattr(model, "get_vweight_path") and callable(model.get_vweight_path):
        try:
            V_path = model.get_vweight_path()
        except Exception:
            V_path = None
    if V_path is None and hasattr(model, "_v_path"):
        V_path = getattr(model, "_v_path", None)

    loss = getattr(model, "_loss", None)
    has_loss = loss is not None and len(np.asarray(loss).ravel()) > 0
    has_vpath = V_path is not None and np.asarray(V_path).ndim == 2

    # === Figure-Layout ===
    if has_vpath and has_loss:
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
        show_weights = True
        show_loss_ax = True
    elif has_vpath and not has_loss:
        fig, ax_top = plt.subplots(1, 1, figsize=(figsize[0], max(3.0, figsize[1]/2)))
        ax_bot = None
        show_weights = True
        show_loss_ax = False
    elif has_loss and not has_vpath:
        fig, ax_bot = plt.subplots(1, 1, figsize=(figsize[0], max(3.0, figsize[1]/2)))
        ax_top = None
        show_weights = False
        show_loss_ax = True
    else:
        # Weder Gewichte noch Loss
        fig, ax_empty = plt.subplots(1, 1, figsize=(figsize[0], max(3.0, figsize[1]/2)))
        ax_empty.text(0.5, 0.5, "Keine Daten (weder Gewichte noch Loss) gefunden",
                      ha="center", va="center", fontsize=12)
        ax_empty.axis("off")
        plt.tight_layout()
        if show:
            plt.show()
            if return_fig:
                plt.close(fig)
        return fig if return_fig else None

    # === Oben: Distanzgewichte ===
    if show_weights:
        V_path = np.asarray(V_path, dtype=float)
        steps = V_path.shape[0]
        idx = np.arange(steps)

        # Auswahl
        if select_v is not None:
            V_plot = V_path[:, select_v]
            if isinstance(select_v, (list, tuple, np.ndarray)):
                labels_idx = list(select_v)
            elif isinstance(select_v, slice):
                labels_idx = list(range(*select_v.indices(V_path.shape[1])))
            else:
                labels_idx = [select_v]
        else:
            V_plot = V_path
            labels_idx = list(range(V_path.shape[1]))

        # Downsample
        if downsample_v and downsample_v > 1:
            idx = idx[::downsample_v]
            V_plot = V_plot[::downsample_v, :]

        # Normalisierung
        if normalize_v:
            mn = np.nanmin(V_plot, axis=0)
            mx = np.nanmax(V_plot, axis=0)
            span = np.where((mx - mn) == 0, 1.0, (mx - mn))
            V_plot = (V_plot - mn) / span

        # Glätten
        if smooth_v and smooth_v > 1:
            V_plot = np.column_stack([_moving_average(V_plot[:, j], smooth_v) for j in range(V_plot.shape[1])])

        # Plot
        for j in range(V_plot.shape[1]):
            (ax_top if has_loss else ax_top).plot(idx, V_plot[:, j], lw=1.5, label=f"v{labels_idx[j]}")
        ax = ax_top  # Alias
        ax.set_title("Verlauf der Distanzgewichte")
        ax.set_xlabel("Step")
        ax.set_ylabel("Gewicht" + (" (norm.)" if normalize_v else ""))
        if V_plot.shape[1] <= 12:
            ax.legend(frameon=False)
        else:
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)

    # === Unten: Loss ===
    if show_loss_ax:
        loss = np.asarray(loss, dtype=float).ravel()
        ax = ax_bot  # Alias
        ax.plot(np.arange(len(loss)), loss, lw=1.5, marker='o', ms=3, markevery=max(1, len(loss)//50))
        ax.set_title("Loss-Verlauf während des Trainings")
        ax.set_xlabel("Update-Index")
        ax.set_ylabel("GLVQ-Loss")
        ax.grid(True, alpha=0.3)

    # === Render/Return ohne Doppelanzeige ===
    plt.tight_layout()
    if show:
        plt.show()
        if return_fig:
            plt.close(fig)   # verhindert Doppel-Render in Jupyter

    return fig if return_fig else None
