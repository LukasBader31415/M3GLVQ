# functions.py
# -*- coding: utf-8 -*-
"""
Utility functions for:
- aligning pairwise dissimilarity/similarity matrices to a DataFrame,
-  visualizing prototype-based models from dissimilarity matrices,
- cross-validated grid search and Bayesian optimization for GLVQ-like models.

Public API:
    align_square_matrix_to_df_order
    align_matrices_to_df
    visualize_from_dissimilarity_and_model
    stratified_kfold_grid_search
    stratified_kfold_bayes_search
"""
from __future__ import annotations
from typing import Any, Dict, Tuple, List, Union, Any, Optional, Type
from pathlib import Path
import pickle
import itertools
import inspect


import numpy as np
import pandas as pd
import optuna
from optuna.trial import TrialState

from scipy.spatial.distance import cdist
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.manifold import TSNE, MDS
from umap import UMAP

import matplotlib.pyplot as plt


# =============================================================================
# Basic alignment helpers
# =============================================================================


def align_square_matrix_to_df_order(
    D_df: pd.DataFrame,
    df: pd.DataFrame,
    id_col: str,
    return_numpy: bool = False,
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Reorder a square pairwise matrix D_df (typically dissimilarity or similarity; index=IDs, columns=IDs)
    to match the order of df[id_col].

    All IDs are cast to str to avoid type mismatches (e.g. "1000013" vs 1000013).
    Raises an error if required IDs are missing or NaNs appear after reindexing.

    Parameters
    ----------
    D_df : pd.DataFrame
        Square pairwise matrix with identical ID sets in index and columns.
    df : pd.DataFrame
        DataFrame whose column `id_col` defines the desired order.
    id_col : str
        Column name in `df` that contains the IDs.
    return_numpy : bool, default False
        If True, returns a NumPy array; otherwise a DataFrame.

    Returns
    -------
    pd.DataFrame or np.ndarray
        Reordered matrix.
    """
    D_df = D_df.copy()
    D_df.index = D_df.index.astype(str)
    D_df.columns = D_df.columns.astype(str)
    df_ids = df[id_col].astype(str)

    # Consistency checks
    if set(D_df.index) != set(D_df.columns):
        raise ValueError("Index and columns of D_df contain different ID sets.")

    missing = set(df_ids) - set(D_df.index)
    if missing:
        ex = ", ".join(list(missing)[:5])
        more = " ..." if len(missing) > 5 else ""
        raise ValueError(f"IDs from df are missing in D_df: {ex}{more}")

    # Reindex in df order
    D_sorted = D_df.loc[df_ids, df_ids]

    # Optional symmetry check (typical for dissimilarity matrices)
    if not np.allclose(D_sorted.values, D_sorted.values.T, equal_nan=True):
        print("Warning: D_sorted is not exactly symmetric.")

    # NaN check
    if D_sorted.isna().any().any():
        nmiss = int(D_sorted.isna().sum().sum())
        raise ValueError(f"Matrix contains {nmiss} NaN values after reindexing.")

    return D_sorted.values if return_numpy else D_sorted


def align_matrices_to_df(
    mats: Dict[str, pd.DataFrame],
    df: pd.DataFrame,
    id_col: str = "CustomerCode",
    set_diag_zero: bool = True,
    set_globals: bool = False,
    verbose: bool = True,
    matrices_as_numpy: bool = True,
) -> Tuple[Dict[str, Union[np.ndarray, pd.DataFrame]], pd.Series, List]:
    """
    Align multiple square pairwise matrices to the same ID subset & order defined by df[id_col].

    Steps:
      1) Compute the intersection of IDs over all matrix indices and columns.
      2) Restrict df to this intersection, preserving the original order
         (first occurrence of each ID).
      3) Reindex each matrix to this order (rows and columns).
      4) Optionally:
           - set the diagonal to zero,
           - convert matrices to NumPy arrays,
           - store aligned matrices in globals().

    Parameters
    ----------
    mats : dict[str, pd.DataFrame]
        Dictionary of square matrices, all with same ID space (index & columns).
    df : pd.DataFrame
        DataFrame that contains an ID column to define the target order.
    id_col : str, default "CustomerCode"
        Column name in `df` with IDs.
    set_diag_zero : bool, default True
        If True, sets diagonal entries of the aligned matrices to 0.
    set_globals : bool, default False
        If True, store each aligned matrix in globals() under "<name>_p_aligned".
    verbose : bool, default True
        If True, print short progress/info messages.
    matrices_as_numpy : bool, default True
        If True, return matrices as NumPy arrays; otherwise as DataFrames.

    Returns
    -------
    aligned : dict[str, np.ndarray or pd.DataFrame]
        Aligned matrices.
    df_cut_slim : pd.Series
        A slim series taken from df_cut (typically the 2nd column as a “label”).
    ordered_ids : list
        List of IDs in the final order.
    """
    if not mats:
        raise ValueError("mats is empty.")
    if id_col not in df.columns:
        raise KeyError(f"Column '{id_col}' not found in DataFrame.")

    # 1) Intersection of IDs over all matrices
    idx_sets = [set(M.index) for M in mats.values()]
    col_sets = [set(M.columns) for M in mats.values()]
    common_ids = set.intersection(*idx_sets) & set.intersection(*col_sets)
    if verbose:
        print(f"[1] Common ID set across all matrices: {len(common_ids)} IDs")

    # 2) Restrict df to common IDs and preserve original order (first occurrence)
    df_cut = df[df[id_col].isin(common_ids)].copy()
    seen = set()
    ordered_ids = [
        x for x in df_cut[id_col].tolist()
        if not (x in seen or seen.add(x))
    ]
    if verbose:
        print(
            f"[2] df restricted to common IDs: {len(df_cut)} rows | "
            f"unique IDs in df order: {len(ordered_ids)}"
        )

    # 3) Reindex each matrix
    aligned: Dict[str, Union[np.ndarray, pd.DataFrame]] = {}
    n = len(ordered_ids)
    for name, M in mats.items():
        A_df = M.reindex(index=ordered_ids, columns=ordered_ids)
        if set_diag_zero and n > 0:
            A_df.values[np.arange(n), np.arange(n)] = 0.0

        if matrices_as_numpy:
            A = A_df.to_numpy(copy=True)
        else:
            A = A_df

        aligned[name] = A

        if set_globals:
            globals()[f"{name}_p_aligned"] = A

        if verbose:
            shape = A.shape
            typ = "ndarray" if matrices_as_numpy else "DataFrame"
            print(f"[3] {name}_p_aligned -> {shape} ({typ})")

    # 4) Try to return a slim series from df_cut (2nd column if available)
    try:
        if df_cut.shape[1] >= 2:
            df_cut_slim = df_cut[df_cut.columns[1]]
        else:
            df_cut_slim = df_cut.iloc[:, 0]
        if verbose:
            print(
                f"[4] df_cut -> Series '{df_cut_slim.name}' "
                f"with length {len(df_cut_slim)}"
            )
    except Exception:
        df_cut_slim = df_cut.iloc[:, 0]

    return aligned, df_cut_slim, ordered_ids


# =============================================================================
# Embedding & visualization from dissimilarity matrix
# =============================================================================


def _filter_kwargs(cls, kwargs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Filter a kwargs dict so that only arguments supported by cls.__init__ remain."""
    if kwargs is None:
        return {}
    sig = __import__("inspect").signature(cls.__init__).parameters
    return {k: v for k, v in kwargs.items() if k in sig}


def _validate_dissimilarity_matrix(D: np.ndarray) -> np.ndarray:
    """
    Validate a dissimilarity matrix (distance-like, but not necessarily metric):
      - must be square,
      - symmetric,
      - non-negative,
      - diagonal set to zero.
    Note: no assumption of metric properties (triangle inequality etc.).
    """
    D = np.asarray(D, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("Dissimilarity matrix must be square (n x n).")
    np.fill_diagonal(D, 0.0)
    if not np.allclose(D, D.T, atol=1e-8, rtol=1e-8):
        raise ValueError("Dissimilarity matrix must be symmetric.")
    if (D < -1e-12).any():
        raise ValueError("Dissimilarities must be non-negative.")
    return D


def classical_mds_from_dissimilarity(D: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Classical MDS (Torgerson) embedding from a dissimilarity matrix.

    Parameters
    ----------
    D : np.ndarray
        Dissimilarity matrix (n x n).
    n_components : int, default 2
        Number of output dimensions.

    Returns
    -------
    np.ndarray
        Embedding of shape (n, n_components).
    """
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


def _check_tsne_perplexity(perplexity: float, n_samples: int) -> None:
    """
    Simple sanity check for t-SNE perplexity:
    rule of thumb: 3 * perplexity < n_samples - 1
    """
    if perplexity >= (n_samples - 1) / 3:
        raise ValueError(
            f"t-SNE: perplexity={perplexity} is too large for n={n_samples}. "
            f"Use < {(n_samples - 1) / 3:.2f}."
        )


def _load_embedding_2d(path: Union[str, Path], expected_n: Optional[int] = None) -> np.ndarray:
    """
    Load a 2D embedding stored as .npy.

    Parameters
    ----------
    path : str or Path
        File path to .npy file.
    expected_n : int, optional
        If given, assert that the number of rows matches expected_n.

    Returns
    -------
    np.ndarray
        Embedding (n, 2).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Embedding not found: {p}")
    if p.suffix != ".npy":
        raise ValueError(f"Only .npy is supported, got: {p.suffix}")

    X = np.load(p)
    X = np.asarray(X, float)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(f"Expected shape (n, 2), got {X.shape} in '{p}'.")
    if expected_n is not None and X.shape[0] != expected_n:
        raise ValueError(
            f"Row count mismatch: expected_n={expected_n}, loaded n={X.shape[0]} from '{p}'."
        )
    return X


def _save_embedding_2d(X: np.ndarray, path: Union[str, Path]) -> None:
    """
    Save a 2D embedding as .npy (atomic write via temporary file).
    """
    p = Path(path)
    if p.suffix != ".npy":
        raise ValueError(f"Only .npy is supported, got: {p.suffix}")
    p.parent.mkdir(parents=True, exist_ok=True)

    X = np.asarray(X, float)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(f"_save_embedding_2d expects shape (n, 2); got {X.shape}.")

    tmp = p.with_name(p.name + ".tmp.npy")
    np.save(tmp, X)
    tmp.replace(p)


# --- Styling constants for visualization ---
COL_DATA_0 = "tab:blue"
COL_DATA_1 = "tab:orange"
COL_PROTO_0 = "red"     # prototypes of class 0
COL_PROTO_1 = "green"   # prototypes of class 1
COL_TRAJ_NEG = "red"
COL_TRAJ_POS = "green"
COL_MARGIN = "black"

S_DATA = 10
ALPHA_DATA = 0.75
S_PROTO = 140
EDGE_LW = 0.8
LW_TRAJ = 1.2
S_TRAJ_START = 40
S_TRAJ_END = 110
ALPHA_TRAJ = 0.85
LW_MARGIN = 1.2
LW_LEVELS = 0.6
ALPHA_FILL = 0.06


def _draw_panel(
    ax,
    X2: np.ndarray,
    y_np: np.ndarray,
    model,
    title: str = "",
    show_trajectories: bool = True,   # <-- NEW
) -> None:
    """
    Draw a single 2D panel with:
      - data points (class 0 and 1),
      - prototypes,
      - (optional) prototype trajectories, if provided by the model,
      - a distance-based (in embedding space) “margin” contour between prototype sets.
    """
    X2 = np.asarray(X2)
    y_np = np.asarray(y_np).ravel()

    # Prototypes
    widx = np.asarray(getattr(model, "_w"), int)
    yW = np.asarray(getattr(model, "_y"), int)

    # Prototype index history (trajectories)
    H = None
    if hasattr(model, "_w_history") and model._w_history is not None and len(model._w_history) > 1:
        H = np.asarray(model._w_history, int)
    elif hasattr(model, "get_vweight_path") and callable(model.get_vweight_path):
        # some models re-use this method name
        try:
            path = model.get_vweight_path()
            if path is not None and len(path) > 1:
                H = np.asarray(path, int)
        except Exception:
            H = None

    # Trajectories (only if requested)
    if show_trajectories and H is not None and H.ndim == 2 and H.shape[0] > 1:
        for k in range(H.shape[1]):
            path = X2[H[:, k], :]
            col = COL_TRAJ_POS if (k < len(yW) and yW[k] == 1) else COL_TRAJ_NEG
            ax.plot(path[:, 0], path[:, 1], lw=LW_TRAJ, alpha=ALPHA_TRAJ, color=col)
            ax.scatter(path[0, 0], path[0, 1], s=S_TRAJ_START, marker="o", color=col)
            ax.scatter(
                path[-1, 0],
                path[-1, 1],
                s=S_TRAJ_END,
                marker="P",
                edgecolor="k",
                linewidths=EDGE_LW,
                color=col,
            )

    # Data points
    m0 = y_np == 0
    m1 = y_np == 1
    if np.any(m0):
        ax.scatter(
            X2[m0, 0],
            X2[m0, 1],
            s=S_DATA,
            alpha=ALPHA_DATA,
            label="Class 0",
            color=COL_DATA_0,
        )
    if np.any(m1):
        ax.scatter(
            X2[m1, 0],
            X2[m1, 1],
            s=S_DATA,
            alpha=ALPHA_DATA,
            label="Class 1",
            color=COL_DATA_1,
        )

    # Prototypes
    W_actual = X2[widx, :]
    if np.any(yW == 0):
        ax.scatter(
            W_actual[yW == 0, 0],
            W_actual[yW == 0, 1],
            s=S_PROTO,
            marker="P",
            edgecolor="k",
            linewidths=EDGE_LW,
            color=COL_PROTO_0,
            label="Prototype (Class 0)",
        )
    if np.any(yW == 1):
        ax.scatter(
            W_actual[yW == 1, 0],
            W_actual[yW == 1, 1],
            s=S_PROTO,
            marker="P",
            edgecolor="k",
            linewidths=EDGE_LW,
            color=COL_PROTO_1,
            label="Prototype (Class 1)",
        )

    # Margin and soft levels (two prototype classes only)
    uniqW = np.unique(yW)
    if len(uniqW) == 2 and W_actual.shape[0] >= 2:
        pad = 0.05 * (X2.max(0) - X2.min(0)).max() + 1e-9
        x_min, x_max = X2[:, 0].min() - pad, X2[:, 0].max() + pad
        y_min, y_max = X2[:, 1].min() - pad, X2[:, 1].max() + pad
        nx = ny = 300
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, nx),
            np.linspace(y_min, y_max, ny),
        )
        grid = np.c_[xx.ravel(), yy.ravel()]
        # Euclidean distances in the 2D embedding space (this is a true metric here)
        DW = cdist(grid, W_actual, metric="euclidean")
        c0 = np.where(yW == uniqW[0])[0]
        c1 = np.where(yW == uniqW[1])[0]
        if c0.size and c1.size:
            d0 = DW[:, c0].min(1)
            d1 = DW[:, c1].min(1)
            MU = ((d0 - d1) / (d0 + d1 + 1e-9)).reshape(xx.shape)

            cs = ax.contour(
                xx,
                yy,
                MU,
                levels=[0.0],
                linewidths=LW_MARGIN,
                linestyles="--",
                colors=[COL_MARGIN],
            )
            if hasattr(cs, "collections") and len(cs.collections) > 0:
                cs.collections[0].set_label("Margin boundary (μ = 0)")

            ax.contour(
                xx,
                yy,
                MU,
                levels=[-0.5, -0.25, 0.25, 0.5],
                linewidths=LW_LEVELS,
                linestyles=":",
            )
            ax.contourf(
                xx,
                yy,
                MU,
                levels=np.linspace(-1, 1, 21),
                alpha=ALPHA_FILL,
            )

    ax.set_title(title)
    ax.set_xlabel(f"{title} 1")
    ax.set_ylabel(f"{title} 2")
    ax.legend(loc="best", fontsize=9)



def visualize_from_dissimilarity_and_model(
    model,
    D: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series, List[int]],
    *,
    tsne_params: Optional[Dict[str, Any]] = None,
    umap_params: Optional[Dict[str, Any]] = None,
    use_classical_mds: bool = True,
    figsize=(18, 6),
    random_state: int = 42,
    show_trajectories: bool = True,  # <-- NEW
    # MDS I/O
    mds_load_path: Optional[Union[str, Path]] = None,
    mds_save_path: Optional[Union[str, Path]] = None,
    # t-SNE I/O
    tsne_load_path: Optional[Union[str, Path]] = None,
    tsne_save_path: Optional[Union[str, Path]] = None,
    # UMAP I/O
    umap_load_path: Optional[Union[str, Path]] = None,
    umap_save_path: Optional[Union[str, Path]] = None,
):
    """
    Compute 2D embeddings (MDS, t-SNE, UMAP) from a dissimilarity matrix and
    visualize a prototype-based model in all three spaces.

    Parameters
    ----------
    model : object
        Trained GLVQ-like model with attributes `_w` (prototype indices) and
        `_y` (prototype labels 0/1). Optionally `_w_history` or
        `get_vweight_path()` for trajectories.
    D : array-like (n x n) or DataFrame
        Dissimilarity matrix (distance-like; not necessarily metric).
    y : array-like of shape (n,)
        Binary labels (0/1).
    tsne_params, umap_params : dict, optional
        Additional keyword arguments passed to TSNE/UMAP (filtered to valid keys).
    use_classical_mds : bool, default True
        If True, use classical MDS; otherwise sklearn MDS.
    figsize : tuple, default (18, 6)
        Matplotlib figure size.
    random_state : int, default 42
        Random seed for stochastic embeddings.
    mds_load_path, tsne_load_path, umap_load_path : str or Path, optional
        If given and file exists, load precomputed embeddings from .npy.
    mds_save_path, tsne_save_path, umap_save_path : str or Path, optional
        If given, save computed embeddings as .npy.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : tuple of Axes
    embeddings : dict
        {"MDS": X_mds, "t-SNE": X_tsne, "UMAP": X_umap}
    """
    D = _validate_dissimilarity_matrix(D)
    y = np.asarray(y).ravel().astype(int)
    n = D.shape[0]

    # --- MDS ---
    X_mds = None
    if mds_load_path is not None and Path(mds_load_path).exists():
        X_mds = _load_embedding_2d(mds_load_path, expected_n=n)
    if X_mds is None:
        if use_classical_mds:
            X_mds = classical_mds_from_dissimilarity(D, n_components=2)
        else:
            mds_params = dict(
                n_components=2,
                dissimilarity="precomputed",
                random_state=random_state,
            )
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
        tsp.setdefault("n_components", 2)
        tsp.setdefault("perplexity", 45)
        tsp.setdefault("learning_rate", 100)
        tsp.setdefault("random_state", random_state)
        tsp.setdefault("n_iter", 1000)
        tsp.setdefault("early_exaggeration", 12.0)
        tsp.setdefault("init", "random")
        tsp["metric"] = "precomputed"  # expects a dissimilarity matrix
        tsp.setdefault("square_distances", True)
        tsp.setdefault("verbose", 0)
        tsp.setdefault("n_jobs", -1)

        _check_tsne_perplexity(tsp["perplexity"], n_samples=n)
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
        ump.setdefault("n_components", 2)
        ump.setdefault("n_neighbors", 40)
        ump.setdefault("min_dist", 0.15)
        ump.setdefault("random_state", random_state)
        ump.setdefault("init", "random")
        ump["metric"] = "precomputed"  # expects a dissimilarity matrix
        ump = _filter_kwargs(UMAP, ump)
        X_umap = UMAP(**ump).fit_transform(D)
        if umap_save_path is not None:
            _save_embedding_2d(X_umap, umap_save_path)

    # --- Plot ---
    fig, (ax_mds, ax_tsne, ax_umap) = plt.subplots(1, 3, figsize=figsize)
    _draw_panel(ax_mds,  X_mds,  y, model, "MDS",  show_trajectories=show_trajectories)
    _draw_panel(ax_tsne, X_tsne, y, model, "t-SNE", show_trajectories=show_trajectories)
    _draw_panel(ax_umap, X_umap, y, model, "UMAP", show_trajectories=show_trajectories)
    plt.tight_layout()
    plt.show()

    embeddings = {"MDS": X_mds, "t-SNE": X_tsne, "UMAP": X_umap}
    return fig, (ax_mds, ax_tsne, ax_umap), embeddings


# =============================================================================
# Cross-validation and model selection (Grid Search / Bayesian Search)
# =============================================================================


def _confusion_metrics(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
    as_percent: bool = True,
) -> Dict[str, float]:
    """
    Simple binary confusion metrics:
    - TP, FP, TN, FN
    - recall (TPR), specificity (TNR), precision, f1
    - balanced_accuracy = (recall + specificity) / 2

    If as_percent=True, TP/FP/TN/FN are returned in percent of total samples.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()

    if y_true.size == 0:
        raise ValueError("y_true must not be empty.")

    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if cm.shape != (2, 2):
        # fallback when only a single class is present
        tn = fp = fn = tp = 0
        if np.all(y_true == 0):
            tn = int(np.sum(y_pred == 0))
            fp = int(np.sum(y_pred == 1))
        elif np.all(y_true == 1):
            tp = int(np.sum(y_pred == 1))
            fn = int(np.sum(y_pred == 0))
    else:
        tn, fp, fn, tp = cm.ravel()

    total = max(1, len(y_true))
    if as_percent:
        TP = 100.0 * tp / total
        FP = 100.0 * fp / total
        TN = 100.0 * tn / total
        FN = 100.0 * fn / total
    else:
        TP, FP, TN, FN = float(tp), float(fp), float(tn), float(fn)

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    if (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    balanced_acc = 0.5 * (recall + specificity)

    return {
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "balanced_accuracy": balanced_acc,
    }


def _slice_dissimilarity_like(
    D: Union[pd.DataFrame, np.ndarray, List[np.ndarray]],
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
):
    """
    Slice a dissimilarity structure into Train-Train and Val-Train parts.

    Supported inputs:
      - single dissimilarity matrix as DataFrame (n x n)
      - single dissimilarity matrix as ndarray (n x n)
      - list/tuple of dissimilarity matrices [D1, D2, ...], each (n x n)
      - 3D tensor (n x n x p), treated as channels

    Returns
    -------
    D_tr, D_va_tr
        Same structure as input (matrix, list of matrices, or tensor).
    """
    if isinstance(D, pd.DataFrame):
        D_tr = D.iloc[tr_idx, tr_idx].to_numpy()
        D_va_tr = D.iloc[va_idx, tr_idx].to_numpy()
        return D_tr, D_va_tr

    if isinstance(D, (list, tuple)):
        D_tr_list = []
        D_va_tr_list = []
        for M in D:
            M_arr = np.asarray(M)
            if M_arr.ndim != 2 or M_arr.shape[0] != M_arr.shape[1]:
                raise ValueError("Each element in D must be a square matrix.")
            D_tr_list.append(M_arr[np.ix_(tr_idx, tr_idx)])
            D_va_tr_list.append(M_arr[np.ix_(va_idx, tr_idx)])
        return D_tr_list, D_va_tr_list

    D_arr = np.asarray(D)
    if D_arr.ndim == 2:
        if D_arr.shape[0] != D_arr.shape[1]:
            raise ValueError("Dissimilarity matrix must be square.")
        D_tr = D_arr[np.ix_(tr_idx, tr_idx)]
        D_va_tr = D_arr[np.ix_(va_idx, tr_idx)]
        return D_tr, D_va_tr

    if D_arr.ndim == 3:
        # assume axes: (n, n, p)
        D_tr = D_arr[tr_idx][:, tr_idx, :]
        D_va_tr = D_arr[va_idx][:, tr_idx, :]
        return D_tr, D_va_tr

    raise ValueError(f"Unsupported type/shape for D: type={type(D)}, ndim={D_arr.ndim}")


def _get_final_vweights_from_model(model) -> Optional[np.ndarray]:
    """
    Try to read final v-weights from a GLVQ-like model.

    Order of preference:
      1) model.get_vweight_path() -> last row
      2) attribute model._vWeights

    Returns
    -------
    np.ndarray or None
    """
    if hasattr(model, "get_vweight_path"):
        try:
            vpath = model.get_vweight_path()
        except Exception:
            vpath = None

        if vpath is not None:
            vpath = np.asarray(vpath)
            if vpath.ndim >= 1 and vpath.size > 0:
                return vpath[-1]

    if hasattr(model, "_vWeights"):
        v = getattr(model, "_vWeights")
        if v is not None:
            return np.asarray(v, dtype=float)

    return None


def stratified_kfold_full_metrics(
    D,
    y,
    model_cls: Type,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
    model_params: Optional[Dict[str, Any]] = None,
    conf_as_percent: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Stratified K-fold cross-validation for GLVQ-like models with a pre-computed
    dissimilarity/kernel-like structure D.

    Adds per-fold prototype export (for M3GLVQ / GLVQ-like models):
      - final_prototypes: list[int] / np.ndarray of prototype indices (e.g. clf._w)
      - prototype_labels: list[int] / np.ndarray of labels per prototype (e.g. clf._y)

    Parameters
    ----------
    D : dissimilarity structure
        See `_slice_dissimilarity_like` for supported formats.
    y : array-like
        Binary labels (0/1).
    model_cls : Type
        Model class (e.g. MGLVQ, vMGLVQ_V1, vMGLVQ_V2, M3GLVQ...).
    model_params : dict, optional
        Initialization parameters for model_cls.
    conf_as_percent : bool, default True
        If True, TP/FP/TN/FN are returned in percent.

    Returns
    -------
    dict
        {
          "folds": [...],         # per-fold metrics, final v-weights, prototypes
          "averages": {...},      # averaged metrics over folds
          "n_splits": int,
          "model_cls": model_cls,
          "model_params": dict,
        }
    """
    if model_params is None:
        model_params = {}

    model_params = _filter_kwargs(model_cls, model_params)

    y = np.asarray(y).astype(int).ravel()
    n = len(y)
    if n == 0:
        raise ValueError("y must not be empty.")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    fold_results = []
    for f, (tr_idx, va_idx) in enumerate(skf.split(np.arange(n), y), start=1):
        D_tr, D_va_tr = _slice_dissimilarity_like(D, tr_idx, va_idx)

        y_tr = y[tr_idx]
        y_va = y[va_idx]

        tr_pos = int(y_tr.sum())
        tr_n = len(y_tr)
        va_pos = int(y_va.sum())
        va_n = len(y_va)
        tr_pos_rate = tr_pos / tr_n if tr_n else 0.0
        va_pos_rate = va_pos / va_n if va_n else 0.0

        clf = model_cls(**model_params)
        clf.fit(D_tr, y_tr)
        y_pred = clf.predict(D_va_tr)

        metrics = _confusion_metrics(y_va, y_pred, as_percent=conf_as_percent)
        metrics["balanced_accuracy_sklearn"] = balanced_accuracy_score(y_va, y_pred)
        metrics["f1_macro"] = f1_score(y_va, y_pred, average="macro")

        final_vweights = _get_final_vweights_from_model(clf)

        # -------- NEW: export prototypes + their labels (if available) --------
        final_prototypes = None
        prototype_labels = None

        if hasattr(clf, "_w") and getattr(clf, "_w") is not None:
            try:
                final_prototypes = np.asarray(getattr(clf, "_w"), dtype=int).ravel()
            except Exception:
                final_prototypes = None

        if hasattr(clf, "_y") and getattr(clf, "_y") is not None:
            try:
                prototype_labels = np.asarray(getattr(clf, "_y"), dtype=int).ravel()
            except Exception:
                prototype_labels = None

        # If only one of them exists or lengths mismatch -> discard to avoid corrupt table rows
        if final_prototypes is None or prototype_labels is None:
            final_prototypes = None
            prototype_labels = None
        elif final_prototypes.shape[0] != prototype_labels.shape[0]:
            final_prototypes = None
            prototype_labels = None

        res = {
            "fold": f,
            "train_pos_count": tr_pos,
            "train_total": tr_n,
            "train_pos_rate": tr_pos_rate,
            "val_pos_count": va_pos,
            "val_total": va_n,
            "val_pos_rate": va_pos_rate,
            **metrics,
        }

        if final_vweights is not None:
            res["final_vweights"] = final_vweights

        # NEW: attach prototypes + labels to fold result
        if final_prototypes is not None and prototype_labels is not None:
            res["final_prototypes"] = final_prototypes
            res["prototype_labels"] = prototype_labels

        fold_results.append(res)

        if verbose:
            ba = metrics["balanced_accuracy"]
            proto_info = ""
            if final_prototypes is not None:
                proto_info = f" | protos={final_prototypes.size}"
            print(
                f"[Fold {f}] bal_acc={ba:.4f}{proto_info} | "
                f"pos_rate train={tr_pos_rate:.3f}, val={va_pos_rate:.3f}"
            )

    keys_to_avg = [
        "TP",
        "FP",
        "TN",
        "FN",
        "recall",
        "specificity",
        "precision",
        "f1",
        "balanced_accuracy",
        "balanced_accuracy_sklearn",
        "f1_macro",
        "train_pos_rate",
        "val_pos_rate",
    ]
    avg = {k: float(np.mean([fr[k] for fr in fold_results])) for k in keys_to_avg}
    
    #ALT
    # return {
    #     "folds": fold_results,
    #     "averages": avg,
    #     "n_splits": n_splits,
    #     "model_cls": model_cls,
    #     "model_params": model_params,
    # }
    #NEU
    model_cls_str = f"{model_cls.__module__}.{model_cls.__qualname__}"
    return {
            "folds": fold_results,
            "averages": avg,
            "n_splits": n_splits,
            "model_cls": model_cls_str,
            "model_params": model_params,
        }




def generate_K_combinations(label_to_values: Dict[int, List[int]]) -> List[Dict[int, int]]:
    """
    Generate all combinations of K-values per label.

    Example
    -------
    label_to_values = {
        0: [4, 6, 8],
        1: [4, 6]
    }

    -> [
         {0: 4, 1: 4},
         {0: 4, 1: 6},
         {0: 6, 1: 4},
         {0: 6, 1: 6},
         {0: 8, 1: 4},
         {0: 8, 1: 6},
       ]
    """
    labels = sorted(label_to_values.keys())
    value_lists = [label_to_values[lab] for lab in labels]

    combos = []
    for values in __import__("itertools").product(*value_lists):
        k_dict = {lab: val for lab, val in zip(labels, values)}
        combos.append(k_dict)
    return combos


def _print_progress_bar(step: int, total: int, bar_len: int = 30, prefix: str = "") -> None:
    """
    Simple console text progress bar.

    Parameters
    ----------
    step : int
        Current step (1-based).
    total : int
        Total number of steps.
    """
    if total <= 0:
        return
    frac = step / total
    filled = int(bar_len * frac)
    bar = "#" * filled + "-" * (bar_len - filled)
    pct = 100.0 * frac
    text = f"\r{prefix}[{bar}] {step}/{total} ({pct:5.1f}%)"
    print(text, end="", flush=True)
    if step == total:
        print()  # newline at the end


def _build_model_params_from_trial(
    trial: optuna.trial.Trial,
    model_cls: Type,
    param_grid: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Construct model_params for model_cls from an Optuna trial and a param_grid.

    Logic:
      - param_grid["label_to_K_values"]: per label a set of candidate K values;
        turned into a dict K[label] and passed as 'K' to model_params.
      - Other keys:
          * list length 1 -> constant value,
          * list length >1 -> trial.suggest_categorical(...),
          * non-list       -> constant value.
    """
    pg = dict(param_grid)
    params: Dict[str, Any] = {}

    # Special case: label_to_K_values -> K dict
    if "label_to_K_values" in pg:
        label_to_K_values = pg["label_to_K_values"]
        K_dict: Dict[int, int] = {}
        for label in sorted(label_to_K_values.keys()):
            candidates = list(label_to_K_values[label])
            if len(candidates) == 1:
                chosen = candidates[0]
            else:
                chosen = trial.suggest_categorical(
                    f"K_label_{label}",
                    candidates,
                )
            K_dict[label] = chosen
        params["K"] = K_dict

    # Remaining parameters
    for key, values in pg.items():
        if key == "label_to_K_values":
            continue

        if not isinstance(values, (list, tuple)):
            params[key] = values
            continue

        values_list = list(values)
        if len(values_list) == 1:
            params[key] = values_list[0]
        else:
            params[key] = trial.suggest_categorical(key, values_list)

    params = _filter_kwargs(model_cls, params)
    return params


def stratified_kfold_grid_search(
    D,
    y,
    model_cls: Type,
    param_grid: Dict[str, Any],
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
    conf_as_percent: bool = True,
    verbose: bool = True,
    scoring: str = "balanced_accuracy",
    save_path: Optional[Union[str, Path]] = None,
    table_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Exhaustive grid search over param_grid for a GLVQ-like model using
    stratified K-fold cross-validation.

    - param_grid may contain 'label_to_K_values' (dict: label -> list of Ks).
      This is expanded into all K combinations, passed as 'K' to the model.

    Results can be saved as:
      - `save_path` : pickle containing all results and the best configuration.
      - `table_path`: pickle of a pandas DataFrame with one row per fold
                      and parameter combination (including final v-weights).

    Table enhancement:
      - Adds exactly two columns:
          * prototypes_label_0 : list[int] of prototype indices assigned to label 0
          * prototypes_label_1 : list[int] of prototype indices assigned to label 1

    IMPORTANT:
      This function will try to obtain prototypes/labels per fold via (in order):
        1) fold_metrics["final_prototypes"] and fold_metrics["prototype_labels"]
        2) fold_metrics["model"]._w and fold_metrics["model"]._y
      If neither is available, the two columns will be set to None.

    Returns
    -------
    dict
        {
          "all_results": [...],
          "best_result": {...},
          "best_score": float,
          "scoring": str,
          "model_cls": model_cls,
          "n_splits": int,
          "param_grid": dict,
        }
    """
    y = np.asarray(y).astype(int).ravel()
    if y.size == 0:
        raise ValueError("y must not be empty.")

    param_grid = dict(param_grid)
    if "label_to_K_values" in param_grid:
        label_to_K_values = param_grid.pop("label_to_K_values")
        K_list = generate_K_combinations(label_to_K_values)
        if not K_list:
            raise ValueError(
                "Could not generate K combinations from label_to_K_values."
            )
        param_grid["K"] = K_list
        if verbose:
            print(f"Generated {len(K_list)} K combinations from label_to_K_values.")

    param_list = list(ParameterGrid(param_grid))
    total = len(param_list)
    if total == 0:
        raise ValueError("param_grid produced no combinations.")

    all_results: List[Dict[str, Any]] = []
    best_score = -np.inf
    best_result: Optional[Dict[str, Any]] = None
    table_rows: List[Dict[str, Any]] = []

    def _extract_prototypes_and_labels_from_fold(fm: Any):
        """
        Returns (protos, labels) as 1D int arrays or (None, None).

        Supports:
          - fm["final_prototypes"], fm["prototype_labels"]
          - fm["model"]._w, fm["model"]._y
        """
        if not isinstance(fm, dict):
            return None, None

        # 1) preferred: explicit keys
        protos = fm.get("final_prototypes", None)
        labels = fm.get("prototype_labels", None)

        # 2) fallback: stored model object
        if (protos is None or labels is None) and ("model" in fm):
            m = fm.get("model", None)
            if m is not None:
                if protos is None and hasattr(m, "_w"):
                    protos = getattr(m, "_w", None)
                if labels is None and hasattr(m, "_y"):
                    labels = getattr(m, "_y", None)

        if protos is None or labels is None:
            return None, None

        p_arr = np.asarray(protos, dtype=int).ravel()
        l_arr = np.asarray(labels, dtype=int).ravel()

        if p_arr.shape[0] != l_arr.shape[0]:
            # inconsistent fold payload
            return None, None

        return p_arr, l_arr

    for i, params in enumerate(param_list, start=1):
        params_filtered = _filter_kwargs(model_cls, params)

        if verbose:
            _print_progress_bar(i, total, prefix="Grid-Search: ")

        cv_res = stratified_kfold_full_metrics(
            D=D,
            y=y,
            model_cls=model_cls,
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
            model_params=params_filtered,
            conf_as_percent=conf_as_percent,
            verbose=False,
        )

        score = cv_res["averages"][scoring]
        folds = cv_res.get("folds", None)

        # -------- collect v-weights per fold --------
        final_vweights_per_fold = []
        if folds is not None:
            for fold_metrics in folds:
                if isinstance(fold_metrics, dict) and "final_vweights" in fold_metrics:
                    v = np.asarray(fold_metrics["final_vweights"], dtype=float)
                    final_vweights_per_fold.append(v)

        if final_vweights_per_fold:
            V_stack = np.stack(final_vweights_per_fold, axis=0)
            final_vweights_mean = V_stack.mean(axis=0)
        else:
            final_vweights_mean = None

        # -------- collect prototypes/labels per fold (optional in result_dict) --------
        final_prototypes_per_fold = []
        prototype_labels_per_fold = []
        if folds is not None:
            for fold_metrics in folds:
                p_arr, l_arr = _extract_prototypes_and_labels_from_fold(fold_metrics)
                if p_arr is not None and l_arr is not None:
                    final_prototypes_per_fold.append(p_arr)
                    prototype_labels_per_fold.append(l_arr)

        result_entry = {
            "params": params_filtered,
            "cv_result": cv_res,
            "score": score,
            "final_vweights_per_fold": final_vweights_per_fold if final_vweights_per_fold else None,
            "final_vweights_mean": final_vweights_mean,
            "final_prototypes_per_fold": final_prototypes_per_fold if final_prototypes_per_fold else None,
            "prototype_labels_per_fold": prototype_labels_per_fold if prototype_labels_per_fold else None,
        }
        all_results.append(result_entry)

        if score > best_score:
            best_score = score
            best_result = result_entry

        # -------- build fold rows for table --------
        if folds is not None:
            for fold_idx, fold_metrics in enumerate(folds):
                # normalize fold dict
                if isinstance(fold_metrics, dict):
                    final_vweights = fold_metrics.get("final_vweights", None)

                    # remove non-metric payload keys from metrics_dict
                    metrics_dict = {
                        k: v
                        for k, v in fold_metrics.items()
                        if k not in (
                            "final_vweights",
                            "final_prototypes",
                            "prototype_labels",
                            "model",
                        )
                    }
                else:
                    metrics_dict = fold_metrics
                    final_vweights = None

                row = {"fold_index": fold_idx}

                # metrics
                if isinstance(metrics_dict, dict):
                    for metric_name, value in metrics_dict.items():
                        row[metric_name] = value

                    if scoring in metrics_dict:
                        row[f"{scoring}_fold"] = metrics_dict[scoring]

                # v-weights (wide columns)
                if final_vweights is not None:
                    v_arr = np.asarray(final_vweights, dtype=float).ravel()
                    for j, vj in enumerate(v_arr):
                        row[f"vweight_{j}"] = vj

                # --- prototypes grouped into exactly two columns (label 0/1) ---
                p_arr, l_arr = _extract_prototypes_and_labels_from_fold(fold_metrics)
                if p_arr is not None and l_arr is not None:
                    row["prototypes_label_0"] = p_arr[l_arr == 0].tolist()
                    row["prototypes_label_1"] = p_arr[l_arr == 1].tolist()
                else:
                    row["prototypes_label_0"] = None
                    row["prototypes_label_1"] = None

                # params
                for p_name, p_val in params_filtered.items():
                    row[p_name] = p_val

                table_rows.append(row)
        else:
            avg_metrics = cv_res.get("averages", {})
            row = {"fold_index": -1}

            if isinstance(avg_metrics, dict):
                for metric_name, value in avg_metrics.items():
                    row[metric_name] = value

            for p_name, p_val in params_filtered.items():
                row[p_name] = p_val

            row["prototypes_label_0"] = None
            row["prototypes_label_1"] = None

            table_rows.append(row)

    result_dict = {
        "all_results": all_results,
        "best_result": best_result,
        "best_score": best_score,
        "scoring": scoring,
        "model_cls": f"{model_cls.__module__}.{model_cls.__qualname__}",   #"model_cls": model_cls,
        "n_splits": n_splits,
        "param_grid": param_grid,
    }

    if verbose and best_result is not None:
        best_params = best_result["params"]
        best_avg = best_result["cv_result"]["averages"]

        print("\n" + "=" * 80)
        print("Best parameter combination (Grid Search):")
        print(best_params)
        print(f"\nBest {scoring}: {best_score:.4f}")
        print("\nMetrics (mean over folds):")
        for k, v in best_avg.items():
            print(f"  {k}: {v:.4f}")
        if best_result.get("final_vweights_mean") is not None:
            print("\nFinal mean v-weights (over folds):")
            print(best_result["final_vweights_mean"])
        print("=" * 80)

    # Save results
    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump(result_dict, f)
        if verbose:
            print(f"Grid search results saved to '{p}'.")

    # Save fold table
    if table_path is None and save_path is not None:
        p = Path(save_path)
        table_path = p.with_name(p.stem + "_all_folds.pkl")

    if table_path is not None:
        table_path = Path(table_path)
        table_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(table_rows)
        with table_path.open("wb") as f:
            pickle.dump(df, f)

        if verbose:
            print(
                f"Fold table saved to '{table_path}' "
                f"(pandas.DataFrame as pickle)."
            )

    return result_dict



def stratified_kfold_bayes_search(
    D,
    y,
    model_cls: Type,
    param_grid: Dict[str, Any],
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
    conf_as_percent: bool = True,
    verbose: bool = True,
    scoring: str = "balanced_accuracy",
    n_trials: int = 50,
    save_path: Optional[Union[str, Path]] = None,
    study_name: Optional[str] = None,
    sampler: Optional[optuna.samplers.BaseSampler] = None,
    pruner: Optional[optuna.pruners.BasePruner] = None,
    table_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    y = np.asarray(y).astype(int).ravel()
    if y.size == 0:
        raise ValueError("y must not be empty.")

    def objective(trial: optuna.trial.Trial) -> float:
        model_params = _build_model_params_from_trial(
            trial=trial,
            model_cls=model_cls,
            param_grid=param_grid,
        )

        cv_res = stratified_kfold_full_metrics(
            D=D,
            y=y,
            model_cls=model_cls,
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
            model_params=model_params,
            conf_as_percent=conf_as_percent,
            verbose=False,
        )

        score = cv_res["averages"][scoring]

        # store only lightweight, pickle-safe payload
        trial.set_user_attr("model_params", model_params)
        trial.set_user_attr("averages", cv_res["averages"])
        trial.set_user_attr("folds", cv_res["folds"])  # optional; needed for table_rows / vweights

        if verbose:
            print(f"[Trial {trial.number}] {scoring}={score:.4f}")

        return score

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        sampler=sampler,
        pruner=pruner,
    )

    if verbose:
        print(f"Starting Bayesian optimization with {n_trials} trials ...")

    study.optimize(objective, n_trials=n_trials)

    all_results: List[Dict[str, Any]] = []
    best_result: Optional[Dict[str, Any]] = None
    best_score = -np.inf
    table_rows: List[Dict[str, Any]] = []

    for tr in study.trials:
        if tr.state != TrialState.COMPLETE:
            continue

        model_params = tr.user_attrs.get("model_params", None)
        avg = tr.user_attrs.get("averages", None)
        folds = tr.user_attrs.get("folds", None)
        score = tr.value

        # -------- collect v-weights per fold --------
        final_vweights_per_fold = []
        if folds is not None:
            for fold_metrics in folds:
                if isinstance(fold_metrics, dict) and "final_vweights" in fold_metrics:
                    v = np.asarray(fold_metrics["final_vweights"], dtype=float)
                    final_vweights_per_fold.append(v)

        if final_vweights_per_fold:
            V_stack = np.stack(final_vweights_per_fold, axis=0)
            final_vweights_mean = V_stack.mean(axis=0)
        else:
            final_vweights_mean = None

        entry = {
            "trial_number": tr.number,
            "params": model_params,
            "averages": avg,
            "score": score,
            "folds": folds,
            "final_vweights_per_fold": final_vweights_per_fold if final_vweights_per_fold else None,
            "final_vweights_mean": final_vweights_mean,
        }
        all_results.append(entry)

        if score > best_score:
            best_score = score
            best_result = entry

        # -------- build rows for DataFrame --------
        if folds is not None:
            for fold_idx, fold_metrics in enumerate(folds):
                if isinstance(fold_metrics, dict):
                    final_vweights = fold_metrics.get("final_vweights", None)
                    metrics_dict = {k: v for k, v in fold_metrics.items() if k != "final_vweights"}
                else:
                    metrics_dict = fold_metrics
                    final_vweights = None

                row = {"trial_number": tr.number, "fold_index": fold_idx}

                if isinstance(metrics_dict, dict):
                    for metric_name, value in metrics_dict.items():
                        row[metric_name] = value
                    if scoring in metrics_dict:
                        row[f"{scoring}_fold"] = metrics_dict[scoring]

                if final_vweights is not None:
                    v_arr = np.asarray(final_vweights, dtype=float).ravel()
                    for j, vj in enumerate(v_arr):
                        row[f"vweight_{j}"] = vj

                if model_params is not None:
                    for p_name, p_val in model_params.items():
                        row[p_name] = p_val

                table_rows.append(row)

    result_dict = {
        "all_results": all_results,
        "best_result": best_result,
        "best_score": best_score,
        "scoring": scoring,
        "model_cls": f"{model_cls.__module__}.{model_cls.__qualname__}",
        "n_splits": n_splits,
        "param_grid": param_grid,
        "study_best_trial_number": study.best_trial.number,
    }

    if verbose and best_result is not None:
        print("\n" + "=" * 80)
        print("Best parameter combination (Bayesian Optimization):")
        print(best_result["params"])
        print(f"\nBest {scoring}: {best_score:.4f}")
        if best_result.get("averages") is not None:
            print("\nMetrics (mean over folds):")
            for k, v in best_result["averages"].items():
                print(f"  {k}: {v:.4f}")
        if best_result.get("final_vweights_mean") is not None:
            print("\nFinal mean v-weights (over folds):")
            print(best_result["final_vweights_mean"])
        print("=" * 80)

    # Save overall results
    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump(result_dict, f)
        if verbose:
            print(f"Bayes search results saved to '{p}'.")

    # Save trial/fold table
    if table_path is None and save_path is not None:
        p = Path(save_path)
        table_path = p.with_name(p.stem + "_all_trials.pkl")

    if table_path is not None:
        table_path = Path(table_path)
        table_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(table_rows)
        with table_path.open("wb") as f:
            pickle.dump(df, f)
        if verbose:
            print(f"Trial/fold table saved to '{table_path}' (pandas.DataFrame as pickle).")

    return result_dict



def glvq_mu_values_train(model, y_train, eps=1e-5):
    """
    Compute the GLVQ mu value for each training sample:
    mu_i = (d_plus - d_minus) / (d_plus + d_minus)
    based on the learned prototypes.
    """
    # Full pairwise dissimilarity matrix of the training set (m x m)
    D_all = model.final_matrix_
    m = D_all.shape[0]

    # Dissimilarities to all prototypes (m x K)
    proto_indices = model._w          # indices into the training set
    proto_labels = model._y           # prototype class labels (K,)
    D_to_proto = D_all[:, proto_indices]  # (m, K)

    dp = np.empty(m)   # d_+
    dm = np.empty(m)   # d_-

    classes = np.unique(y_train)
    for c in classes:
        mask_x = (y_train == c)          # samples of class c
        mask_pos = (proto_labels == c)   # prototypes of class c
        mask_neg = ~mask_pos             # prototypes of other classes

        D_c = D_to_proto[mask_x]         # (n_c, K)
        dp[mask_x] = D_c[:, mask_pos].min(axis=1)
        dm[mask_x] = D_c[:, mask_neg].min(axis=1)

    mu = (dp - dm) / (dp + dm + eps)

    # Optionally apply the model's phi function:
    mu_phi = model.phi(mu)

    return mu, mu_phi



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
    show=True,          # show the plot
    return_fig=False    # return the matplotlib Figure
):
    # === Collect data ===
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

    # === Figure layout ===
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
        # Neither weights nor loss available
        fig, ax_empty = plt.subplots(1, 1, figsize=(figsize[0], max(3.0, figsize[1]/2)))
        ax_empty.text(0.5, 0.5, "No data found (neither weights nor loss)",
                      ha="center", va="center", fontsize=12)
        ax_empty.axis("off")
        plt.tight_layout()
        if show:
            plt.show()
            if return_fig:
                plt.close(fig)
        return fig if return_fig else None

    # === Top: distance weights (v-weights) ===
    if show_weights:
        V_path = np.asarray(V_path, dtype=float)
        steps = V_path.shape[0]
        idx = np.arange(steps)

        # Selection
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

        # Normalization
        if normalize_v:
            mn = np.nanmin(V_plot, axis=0)
            mx = np.nanmax(V_plot, axis=0)
            span = np.where((mx - mn) == 0, 1.0, (mx - mn))
            V_plot = (V_plot - mn) / span

        # Smoothing
        if smooth_v and smooth_v > 1:
            V_plot = np.column_stack([_moving_average(V_plot[:, j], smooth_v) for j in range(V_plot.shape[1])])

        # Plot
        for j in range(V_plot.shape[1]):
            (ax_top if has_loss else ax_top).plot(idx, V_plot[:, j], lw=1.5, label=f"v{labels_idx[j]}")
        ax = ax_top  # Alias
        ax.set_title("Evolution of distance weights")
        ax.set_xlabel("Step")
        ax.set_ylabel("Weight" + (" (normalized)" if normalize_v else ""))
        if V_plot.shape[1] <= 12:
            ax.legend(frameon=False)
        else:
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)

    # === Bottom: loss ===
    if show_loss_ax:
        loss = np.asarray(loss, dtype=float).ravel()
        ax = ax_bot  # Alias
        ax.plot(np.arange(len(loss)), loss, lw=1.5, marker='o', ms=3, markevery=max(1, len(loss)//50))
        ax.set_title("Loss evolution during training")
        ax.set_xlabel("Update index")
        ax.set_ylabel("GLVQ loss")
        ax.grid(True, alpha=0.3)

    # === Render / return without double display (e.g., in notebooks) ===
    plt.tight_layout()
    if show:
        plt.show()
        if return_fig:
            plt.close(fig)   # prevent double render in notebooks

    return fig if return_fig else None


def _expand_param_grid_with_label_to_K_values(
    param_grid: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Expands param_grid into a list of concrete parameter dicts.

    Special key:
    - 'label_to_K_values': dict[label -> list_of_K]
      generates cartesian product across labels.
    """
    grid = dict(param_grid)
    label_to_K_values = grid.pop("label_to_K_values", None)

    keys = list(grid.keys())
    values_lists = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]

    base_combos: List[Dict[str, Any]] = []
    for vals in itertools.product(*values_lists):
        base_combos.append({k: v for k, v in zip(keys, vals)})

    if label_to_K_values is None:
        return base_combos

    labels = sorted(label_to_K_values.keys())
    per_label_lists = [label_to_K_values[lbl] for lbl in labels]

    full_combos: List[Dict[str, Any]] = []
    for base in base_combos:
        for k_vals in itertools.product(*per_label_lists):
            ltK = {lbl: k for lbl, k in zip(labels, k_vals)}
            d = dict(base)
            d["label_to_K_values"] = ltK
            full_combos.append(d)

    return full_combos


def _sanitize_model_params_for_cls(model_cls: Type, model_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fix for your error:
    - If model __init__ needs 'K' but grid provides 'label_to_K_values',
      map it to 'K' and drop 'label_to_K_values' (unless accepted).

    Also:
    - If model __init__ does NOT accept **kwargs, remove unknown keys.
    """
    mp = dict(model_params)

    sig = inspect.signature(model_cls.__init__)
    init_params = set(sig.parameters.keys())

    # Core fix: label_to_K_values -> K
    if "K" in init_params and "K" not in mp and "label_to_K_values" in mp:
        mp["K"] = mp["label_to_K_values"]

    # Remove label_to_K_values if not accepted
    if "label_to_K_values" in mp and "label_to_K_values" not in init_params:
        mp.pop("label_to_K_values", None)

    # Remove unknown keys unless **kwargs exists
    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if not has_kwargs:
        mp = {k: v for k, v in mp.items() if k in init_params}

    return mp


def stratified_kfold_grid_search_2(
    D,
    y,
    model_cls: Type,
    param_grid: Dict[str, Any],
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42,
    conf_as_percent: bool = True,
    verbose: bool = True,
    scoring: str = "balanced_accuracy",
    save_path: Optional[Union[str, Path]] = None,
    table_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    GridSearch over param_grid (incl. label_to_K_values) using stratified_kfold_full_metrics.
    Produces same result structure + optional fold table pickles.
    """
    y = np.asarray(y).astype(int).ravel()
    if y.size == 0:
        raise ValueError("y must not be empty.")

    combos = _expand_param_grid_with_label_to_K_values(param_grid)
    if len(combos) == 0:
        raise ValueError("param_grid produced 0 combinations.")

    if verbose:
        print(f"Starting GridSearch with {len(combos)} combinations ...")

    all_results: List[Dict[str, Any]] = []
    best_result: Optional[Dict[str, Any]] = None
    best_score = -np.inf
    table_rows: List[Dict[str, Any]] = []

    for trial_number, raw_model_params in enumerate(combos):
        # IMPORTANT: this line fixes your K-missing error
        model_params = _sanitize_model_params_for_cls(model_cls, raw_model_params)

        cv_res = stratified_kfold_full_metrics(
            D=D,
            y=y,
            model_cls=model_cls,
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
            model_params=model_params,
            conf_as_percent=conf_as_percent,
            verbose=False,
        )

        score = cv_res["averages"][scoring]

        if verbose:
            print(f"[Trial {trial_number}] {scoring}={score:.4f}")

        # vweights per fold
        final_vweights_per_fold = []
        if cv_res is not None and cv_res.get("folds") is not None:
            for fold_metrics in cv_res["folds"]:
                if isinstance(fold_metrics, dict) and "final_vweights" in fold_metrics:
                    v = np.asarray(fold_metrics["final_vweights"], dtype=float)
                    final_vweights_per_fold.append(v)

        if final_vweights_per_fold:
            V_stack = np.stack(final_vweights_per_fold, axis=0)
            final_vweights_mean = V_stack.mean(axis=0)
        else:
            final_vweights_mean = None

        entry = {
            "trial_number": trial_number,
            "params": model_params,
            "cv_result": cv_res,
            "score": score,
            "final_vweights_per_fold": final_vweights_per_fold if final_vweights_per_fold else None,
            "final_vweights_mean": final_vweights_mean,
        }
        all_results.append(entry)

        if score > best_score:
            best_score = score
            best_result = entry

        # fold table rows
        if cv_res is not None and cv_res.get("folds") is not None:
            for fold_idx, fold_metrics in enumerate(cv_res["folds"]):
                if isinstance(fold_metrics, dict):
                    final_vweights = fold_metrics.get("final_vweights", None)
                    metrics_dict = {k: v for k, v in fold_metrics.items() if k != "final_vweights"}
                else:
                    metrics_dict = fold_metrics
                    final_vweights = None

                row = {"trial_number": trial_number, "fold_index": fold_idx}

                if isinstance(metrics_dict, dict):
                    for metric_name, value in metrics_dict.items():
                        row[metric_name] = value
                    if scoring in metrics_dict:
                        row[f"{scoring}_fold"] = metrics_dict[scoring]

                if final_vweights is not None:
                    v_arr = np.asarray(final_vweights, dtype=float).ravel()
                    for j, vj in enumerate(v_arr):
                        row[f"vweight_{j}"] = vj

                for p_name, p_val in model_params.items():
                    row[p_name] = p_val

                table_rows.append(row)

    result_dict = {
        "all_results": all_results,
        "best_result": best_result,
        "best_score": best_score,
        "scoring": scoring,
        "model_cls": f"{model_cls.__module__}.{model_cls.__qualname__}",    #"model_cls": model_cls,
        "n_splits": n_splits,
        "param_grid": param_grid,
        "best_trial_number": (best_result["trial_number"] if best_result is not None else None),
    }

    if verbose and best_result is not None:
        print("\n" + "=" * 80)
        print("Best parameter combination (GridSearch):")
        print(best_result["params"])
        print(f"\nBest {scoring}: {best_score:.4f}")
        if best_result["cv_result"] is not None:
            print("\nMetrics (mean over folds):")
            for k, v in best_result["cv_result"]["averages"].items():
                print(f"  {k}: {v:.4f}" if isinstance(v, (int, float, np.floating)) else f"  {k}: {v}")
        if best_result.get("final_vweights_mean") is not None:
            print("\nFinal mean v-weights (over folds):")
            print(best_result["final_vweights_mean"])
        print("=" * 80)

    # Save overall results
    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump(result_dict, f)
        if verbose:
            print(f"GridSearch results saved to '{p}'.")

    # Save trial/fold table
    if table_path is None and save_path is not None:
        p = Path(save_path)
        table_path = p.with_name(p.stem + "_all_trials.pkl")

    if table_path is not None:
        table_path = Path(table_path)
        table_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(table_rows)
        with table_path.open("wb") as f:
            pickle.dump(df, f)
        if verbose:
            print(f"Trial/fold table saved to '{table_path}' (pandas.DataFrame as pickle).")

    return result_dict

