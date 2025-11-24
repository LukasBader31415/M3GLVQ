# functions.py
# -*- coding: utf-8 -*-
"""
Utility functions for:
- aligning distance/similarity matrices to a DataFrame,
- visualizing prototype-based models from distance matrices,
- cross-validated grid search and Bayesian optimization for GLVQ-like models.

Public API:
    align_square_matrix_to_df_order
    align_matrices_to_df
    visualize_from_distance_and_model
    stratified_kfold_grid_search
    stratified_kfold_bayes_search
"""

from typing import Dict, Tuple, List, Union, Any, Optional, Type
from pathlib import Path
import pickle

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
    Reorder a square distance/similarity matrix D_df (index=IDs, columns=IDs)
    to match the order of df[id_col].

    All IDs are cast to str to avoid type mismatches (e.g. "1000013" vs 1000013).
    Raises an error if required IDs are missing or NaNs appear after reindexing.

    Parameters
    ----------
    D_df : pd.DataFrame
        Square matrix with identical ID sets in index and columns.
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

    # Optional symmetry check (typical for distance matrices)
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
    Align multiple square matrices to the same ID subset & order defined by df[id_col].

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
# Embedding & visualization from distance matrix
# =============================================================================


def _filter_kwargs(cls, kwargs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Filter a kwargs dict so that only arguments supported by cls.__init__ remain."""
    if kwargs is None:
        return {}
    sig = __import__("inspect").signature(cls.__init__).parameters
    return {k: v for k, v in kwargs.items() if k in sig}


def _validate_distance_matrix(D: np.ndarray) -> np.ndarray:
    """
    Validate a distance matrix:
      - must be square,
      - symmetric,
      - non-negative,
      - diagonal set to zero.
    """
    D = np.asarray(D, dtype=np.float64)
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("Distance matrix must be square (n x n).")
    np.fill_diagonal(D, 0.0)
    if not np.allclose(D, D.T, atol=1e-8, rtol=1e-8):
        raise ValueError("Distance matrix must be symmetric.")
    if (D < -1e-12).any():
        raise ValueError("Distances must be non-negative.")
    return D


def classical_mds_from_distance(D: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Classical MDS (Torgerson) embedding from a distance matrix.

    Parameters
    ----------
    D : np.ndarray
        Distance matrix (n x n).
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
      - a distance-based “margin” contour between prototype sets.
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



def visualize_from_distance_and_model(
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
    Compute 2D embeddings (MDS, t-SNE, UMAP) from a distance matrix and
    visualize a prototype-based model in all three spaces.

    Parameters
    ----------
    model : object
        Trained GLVQ-like model with attributes `_w` (prototype indices) and
        `_y` (prototype labels 0/1). Optionally `_w_history` or
        `get_vweight_path()` for trajectories.
    D : array-like (n x n) or DataFrame
        Distance matrix.
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
        tsp["metric"] = "precomputed"
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
        ump["metric"] = "precomputed"
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


def _slice_distance_like(
    D: Union[pd.DataFrame, np.ndarray, List[np.ndarray]],
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
):
    """
    Slice a distance structure into Train-Train and Val-Train parts.

    Supported inputs:
      - single distance matrix as DataFrame (n x n)
      - single distance matrix as ndarray (n x n)
      - list/tuple of distance matrices [D1, D2, ...], each (n x n)
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
            raise ValueError("Distance matrix must be square.")
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
    distance/kernel structure D.

    Parameters
    ----------
    D : distance structure
        See `_slice_distance_like` for supported formats.
    y : array-like
        Binary labels (0/1).
    model_cls : Type
        Model class (e.g. MGLVQ, vMGLVQ_V1, vMGLVQ_V2).
    model_params : dict, optional
        Initialization parameters for model_cls.
    conf_as_percent : bool, default True
        If True, TP/FP/TN/FN are returned in percent.

    Returns
    -------
    dict
        {
          "folds": [...],         # per-fold metrics and final v-weights
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
        D_tr, D_va_tr = _slice_distance_like(D, tr_idx, va_idx)

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

        fold_results.append(res)

        if verbose:
            ba = metrics["balanced_accuracy"]
            print(
                f"[Fold {f}] bal_acc={ba:.4f} | "
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

    return {
        "folds": fold_results,
        "averages": avg,
        "n_splits": n_splits,
        "model_cls": model_cls,
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

        result_entry = {
            "params": params_filtered,
            "cv_result": cv_res,
            "score": score,
            "final_vweights_per_fold": final_vweights_per_fold if final_vweights_per_fold else None,
            "final_vweights_mean": final_vweights_mean,
        }
        all_results.append(result_entry)

        if score > best_score:
            best_score = score
            best_result = result_entry

        # build fold rows for table
        if folds is not None:
            for fold_idx, fold_metrics in enumerate(folds):
                if isinstance(fold_metrics, dict):
                    final_vweights = fold_metrics.get("final_vweights", None)
                    metrics_dict = {
                        k: v for k, v in fold_metrics.items() if k != "final_vweights"
                    }
                else:
                    metrics_dict = fold_metrics
                    final_vweights = None

                row = {"fold_index": fold_idx}

                for metric_name, value in metrics_dict.items():
                    row[metric_name] = value

                if scoring in metrics_dict:
                    row[f"{scoring}_fold"] = metrics_dict[scoring]

                if final_vweights is not None:
                    v_arr = np.asarray(final_vweights, dtype=float).ravel()
                    for j, vj in enumerate(v_arr):
                        row[f"vweight_{j}"] = vj

                for p_name, p_val in params_filtered.items():
                    row[p_name] = p_val

                table_rows.append(row)
        else:
            avg_metrics = cv_res.get("averages", {})
            row = {"fold_index": -1}
            for metric_name, value in avg_metrics.items():
                row[metric_name] = value
            for p_name, p_val in params_filtered.items():
                row[p_name] = p_val
            table_rows.append(row)

    result_dict = {
        "all_results": all_results,
        "best_result": best_result,
        "best_score": best_score,
        "scoring": scoring,
        "model_cls": model_cls,
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
    """
    Bayesian optimization (Optuna) over a param_grid search space for a
    GLVQ-like model using stratified_kfold_full_metrics as objective.

    param_grid may contain 'label_to_K_values' for per-label K candidates.

    For each trial + fold, one row is added to a DataFrame (optional output),
    including metrics, hyperparameters and final v-weights (if available).

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
          "study_best_trial_number": int,
        }
    """
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
        trial.set_user_attr("cv_result", cv_res)
        trial.set_user_attr("model_params", model_params)

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

    all_results = []
    best_result = None
    best_score = -np.inf
    table_rows: List[Dict[str, Any]] = []

    for tr in study.trials:
        if tr.state != TrialState.COMPLETE:
            continue

        model_params = tr.user_attrs.get("model_params", None)
        cv_res = tr.user_attrs.get("cv_result", None)
        score = tr.value

        final_vweights_per_fold = []
        if cv_res is not None and "folds" in cv_res and cv_res["folds"] is not None:
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
            "trial_number": tr.number,
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

        # build rows for DataFrame
        if cv_res is not None and "folds" in cv_res and cv_res["folds"] is not None:
            for fold_idx, fold_metrics in enumerate(cv_res["folds"]):
                if isinstance(fold_metrics, dict):
                    final_vweights = fold_metrics.get("final_vweights", None)
                    metrics_dict = {
                        k: v for k, v in fold_metrics.items() if k != "final_vweights"
                    }
                else:
                    metrics_dict = fold_metrics
                    final_vweights = None

                row = {
                    "trial_number": tr.number,
                    "fold_index": fold_idx,
                }

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
        "model_cls": model_cls,
        "n_splits": n_splits,
        "param_grid": param_grid,
        "study_best_trial_number": study.best_trial.number,
    }

    if verbose and best_result is not None:
        print("\n" + "=" * 80)
        print("Best parameter combination (Bayesian Optimization):")
        print(best_result["params"])
        print(f"\nBest {scoring}: {best_score:.4f}")
        if best_result["cv_result"] is not None:
            print("\nMetrics (mean over folds):")
            for k, v in best_result["cv_result"]["averages"].items():
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
            print(
                f"Trial/fold table saved to '{table_path}' "
                f"(pandas.DataFrame as pickle)."
            )

    return result_dict
