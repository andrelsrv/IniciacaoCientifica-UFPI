"""Carrega o cache de features do manifesto pre-calculado por
precompute_manifest_features.py, evitando reextrair a cada retreino."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_manifest_cache(cache_path: Path):
    """Retorna (train_x, train_y, validation, feature_names).

    validation e um dict {condition_name: [(x, y), ...]}, no mesmo formato
    usado pelos scripts de treino existentes (train_robust_classifier_v14+).
    """
    data = np.load(cache_path, allow_pickle=True)
    x_all = data["X"]
    y_all = data["y"]
    split_all = data["split"]
    condition_all = data["condition"]
    names = tuple(data["names"])

    train_mask = split_all == "train"
    train_x = x_all[train_mask]
    train_y = y_all[train_mask].tolist()

    validation: dict[str, list[tuple[np.ndarray, str]]] = {}
    val_mask = split_all == "validation"
    for cond in sorted(set(condition_all[val_mask].tolist())):
        cond_mask = val_mask & (condition_all == cond)
        validation[cond] = list(zip(list(x_all[cond_mask]), y_all[cond_mask].tolist()))

    return train_x, train_y, validation, names
