"""
src/interpolazione.py
======================
Implementazione dell'interpolazione spaziale: il modello ML predice le 
variabili duali solo per un sottoinsieme di nodi "ancora" (campionati casualmente), 
e i restanti nodi ottengono il loro potenziale per interpolazione spaziale 
(triangolazione di Delaunay via scipy.interpolate.LinearNDInterpolator), con fallback 
al vicino più prossimo per i nodi fuori dal convex hull delle ancore.

Vantaggio: il costo di model.predict() scala con sample_ratio * N invece 
che con N, mentre l'interpolazione stessa è calcolata velocemente in C da scipy.
"""

import random

import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


def genera_predizioni_interpolate(
    G: nx.MultiDiGraph,
    model,
    target,
    periodo_giorno: float | None = None,
    sample_ratio: float = 0.1,
    scale_factor: float = 10.0,
    min_ancore: int = 100,
    seed: int | None = None,
) -> tuple[dict, dict]:
    """
    Genera i potenziali predetti calcolando il modello ML solo su un
    campione di nodi "ancora" (sample_ratio della popolazione, minimo
    min_ancore), e ottenendo i restanti per interpolazione spaziale
    lineare (Delaunay) con fallback nearest-neighbor.

    Il target e' sempre incluso tra le ancore, per garantire che il suo
    potenziale resti esattamente 0.

    L'inversione di segno rispetto all'output grezzo del modello viene 
    eseguita internamente per garantire compatibilità con la sanazione BCF.

    Restituisce (y_hat, y_hat_int).
    """
    if seed is not None:
        random.seed(seed)

    nodi_totali = list(G.nodes(data=True))
    N = len(nodi_totali)

    num_anchors = max(min_ancore, int(N * sample_ratio))
    tutti_ids = [n[0] for n in nodi_totali]
    anchor_ids = set(random.sample(tutti_ids, min(num_anchors, N)))
    anchor_ids.add(target)  # garantisce precisione esatta sul target

    anchors = [n for n in nodi_totali if n[0] in anchor_ids]
    others = [n for n in nodi_totali if n[0] not in anchor_ids]

    target_lat = G.nodes[target]["y"]
    target_lon = G.nodes[target]["x"]

    X_anchors, anchor_coords = [], []
    for n_id, data in anchors:
        node_lat, node_lon = data["y"], data["x"]
        anchor_coords.append([node_lon, node_lat])  # (x, y) per l'interpolatore
        hav_dist = ox.distance.great_circle(node_lat, node_lon, target_lat, target_lon)
        X_anchors.append([node_lat, node_lon, target_lat, target_lon, hav_dist])

    feature_cols = ["node_lat", "node_lon", "target_lat", "target_lon", "haversine_dist_m"]
    df_anchors = pd.DataFrame(X_anchors, columns=feature_cols)

    if periodo_giorno is not None:
        df_anchors["periodo_giorno"] = periodo_giorno
        feature_cols.append("periodo_giorno")

    if hasattr(model, "feature_names_in_"):
        df_anchors = df_anchors[model.feature_names_in_]

    # Inferenza ML solo sulle ancore — il vero collo di bottiglia, ridotto
    # a sample_ratio * N invece di N
    y_anchors_raw = model.predict(df_anchors)

    if others:
        interp_lin = LinearNDInterpolator(anchor_coords, y_anchors_raw)
        interp_near = NearestNDInterpolator(anchor_coords, y_anchors_raw)

        other_coords = [[data["x"], data["y"]] for n_id, data in others]
        y_others_raw = interp_lin(other_coords)

        # I nodi fuori dal convex hull delle ancore restituiscono NaN da
        # LinearNDInterpolator: fallback al vicino più prossimo.
        nan_mask = np.isnan(y_others_raw)
        if np.any(nan_mask):
            nan_coords = np.array(other_coords)[nan_mask]
            y_others_raw[nan_mask] = interp_near(nan_coords)
    else:
        y_others_raw = np.array([])

    y_hat_raw, y_hat_int = {}, {}

    for i, (n_id, _) in enumerate(anchors):
        val = -y_anchors_raw[i] * scale_factor
        y_hat_raw[n_id] = val
        y_hat_int[n_id] = int(round(val))

    for i, (n_id, _) in enumerate(others):
        val = -y_others_raw[i] * scale_factor
        y_hat_raw[n_id] = val
        y_hat_int[n_id] = int(round(val))

    # Forza esattamente a 0 il potenziale del target
    y_hat_raw[target] = 0.0
    y_hat_int[target] = 0

    return y_hat_raw, y_hat_int
