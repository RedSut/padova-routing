"""
src/predizioni.py
==================
Generazione dei potenziali predetti (y_hat) a partire da un modello ML,
con cache delle coordinate e calcolo vettorizzato della distanza haversine.

Compatibile con qualunque modello che implementi predict(X) e
(opzionalmente) feature_names_in_ — sia scikit-learn nativo, sia il
WrapperXGBoost definito in modelli/pipeline_unificata.py.
"""

import networkx as nx
import numpy as np
import pandas as pd

# Cache globale delle coordinate dei nodi, popolata alla prima chiamata per
# ogni grafo (identificato da id(G)). Evita di rileggere lat/lon da
# G.nodes(data=True) con un ciclo Python ad ogni chiamata di genera_predizioni.
_coord_cache = {"graph_id": None, "nodi_ordine": None, "lats": None, "lons": None}


def _get_coord_arrays(G: nx.MultiDiGraph):
    """Restituisce (nodi_ordine, lats, lons) come array numpy, con caching."""
    if _coord_cache["graph_id"] != id(G):
        nodes_data = G.nodes(data=True)
        nodi_ordine = list(G.nodes())
        lats = np.array([nodes_data[n]["y"] for n in nodi_ordine])
        lons = np.array([nodes_data[n]["x"] for n in nodi_ordine])
        _coord_cache.update(graph_id=id(G), nodi_ordine=nodi_ordine, lats=lats, lons=lons)
    return _coord_cache["nodi_ordine"], _coord_cache["lats"], _coord_cache["lons"]


def _haversine_vettoriale(lats, lons, target_lat, target_lon):
    """
    Distanza haversine (in metri) vettorizzata con numpy, equivalente a
    chiamare ox.distance.great_circle per ogni nodo ma senza ciclo Python.
    Accetta array numpy o scalari per lats/lons.
    """
    R = 6371000.0  # raggio terrestre medio in metri
    lat1, lon1 = np.radians(lats), np.radians(lons)
    lat2, lon2 = np.radians(target_lat), np.radians(target_lon)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return R * 2.0 * np.arcsin(np.sqrt(a))


def genera_predizioni(
    G: nx.MultiDiGraph,
    model,
    target,
    periodo_giorno: float | None = None,
    scale_factor: float = 10.0,
) -> tuple[dict, dict]:
    """
    Genera i potenziali predetti y_hat per ogni nodo del grafo G rispetto
    alla destinazione `target`.

    Scale factor a 10.0 di default perché il modello predice in SECONDI,
    mentre il grafo di calcolo (campo travel_time_d) e' in DECIMI di secondo.
    Usare scale_factor=1.0 se si lavora con travel_time_s, 100.0 con
    travel_time_c, 1000.0 con travel_time_m.

    INVERSIONE DI SEGNO: il modello predice la distanza residua stimata (in
    secondi) dal nodo al target — valori piu' alti = nodo piu' lontano. Per
    usarla come potenziale di Bellman-Ford-Moore (dove costo_ridotto(u,v) =
    peso(u,v) + y_hat[u] - y_hat[v] deve restare piccolo lungo il percorso
    verso il target) serve il segno opposto: da qui il -y_arr.

    Restituisce:
        y_hat     : {nodo -> float}  potenziali float originali
        y_hat_int : {nodo -> int}    potenziali arrotondati (per pesi interi)
    """
    nodes_data = G.nodes(data=True)
    target_lat, target_lon = nodes_data[target]["y"], nodes_data[target]["x"]

    nodi_ordine, lats, lons = _get_coord_arrays(G)
    aria_dist = _haversine_vettoriale(lats, lons, target_lat, target_lon)

    colonne = ["node_lat", "node_lon", "target_lat", "target_lon", "haversine_dist_m"]
    X = pd.DataFrame(
        {
            "node_lat": lats,
            "node_lon": lons,
            "target_lat": target_lat,
            "target_lon": target_lon,
            "haversine_dist_m": aria_dist,
        }
    )
    if periodo_giorno is not None:
        X["periodo_giorno"] = periodo_giorno
        colonne.append("periodo_giorno")
    X = X[colonne]

    if hasattr(model, "feature_names_in_"):
        X = X[model.feature_names_in_]

    y_arr = model.predict(X)

    y_hat = dict(zip(nodi_ordine, -y_arr * scale_factor))
    y_hat[target] = 0.0

    y_hat_int = {n: int(round(v)) for n, v in y_hat.items()}
    return y_hat, y_hat_int
