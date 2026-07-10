"""
src/feature_avanzate.py
=========================
Feature aggiuntive per il modello ad anelli, oltre alle 5 base
(node_lat, node_lon, target_lat, target_lon, haversine_dist_m):

- node_degree, target_degree: grado del nodo (densita' di incroci — un
  proxy per "sono in centro affollato" vs "sono in periferia").
- node_road_score, target_road_score: gerarchia della prima strada uscente
  dal nodo (motorway=5 ... living_street=0.5) — un proxy per la velocita'
  tipica della zona.
- heading_deviation_deg: di quanto la strada su cui ci si trova si discosta
  dalla direzione ideale (in linea d'aria) verso il target — 0 = dritto
  verso il target, 180 = direzione opposta.

Le funzioni qui sono VETTORIALIZZATE (numpy), pensate per essere chiamate
su array di migliaia/centinaia di migliaia di nodi in un colpo solo, come
richiesto da genera_predizioni a query-time — non per-nodo con un ciclo
Python, che sarebbe impraticabile alla scala del Veneto (~240k nodi).

Gli attributi che dipendono solo dal grafo (grado, road_score, bearing
della strada uscente) sono precalcolati una volta per grafo e tenuti in
cache (precalcola_attributi_stradali), cosi' chiamate successive di
genera_predizioni sullo stesso grafo non ripetono il lavoro.
"""

import networkx as nx
import numpy as np
import pandas as pd

GERARCHIA_STRADE = {
    "motorway": 5, "trunk": 5, "motorway_link": 4, "trunk_link": 4,
    "primary": 4, "primary_link": 3.5, "secondary": 3, "secondary_link": 2.5,
    "tertiary": 2, "tertiary_link": 1.5,
    "residential": 1, "unclassified": 1, "living_street": 0.5,
}

FEATURE_COLS_BASE = ["node_lat", "node_lon", "target_lat", "target_lon", "haversine_dist_m"]
FEATURE_COLS_EXTRA = [
    "node_degree", "target_degree",
    "node_road_score", "target_road_score",
    "heading_deviation_deg",
]
FEATURE_COLS_AVANZATE = FEATURE_COLS_BASE + FEATURE_COLS_EXTRA

_attributi_cache = {"cache_key": None, "grado": None, "road_score": None, "bearing_strada": None}


def _primo_tag(valore, default=None):
    """OSMnx a volte restituisce liste quando un arco ha piu' tag (es.
    highway=['residential','unclassified']): prendiamo il primo, come fa
    il prototipo originale."""
    if isinstance(valore, list):
        return valore[0] if valore else default
    return valore if valore is not None else default


def precalcola_attributi_stradali(G: nx.MultiDiGraph):
    """
    Precalcola, UNA VOLTA per grafo (con cache), tre dict {nodo: valore}:
    grado, road_score (gerarchia della prima strada uscente), bearing_strada
    (l'orientamento di quella strada, se OSMnx l'ha calcolato in fase di
    download — altrimenti None, e a valle si usera' l'angolo ideale stesso,
    cioe' nessuna penalita' di deviazione per quel nodo).

    Nodi senza archi uscenti (vicoli ciechi) ottengono road_score=0.
    """
    cache_key = (id(G), len(G.nodes()), len(G.edges()))
    if _attributi_cache["cache_key"] == cache_key:
        return _attributi_cache["grado"], _attributi_cache["road_score"], _attributi_cache["bearing_strada"]

    grado = dict(G.degree())
    road_score = {}
    bearing_strada = {}
    for nodo in G.nodes():
        primo_arco = None
        for _, _, data in G.out_edges(nodo, data=True):
            primo_arco = data
            break
        if primo_arco is None:
            road_score[nodo] = 0
            bearing_strada[nodo] = None
            continue
        hw = _primo_tag(primo_arco.get("highway"), "unclassified")
        road_score[nodo] = GERARCHIA_STRADE.get(hw, 1)
        bearing_strada[nodo] = _primo_tag(primo_arco.get("bearing"), None)

    _attributi_cache.update(
        cache_key=cache_key, grado=grado, road_score=road_score, bearing_strada=bearing_strada
    )
    return grado, road_score, bearing_strada


def calcola_bearing_vett(lat1, lon1, lat2, lon2):
    """
    Bearing iniziale (0=Nord, 90=Est), da (lat1,lon1) a (lat2,lon2),
    vettorializzato — stessa formula di calcola_bearing ma su array numpy
    invece che su scalari con math.*. Accetta anche scalari.
    """
    lat1r, lon1r = np.radians(lat1), np.radians(lon1)
    lat2r, lon2r = np.radians(lat2), np.radians(lon2)
    dlon = lon2r - lon1r
    x = np.sin(dlon) * np.cos(lat2r)
    y = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360.0) % 360.0


def _haversine_locale(lats, lons, target_lat, target_lon):
    """Copia locale minimale (evita un import circolare con src.predizioni,
    che a sua volta potrebbe voler importare da qui in futuro)."""
    R = 6371000.0
    lat1, lon1 = np.radians(lats), np.radians(lons)
    lat2, lon2 = np.radians(target_lat), np.radians(target_lon)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return R * 2.0 * np.arcsin(np.sqrt(a))


def estrai_feature_avanzate_vettoriale(G, nodi, target, nodes_data=None):
    """
    Versione vettorializzata di estrai_feature_avanzate (prototipo
    originale, per-nodo): calcola le feature di grado/gerarchia/heading per
    un intero array di nodi in un colpo solo, riusando gli attributi
    precalcolati in cache invece di interrogare G nodo per nodo — necessario
    per essere chiamabile da A* a query-time su grafi di centinaia di
    migliaia di nodi.

    nodi: lista o array di node-id (l'ordine determina l'ordine delle righe
          restituite, coerente con l'uso in genera_predizioni).

    Restituisce un DataFrame con le 5 colonne base piu' le 5 avanzate
    (vedi FEATURE_COLS_AVANZATE).
    """
    if nodes_data is None:
        nodes_data = G.nodes(data=True)

    grado, road_score, bearing_strada = precalcola_attributi_stradali(G)

    n_lat = np.array([nodes_data[n]["y"] for n in nodi], dtype=float)
    n_lon = np.array([nodes_data[n]["x"] for n in nodi], dtype=float)
    t_lat, t_lon = nodes_data[target]["y"], nodes_data[target]["x"]

    hav = _haversine_locale(n_lat, n_lon, t_lat, t_lon)

    n_degree = np.array([grado.get(n, 0) for n in nodi], dtype=float)
    t_degree = float(grado.get(target, 0))

    n_road = np.array([road_score.get(n, 1) for n in nodi], dtype=float)
    t_road = float(road_score.get(target, 1))

    angolo_ideale = calcola_bearing_vett(n_lat, n_lon, t_lat, t_lon)
    angolo_strada = np.array(
        [
            bearing_strada.get(n) if bearing_strada.get(n) is not None else angolo_ideale[i]
            for i, n in enumerate(nodi)
        ],
        dtype=float,
    )
    deviazione = np.abs(angolo_ideale - angolo_strada)
    deviazione = np.minimum(deviazione, 360.0 - deviazione)

    n_nodi = len(nodi)
    return pd.DataFrame(
        {
            "node_lat": n_lat, "node_lon": n_lon,
            "target_lat": np.full(n_nodi, t_lat), "target_lon": np.full(n_nodi, t_lon),
            "haversine_dist_m": hav,
            "node_degree": n_degree, "target_degree": np.full(n_nodi, t_degree),
            "node_road_score": n_road, "target_road_score": np.full(n_nodi, t_road),
            "heading_deviation_deg": deviazione,
        }
    )


def modello_richiede_feature_avanzate(model) -> bool:
    """True se il model (tramite feature_names_in_) usa almeno una delle
    feature avanzate — permette a genera_predizioni di calcolarle solo
    quando servono, senza rallentare i modelli vecchi a 5 feature."""
    if not hasattr(model, "feature_names_in_"):
        return False
    return any(c in FEATURE_COLS_EXTRA for c in model.feature_names_in_)
