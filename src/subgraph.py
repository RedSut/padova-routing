"""
src/subgraph.py
================
Implementazione della strategia "Sub-Graph Routing": invece di calcolare 
le predizioni su tutto il grafo, si estrae prima un sottografo nella zona 
di interesse tra source e target, e si lavora solo su quello.
target, e si lavora solo su quello.

Due strategie di ritaglio geometrico:
  - extract_subgraph_bbox     : rettangolo (bounding box) con padding fisso
  - extract_subgraph_ellipse  : ellisse con i due fuochi su source e target

L'ellisse e' generalmente più efficiente del rettangolo a parità di
padding, perché esclude gli angoli "inutili" del rettangolo (zone lontane
sia da source che da target ma comunque dentro il bbox).
"""

import networkx as nx
import numpy as np


def extract_subgraph_bbox(
    G: nx.MultiDiGraph, source, target, padding_km: float = 2.0
) -> nx.MultiDiGraph:
    """
    Ritaglia un sottografo rettangolare attorno alla coppia source/target,
    con un padding fisso in chilometri attorno al bounding box minimo.

    Conversione km -> gradi: 1° di latitudine ≈ 111km; 1° di longitudine
    dipende dal coseno della latitudine (i meridiani convergono ai poli).
    """
    lat_s, lon_s = G.nodes[source]["y"], G.nodes[source]["x"]
    lat_t, lon_t = G.nodes[target]["y"], G.nodes[target]["x"]

    min_lat, max_lat = min(lat_s, lat_t), max(lat_s, lat_t)
    min_lon, max_lon = min(lon_s, lon_t), max(lon_s, lon_t)

    pad_lat_deg = padding_km / 111.0
    lat_mid = np.radians((lat_s + lat_t) / 2.0)
    pad_lon_deg = padding_km / (111.0 * np.cos(lat_mid))

    min_lat -= pad_lat_deg
    max_lat += pad_lat_deg
    min_lon -= pad_lon_deg
    max_lon += pad_lon_deg

    valid_nodes = [
        n for n, d in G.nodes(data=True)
        if min_lat <= d["y"] <= max_lat and min_lon <= d["x"] <= max_lon
    ]

    return G.subgraph(valid_nodes).copy()


def extract_subgraph_ellipse(
    G: nx.MultiDiGraph, source, target, padding_km: float = 2.0
) -> nx.MultiDiGraph:
    """
    Ritaglia un sottografo ellittico con i due fuochi su source e target:
    un nodo N è incluso se D(N,source) + D(N,target) <= D(source,target) +
    padding_km — la definizione geometrica di un'ellisse.

    Approssimazione equirettangolare (sfera -> piano) per il calcolo delle
    distanze in gradi: perfettamente adeguata per scale cittadine o regionali
    (es. l'intero Veneto). La leggera distorsione geometrica non inficia 
    minimamente l'utilità del sottografo.
    """
    lat_s, lon_s = G.nodes[source]["y"], G.nodes[source]["x"]
    lat_t, lon_t = G.nodes[target]["y"], G.nodes[target]["x"]

    nodi = list(G.nodes(data=True))
    node_ids = np.array([n[0] for n in nodi])
    lats = np.array([n[1]["y"] for n in nodi])
    lons = np.array([n[1]["x"] for n in nodi])

    lat_mid = np.radians((lat_s + lat_t) / 2.0)
    cos_lat = np.cos(lat_mid)

    def dist_gradi(lats1, lons1, lat2, lon2):
        dlat = lats1 - lat2
        dlon = (lons1 - lon2) * cos_lat
        return np.sqrt(dlat ** 2 + dlon ** 2)

    dist_s_deg = dist_gradi(lats, lons, lat_s, lon_s)
    dist_t_deg = dist_gradi(lats, lons, lat_t, lon_t)
    dist_focale_deg = dist_gradi(np.array([lat_s]), np.array([lon_s]), lat_t, lon_t)[0]

    padding_deg = padding_km / 111.0

    mask = (dist_s_deg + dist_t_deg) <= (dist_focale_deg + padding_deg)
    valid_nodes = node_ids[mask]

    return G.subgraph(valid_nodes).copy()


def assicura_connessione(
    G_sub: nx.MultiDiGraph, G_completo: nx.MultiDiGraph, source, target, raggio_extra_km: float = 1.0
):
    """
    Verifica che target sia presente in G_sub; se non lo e' (puo' succedere
    ai margini del ritaglio), estende il sottografo includendo un piccolo
    intorno del target preso dal grafo completo.
    """
    if target in G_sub.nodes():
        return G_sub

    extra = extract_subgraph_bbox(G_completo, target, target, padding_km=raggio_extra_km)
    return nx.compose(G_sub, extra)
