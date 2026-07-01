"""
src/subgraph.py
================
Implementazione della strategia "Sub-Graph Routing": invece di calcolare
le predizioni su tutto il grafo, si estrae prima un sottografo nella zona
di interesse tra source e target, e si lavora solo su quello.

Due strategie di ritaglio geometrico:
  - extract_subgraph_bbox     : rettangolo (bounding box) con padding fisso
  - extract_subgraph_ellipse  : ellisse con i due fuochi su source e target
  - extract_subgraph_ellipse_indicizzato : come sopra, ma usando un indice
    spaziale (KD-tree) per evitare di calcolare la distanza esatta su OGNI
    nodo del grafo ad ogni chiamata — utile su grafi grandi (es. Veneto,
    ~240k nodi), dove il ritaglio stesso arriva a pesare oltre il 50% del
    tempo totale della pipeline sul sottografo (misurato empiricamente).

L'ellisse e' generalmente più efficiente del rettangolo a parità di
padding, perché esclude gli angoli "inutili" del rettangolo (zone lontane
sia da source che da target ma comunque dentro il bbox).
"""

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

# Cache dell'indice KD-tree, popolata alla prima chiamata per ogni grafo.
# Stessa strategia di caching di src/predizioni.py: chiave = (id(G), n_nodi),
# per evitare falsi positivi se un vecchio oggetto grafo viene garbage-
# collected e il suo id() riassegnato a un nuovo grafo diverso.
_kdtree_cache = {"cache_key": None, "tree": None, "node_ids": None, "lats": None, "lons": None}


def _get_kdtree(G: nx.MultiDiGraph):
    """
    Costruisce (o riusa dalla cache) un cKDTree sulle coordinate grezze
    (lon, lat) dei nodi di G, in gradi non proiettati. La costruzione
    dell'albero è O(N log N), fatta una sola volta per grafo; ogni query
    successiva è O(log N + k), con k = numero di risultati, invece di
    O(N) per uno scan lineare di tutti i nodi.

    NOTA: l'indice usa gradi grezzi (non una proiezione locale), perché
    il fattore di correzione cos(lat) dipende dalla coppia source/target
    specifica di ogni query (vedi extract_subgraph_ellipse_indicizzato),
    non da una media globale del grafo — usare una proiezione fissa in
    fase di indicizzazione introdurrebbe uno scarto sistematico rispetto
    al filtro esatto, che usa il cos_lat locale della query.
    """
    cache_key = (id(G), len(G.nodes()))
    if _kdtree_cache["cache_key"] != cache_key:
        nodi = list(G.nodes(data=True))
        node_ids = np.array([n[0] for n in nodi])
        lats = np.array([n[1]["y"] for n in nodi])
        lons = np.array([n[1]["x"] for n in nodi])
        punti = np.column_stack([lons, lats])
        tree = cKDTree(punti)
        _kdtree_cache.update(
            cache_key=cache_key, tree=tree, node_ids=node_ids,
            lats=lats, lons=lons,
        )
    return (
        _kdtree_cache["tree"], _kdtree_cache["node_ids"],
        _kdtree_cache["lats"], _kdtree_cache["lons"],
    )


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


def extract_subgraph_ellipse_indicizzato(
    G: nx.MultiDiGraph, source, target, padding_km: float = 2.0
) -> nx.MultiDiGraph:
    """
    Versione indicizzata di extract_subgraph_ellipse: usa un KD-tree
    (costruito una sola volta per grafo, poi cachato) per restringere
    subito ai soli nodi entro un raggio che copre l'intera ellisse,
    invece di calcolare la distanza esatta su TUTTI i nodi del grafo.

    Il raggio di query è pari al semiasse maggiore dell'ellisse (metà
    della somma distanza_focale + padding), misurato dal punto medio tra
    source e target — un cerchio che contiene per costruzione l'intera
    ellisse. Il filtro esatto ellittico viene poi applicato solo ai
    candidati trovati dal KD-tree, non a tutto il grafo.

    Su grafi grandi (es. il Veneto, ~240k nodi) questo riduce
    drasticamente il numero di distanze calcolate esplicitamente, perché
    la query del KD-tree è O(log N + k) invece di O(N).

    Risultato identico a extract_subgraph_ellipse (stessa definizione
    geometrica di ellisse), solo più rapido su grafi grandi.
    """
    lat_s, lon_s = G.nodes[source]["y"], G.nodes[source]["x"]
    lat_t, lon_t = G.nodes[target]["y"], G.nodes[target]["x"]

    tree, node_ids, lats_all, lons_all = _get_kdtree(G)

    lat_mid = (lat_s + lat_t) / 2.0
    lon_mid = (lon_s + lon_t) / 2.0
    cos_lat = np.cos(np.radians(lat_mid))

    # Distanza focale (source-target) nella stessa metrica proiettata
    # usata dal filtro esatto (coerente con extract_subgraph_ellipse)
    dlat_focale = lat_t - lat_s
    dlon_focale = (lon_t - lon_s) * cos_lat
    dist_focale_deg = np.sqrt(dlat_focale ** 2 + dlon_focale ** 2)

    padding_deg = padding_km / 111.0
    semiasse_maggiore = (dist_focale_deg + padding_deg) / 2.0

    # La query al KD-tree lavora in gradi grezzi (lat, lon), NON nella
    # metrica proiettata: la longitudine in gradi grezzi "vale di più"
    # della latitudine quando ci si allontana dall'equatore (1° lon <
    # 1° lat in km). Per non escludere erroneamente candidati validi,
    # usiamo un raggio di query maggiorato di 1/cos_lat (il fattore
    # massimo di distorsione possibile), poi il filtro ESATTO successivo
    # (che usa cos_lat locale corretto) elimina i falsi positivi.
    raggio_query_gradi = semiasse_maggiore / max(cos_lat, 0.1)  # cos_lat>0 per l'Italia
    centro_query = np.array([lon_mid, lat_mid])

    idx_candidati = tree.query_ball_point(centro_query, r=raggio_query_gradi * 1.05)

    if not idx_candidati:
        # Fallback di sicurezza: nessun candidato trovato (caso limite,
        # es. source==target o padding troppo piccolo) — ripiega sulla
        # versione esaustiva per garantire correttezza.
        return extract_subgraph_ellipse(G, source, target, padding_km=padding_km)

    idx_candidati = np.array(idx_candidati)
    node_ids_cand = node_ids[idx_candidati]
    lats_cand = lats_all[idx_candidati]
    lons_cand = lons_all[idx_candidati]

    def dist_gradi(lats1, lons1, lat2, lon2):
        dlat = lats1 - lat2
        dlon = (lons1 - lon2) * cos_lat
        return np.sqrt(dlat ** 2 + dlon ** 2)

    dist_s_deg = dist_gradi(lats_cand, lons_cand, lat_s, lon_s)
    dist_t_deg = dist_gradi(lats_cand, lons_cand, lat_t, lon_t)

    # Filtro ellittico ESATTO (stessa formula di extract_subgraph_ellipse),
    # applicato solo ai candidati del KD-tree, non a tutto il grafo
    mask = (dist_s_deg + dist_t_deg) <= (dist_focale_deg + padding_deg)
    valid_nodes = node_ids_cand[mask]

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
