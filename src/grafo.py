"""
src/grafo.py
============
Caricamento del grafo stradale e operazioni sui pesi degli archi:
costruzione degli archi ridotti per BCF, sanazione del grafo con i
potenziali ottenuti, generazione di traffico sintetico per il training.

Dipendenze esterne: osmnx, networkx, joblib, pandas, numpy
"""

import random
import warnings

import joblib
import networkx as nx
import numpy as np
import osmnx as ox

warnings.filterwarnings("ignore")


def carica_ambiente(graphml_path: str, model_path: str | None = None):
    """
    Carica il grafo OSMnx da file e (opzionalmente) un modello ML (supporta pipeline in .joblib o booster nativi in .json per XGBoost).

    Inizializza su ogni arco quattro rappresentazioni del peso, per evitare
    arrotondamenti ripetuti nelle sezioni successive:
        travel_time    : float originale, in secondi
        travel_time_s  : intero, arrotondato al secondo
        travel_time_d  : intero, arrotondato al decimo di secondo (default
                         usato da tutta la pipeline BCF)
        travel_time_c  : intero, arrotondato al centesimo di secondo
        travel_time_m  : intero, arrotondato al millesimo di secondo

    Parametri:
        graphml_path : percorso del file .graphml (es. "padova_drive.graphml")
        model_path   : percorso del modello .joblib o .json (XGBoost).
                       Se None, restituisce model=None (utile quando si vuole
                       solo il grafo, es. per generare un nuovo dataset di
                       training).

    Restituisce:
        (G, model)
    """
    G = ox.load_graphml(graphml_path)

    for u, v, data in G.edges(data=True):
        if "travel_time" in data:
            tt_float = float(data["travel_time"])
            data["travel_time"] = tt_float
            data["travel_time_s"] = int(round(tt_float))
            data["travel_time_d"] = int(round(tt_float * 10.0))
            data["travel_time_c"] = int(round(tt_float * 100.0))
            data["travel_time_m"] = int(round(tt_float * 1000.0))

    model = None
    if model_path is not None:
        if str(model_path).endswith(".json"):
            import xgboost as xgb
            from modelli.base import WrapperXGBoost
            booster = xgb.Booster()
            booster.load_model(model_path)
            # Recupera le feature name dal booster salvato, se presenti, altrimenti usa un default
            f_names = booster.feature_names if booster.feature_names else ["node_lat", "node_lon", "target_lat", "target_lon", "haversine_dist_m"]
            model = WrapperXGBoost(booster, f_names)
        else:
            model = joblib.load(model_path)

    print(f"Grafo caricato: {len(G.nodes())} nodi, {len(G.edges())} archi")
    return G, model


def costruisci_archi_ridotti(
    G: nx.MultiDiGraph, y_hat_int: dict, weight_attr: str = "travel_time_d"
) -> tuple[list, dict, int, int]:
    """
    Costruisce la lista di archi ridotti (formato testo "u v w") per il
    motore BCF, a partire dai potenziali predetti y_hat_int.

    Aggiunge anche gli archi dal super-nodo artificiale (indice = N, il
    numero di nodi reali) verso ogni nodo reale, con peso 0 — necessario
    perché BCF calcola un single-source shortest path dal super-nodo.

    Restituisce:
        archi        : lista di stringhe "u v w\\n"
        nodo_to_idx  : {nodo OSM -> indice intero 0-based}
        super_idx    : indice del super-nodo (= N)
        n_negativi   : numero di archi con costo ridotto < 0
    """
    nodi_lista = list(G.nodes())
    nodo_to_idx = {n: i for i, n in enumerate(nodi_lista)}
    N = len(nodi_lista)
    super_idx = N

    archi = []
    n_negativi = 0

    for u, v, key, data in G.edges(keys=True, data=True):
        if u == v:
            continue

        tempo_base = data.get(weight_attr, 0)
        costo = tempo_base + y_hat_int[u] - y_hat_int[v]

        if costo < 0:
            n_negativi += 1
        archi.append(f"{nodo_to_idx[u]} {nodo_to_idx[v]} {costo}\n")

    for idx in range(N):
        archi.append(f"{super_idx} {idx} 0\n")

    return archi, nodo_to_idx, super_idx, n_negativi


def sanifica_grafo(
    G: nx.MultiDiGraph,
    y_hat_int: dict,
    phi: dict,
    nodo_to_idx: dict,
    weight_attr: str = "travel_time_d",
) -> nx.MultiDiGraph:
    """
    Applica la correzione w*(u,v) = w_ridotto(u,v) + phi[u] - phi[v] >= 0
    a ogni arco, restituendo una copia del grafo con pesi sanati (sempre
    non-negativi).

    NOTA: se un nodo non e' presente in phi (parsing incompleto di BCF),
    viene usato il fallback phi.get(idx, 0) — assicurarsi che esegui_bcf()
    abbia validato il numero di potenziali prima di chiamare questa funzione,
    altrimenti alcuni nodi potrebbero risultare irraggiungibili nel grafo
    sanato senza nessun errore esplicito.
    """
    G_san = G.copy()
    for u, v, key, data in G_san.edges(keys=True, data=True):
        if u == v:
            data["travel_time"] = 0
            continue

        tempo_base = data.get(weight_attr, 0)
        w_rid = tempo_base + y_hat_int[u] - y_hat_int[v]
        phi_u = phi.get(nodo_to_idx[u], 0)
        phi_v = phi.get(nodo_to_idx[v], 0)

        data[weight_attr] = max(0, w_rid + phi_u - phi_v)
    return G_san


def entro_raggio(G: nx.MultiDiGraph, lat_c: float, lon_c: float, raggio_km: float) -> list:
    """
    Restituisce la lista dei nodi di G entro raggio_km dal punto (lat_c, lon_c).

    Usato per filtrare il training a una zona specifica (es. la sola Padova
    città, invece dell'intera area di ~65km coperta dal grafo) o per
    classificare i nodi in fasce concentriche (modelli "ad anelli").
    """
    from src.predizioni import _haversine_vettoriale

    nodi_lista = list(G.nodes())
    nodes_data = G.nodes(data=True)
    lats = np.array([nodes_data[n]["y"] for n in nodi_lista])
    lons = np.array([nodes_data[n]["x"] for n in nodi_lista])

    distanze_km = _haversine_vettoriale(lats, lons, lat_c, lon_c) / 1000.0
    idx_validi = np.where(distanze_km <= raggio_km)[0]
    return [nodi_lista[i] for i in idx_validi]


def classifica_per_fasce(
    G: nx.MultiDiGraph, lat_c: float, lon_c: float, fasce_km: list[tuple[float, float]]
) -> dict[int, list]:
    """
    Classifica ogni nodo di G in una delle fasce concentriche definite da
    `fasce_km` (lista di tuple (raggio_min, raggio_max) in km), in base alla
    distanza dal centro (lat_c, lon_c).

    Restituisce {indice_fascia: [nodi in quella fascia]}.
    Usato dai modelli "ad anelli" (vedi modelli/training_anelli.py).
    """
    from src.predizioni import _haversine_vettoriale

    nodi_lista = list(G.nodes())
    nodes_data = G.nodes(data=True)
    lats = np.array([nodes_data[n]["y"] for n in nodi_lista])
    lons = np.array([nodes_data[n]["x"] for n in nodi_lista])

    distanze_km = _haversine_vettoriale(lats, lons, lat_c, lon_c) / 1000.0

    nodi_array = np.array(list(G.nodes()))

    nodi_per_fascia = {i: [] for i in range(len(fasce_km))}
    for i, (rmin, rmax) in enumerate(fasce_km):
        idx_validi = np.where((distanze_km >= rmin) & (distanze_km < rmax))[0]
        nodi_per_fascia[i] = nodi_array[idx_validi].tolist()

    return nodi_per_fascia


def genera_traffico_realistico(
    G: nx.MultiDiGraph,
    centri: list[tuple[float, float]],
    fattore_centro: float = 1.5,
    fattore_periferia: float = 0.8,
    raggio_centro_km: float = 3.0,
    seed: int | None = None,
) -> nx.MultiDiGraph:
    """
    Applica una penalizzazione radiale ai pesi degli archi, per simulare
    traffico più intenso in prossimità di uno o più centri. Usato per generare 
    dati di training più realistici di quanto offrirebbero i tempi "lisci" calcolati
    da OSMnx (basati solo su velocità massima e lunghezza dell'arco).

    Per ogni centro nella lista, l'arco riceve un fattore moltiplicativo che 
    decresce linearmente con la distanza dal quel centro: fattore_centro vicino 
    al centro, fattore_periferia oltre raggio_centro_km. 
    Se un arco si trova nell'intersezione tra due o più centri, viene applicato 
    il fattore massimo (traffico peggiore).
    Aggiunge anche una piccola componente di rumore casuale per evitare pesi 
    perfettamente deterministici.

    Restituisce una COPIA del grafo con travel_time (e le sue varianti intere)
    aggiornate.
    """
    from src.predizioni import _haversine_vettoriale

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    G_traffico = G.copy()
    nodes_data = G_traffico.nodes(data=True)

    for u, v, key, data in G_traffico.edges(keys=True, data=True):
        if "travel_time" not in data:
            continue

        lat_u, lon_u = nodes_data[u]["y"], nodes_data[u]["x"]
        lat_v, lon_v = nodes_data[v]["y"], nodes_data[v]["x"]
        lat_mid, lon_mid = (lat_u + lat_v) / 2.0, (lon_u + lon_v) / 2.0

        fattore_max = fattore_periferia

        for lat_c, lon_c in centri:
            d_km = _haversine_vettoriale(
                np.array([lat_mid]), np.array([lon_mid]), lat_c, lon_c
            )[0] / 1000.0

            if d_km <= raggio_centro_km:
                frazione = d_km / raggio_centro_km
                fattore = fattore_centro + (fattore_periferia - fattore_centro) * frazione
                if fattore > fattore_max:
                    fattore_max = fattore

        rumore = np.random.uniform(0.95, 1.05)
        tt_nuovo = float(data["travel_time"]) * fattore_max * rumore

        data["travel_time"] = tt_nuovo
        data["travel_time_s"] = int(round(tt_nuovo))
        data["travel_time_d"] = int(round(tt_nuovo * 10.0))
        data["travel_time_c"] = int(round(tt_nuovo * 100.0))
        data["travel_time_m"] = int(round(tt_nuovo * 1000.0))

    return G_traffico
