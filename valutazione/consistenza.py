"""
valutazione/consistenza.py
=============================
Strumenti per verificare empiricamente:
  1. Che l'euristica predetta sia "consistente" (verifica_consistenza_*),
     condizione necessaria per l'efficienza di Dijkstra sul grafo sanato: per ogni arco (u,v),
     h(u) <= w(u,v) + h(v)

Risultato chiave dimostrato con questi strumenti (vedi sessioni di
sviluppo): un MAE basso sulla predizione del tempo NON garantisce un'
euristica consistente. Le violazioni tendono a concentrarsi nei rami esplorati e
scartati, non sul percorso ottimale finale — quindi la
percentuale di violazioni misurata SOLO sul percorso finale può essere
ingannevole (può risultare 0% anche quando Dijkstra sul grafo sanato esplora più nodi di
Dijkstra vanilla).
"""

import random

import networkx as nx
import numpy as np

from src.algoritmi import dijkstra_con_nodi_visitati
from src.grafo import costruisci_archi_ridotti
from src.predizioni import genera_predizioni


def sanity_check_segno(
    G: nx.MultiDiGraph,
    model,
    target,
    periodo_giorno: float | None = None,
    scale_factor: float = 10.0,
) -> tuple[int, int]:
    """
    Confronta il numero di archi con costo ridotto negativo ottenuto con
    il segno invertito (quello usato in produzione, vedi
    predizioni.genera_predizioni) rispetto al segno diretto.

    Ci aspettiamo che il segno invertito produca SISTEMATICAMENTE meno
    archi negativi: se non e' cosi', e' un segnale che l'inversione di
    segno andrebbe rivista per il modello in uso.

    Il parametro periodo_giorno va passato quando model e' stato allenato
    con la feature temporale (vedi modelli/pipeline_unificata.py,
    usa_traffico=True) — altrimenti la chiamata a model.predict() fallisce
    per mismatch di colonne.

    Restituisce (n_negativi_invertito, n_negativi_diretto).
    """
    _, y_hat_int_invertito = genera_predizioni(
        G, model, target, periodo_giorno=periodo_giorno, scale_factor=scale_factor
    )
    _, _, _, n_neg_invertito = costruisci_archi_ridotti(G, y_hat_int_invertito)

    y_hat_diretto_raw, _ = genera_predizioni(
        G, model, target, periodo_giorno=periodo_giorno, scale_factor=scale_factor
    )
    y_hat_int_diretto = {n: int(round(-v)) for n, v in y_hat_diretto_raw.items()}
    _, _, _, n_neg_diretto = costruisci_archi_ridotti(G, y_hat_int_diretto)

    print(f"Target di test: {target}")
    print(f"  Archi negativi con segno INVERTITO (produzione): {n_neg_invertito}")
    print(f"  Archi negativi con segno DIRETTO (controllo):    {n_neg_diretto}")
    if n_neg_invertito < n_neg_diretto:
        print("  ✅ Il segno invertito produce meno archi negativi: scelta corretta.")
    else:
        print("  ⚠️  Il segno diretto produce meno (o pari) archi negativi: "
              "rivedere l'inversione di segno per questo modello.")

    return n_neg_invertito, n_neg_diretto


def verifica_consistenza_campione(
    G: nx.MultiDiGraph,
    y_hat_int: dict,
    n_archi_campione: int = 5000,
    weight_attr: str = "travel_time_d",
    scale_factor: float = 10.0,
    seed: int = 42,
) -> dict:
    """
    Verifica la condizione di consistenza h(u) <= w(u,v) + h(v) su un
    campione casuale di archi distribuito su TUTTO il grafo.
    """
    random.seed(seed)
    tutti_archi = list(G.edges(keys=True, data=True))
    archi_campione = random.sample(tutti_archi, min(n_archi_campione, len(tutti_archi)))

    violazioni = []
    for u, v, key, data in archi_campione:
        if u == v:
            continue
        w_uv = data.get(weight_attr, 0)
        h_u = y_hat_int.get(u, 0)
        h_v = y_hat_int.get(v, 0)
        violazione = h_u - (w_uv + h_v)
        if violazione > 0:
            violazioni.append(violazione)

    return _riassumi_violazioni(violazioni, len(archi_campione), "campione casuale (tutto il grafo)", scale_factor)

def verifica_consistenza_percorso(
    G: nx.MultiDiGraph,
    y_hat_int: dict,
    source,
    target,
    weight_attr: str = "travel_time_d",
    scale_factor: float = 10.0,
) -> dict:
    """
    Verifica la consistenza SOLO sugli archi del percorso ottimale (Dijkstra
    vanilla) tra source e target — la zona che conta davvero per quella
    coppia specifica.
    """
    path = nx.dijkstra_path(G, source, target, weight=weight_attr)

    violazioni = []
    for u, v in zip(path[:-1], path[1:]):
        data = G.get_edge_data(u, v)
        key = list(data.keys())[0]
        w_uv = data[key].get(weight_attr, 0)
        h_u = y_hat_int.get(u, 0)
        h_v = y_hat_int.get(v, 0)
        violazione = h_u - (w_uv + h_v)
        if violazione > 0:
            violazioni.append(violazione)

    return _riassumi_violazioni(violazioni, len(path) - 1, "percorso ottimale", scale_factor)

def verifica_consistenza_nodi_visitati(
    G: nx.MultiDiGraph,
    G_san: nx.MultiDiGraph,
    y_hat_int: dict,
    source,
    target,
    weight_attr: str = "travel_time_d",
    scale_factor: float = 10.0,
) -> dict:
    """
    Verifica la consistenza sugli archi USCENTI da tutti i nodi
    effettivamente visitati da Dijkstra sul grafo sanato G_san — non
    solo quelli del percorso finale.

    Questo e' il controllo più informativo: misura le violazioni nei rami
    esplorati e scartati durante la ricerca, dove tipicamente si
    concentra il problema anche quando il percorso finale risulta
    perfettamente consistente.

    NOTA: questa funzione non richiama genera_predizioni — riceve
    y_hat_int già calcolato. Se il modello usato per generarlo richiedeva
    periodo_giorno (modello traffico), assicurarsi che sia stato passato
    a monte, alla chiamata di genera_predizioni che ha prodotto y_hat_int.
    """
    nodi_visitati = dijkstra_con_nodi_visitati(G_san, source, target, weight=weight_attr)

    violazioni = []
    n_archi_controllati = 0
    for u in nodi_visitati:
        h_u = y_hat_int.get(u, 0)
        for _, v, key, data in G.edges(u, keys=True, data=True):
            if u == v:
                continue
            w_uv = data.get(weight_attr, 0)
            h_v = y_hat_int.get(v, 0)
            violazione = h_u - (w_uv + h_v)
            n_archi_controllati += 1
            if violazione > 0:
                violazioni.append(violazione)

    risultato = _riassumi_violazioni(
        violazioni, n_archi_controllati, "nodi visitati sul grafo sanato (rami inclusi)", scale_factor
    )
    risultato["n_nodi_visitati"] = len(nodi_visitati)
    return risultato


def _riassumi_violazioni(violazioni: list, n_totale: int, etichetta: str, scale_factor: float = 10.0) -> dict:
    """Helper interno: stampa e restituisce un riassunto delle violazioni trovate."""
    n_violazioni = len(violazioni)
    pct = (n_violazioni / n_totale) * 100 if n_totale > 0 else 0

    print(f"\n=== Consistenza — {etichetta} ===")
    print(f"Archi testati: {n_totale}")
    print(f"Violazioni:    {n_violazioni}  ({pct:.1f}%)")

    risultato = {"n_totale": n_totale, "n_violazioni": n_violazioni, "pct_violazioni": pct}

    if violazioni:
        v_arr = np.array(violazioni)
        risultato["violazione_media_s"] = v_arr.mean() / scale_factor
        risultato["violazione_massima_s"] = v_arr.max() / scale_factor
        risultato["violazione_mediana_s"] = float(np.median(v_arr)) / scale_factor
        print(f"Violazione media:    {risultato['violazione_media_s']:.2f}s")
        print(f"Violazione massima:  {risultato['violazione_massima_s']:.2f}s")
    else:
        print("✅ Nessuna violazione trovata.")

    return risultato
