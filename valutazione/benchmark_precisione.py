"""
valutazione/benchmark_precisione.py
======================================
Confronto del tempo di esecuzione di BCF al variare della precisione di
arrotondamento dei pesi (secondi/decimi/centesimi/millesimi di secondo).

Questo script serve a dimostrare empiricamente che il passaggio ai decimi 
(travel_time_d) offre il compromesso perfetto: 
1. Non perde la precisione dei tempi di percorrenza.
2. Evita casi patologici del BCF: se si usano i "secondi" puri, si perdono
   troppi decimali, molti archi ottengono pesi identici (es. tutti arrotondati a 1s o 2s)
   e questo confonde l'algoritmo di scomposizione in cluster del BCF, rendendolo 
   paradossalmente più lento.
3. Centesimi e millesimi aumentano eccessivamente la scala dei pesi, rallentando BCF.
"""

import networkx as nx
import pandas as pd

from src.bcf import esegui_bcf, esporta_per_bcf
from src.grafo import costruisci_archi_ridotti, sanifica_grafo
from src.predizioni import genera_predizioni

CONFIGURAZIONI_PRECISIONE_DEFAULT = [
    {"nome": "Secondi (travel_time_s)", "weight_attr": "travel_time_s", "scale_factor": 1.0},
    {"nome": "Decimi (travel_time_d)", "weight_attr": "travel_time_d", "scale_factor": 10.0},
    {"nome": "Centesimi (travel_time_c)", "weight_attr": "travel_time_c", "scale_factor": 100.0},
    {"nome": "Millesimi (travel_time_m)", "weight_attr": "travel_time_m", "scale_factor": 1000.0},
]


def benchmark_precisione(
    G: nx.MultiDiGraph,
    model,
    coppie: list[tuple[str, object, object]],
    bcf_bin: str,
    bcf_input_path: str,
    configurazioni: list[dict] | None = None,
    periodo_giorno: float | None = None,
) -> pd.DataFrame:
    """
    Esegue l'intera pipeline (predizioni -> BCF -> Dijkstra) per ciascuna
    configurazione di precisione in `configurazioni`, su tutte le coppie.

    Restituisce un DataFrame con una riga per (configurazione, coppia),
    pronto per essere aggregato con .groupby("configurazione").mean().
    """
    if configurazioni is None:
        configurazioni = CONFIGURAZIONI_PRECISIONE_DEFAULT

    risultati = []

    for cfg in configurazioni:
        print(f"--- Configurazione: {cfg['nome']} ---")
        for dati_coppia in coppie:
            nome = dati_coppia[0]
            source = dati_coppia[1]
            target = dati_coppia[2]
            
            try:
                import time

                t0 = time.time()
                y_hat, y_hat_int = genera_predizioni(
                    G, model, target, scale_factor=cfg["scale_factor"], periodo_giorno=periodo_giorno
                )
                t_pred = time.time() - t0

                archi, nodo_to_idx, art_idx, n_neg = costruisci_archi_ridotti(
                    G, y_hat_int, weight_attr=cfg["weight_attr"]
                )

                esporta_per_bcf(archi, art_idx, bcf_input_path)

                t0 = time.time()
                phi, t_bcf_puro = esegui_bcf(
                    bcf_bin, bcf_input_path, art_idx, len(G.nodes())
                )
                t_bcf_totale = time.time() - t0

                G_san = sanifica_grafo(
                    G, y_hat_int, phi, nodo_to_idx, weight_attr=cfg["weight_attr"]
                )

                t0 = time.time()
                nx.dijkstra_path(G_san, source, target, weight=cfg["weight_attr"])
                t_dijk = time.time() - t0

                risultati.append(
                    {
                        "configurazione": cfg["nome"], "coppia": nome,
                        "t_pred_s": round(t_pred, 4),
                        "t_bcf_puro_s": round(t_bcf_puro, 4),
                        "t_bcf_totale_s": round(t_bcf_totale, 4),
                        "t_dijkstra_s": round(t_dijk, 4),
                        "n_negativi": n_neg, "trovato": True,
                    }
                )
                print(f"  {nome}: BCF puro={t_bcf_puro:.4f}s, n_negativi={n_neg}")

            except Exception as ex:
                print(f"  ❌ {nome}: {ex}")
                risultati.append(
                    {"configurazione": cfg["nome"], "coppia": nome, "trovato": False}
                )
        print()

    return pd.DataFrame(risultati)
