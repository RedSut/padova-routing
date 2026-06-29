"""
valutazione/benchmark_subgraph.py
====================================
Confronto del tempo wall-clock end-to-end (predizioni ML + BCF + Dijkstra
finale) tra la pipeline completa (predizioni e BCF su tutto il grafo) e la
pipeline con Sub-Graph Routing + interpolazione spaziale (predizioni solo
sulle ancore dentro un sottografo ritagliato, BCF solo su quel sottografo,
più piccolo).

Riferimento di confronto: Dijkstra vanilla puro (nessuna predizione), per
vedere quanto le due pipeline si avvicinano al limite "gratuito" di un
algoritmo senza overhead di machine learning.
"""

import time

import networkx as nx
import pandas as pd

from src.algoritmi import dijkstra_benchmark, dijkstra_con_nodi_visitati
from src.bcf import esegui_bcf, esporta_per_bcf
from src.grafo import costruisci_archi_ridotti, sanifica_grafo
from src.interpolazione import genera_predizioni_interpolate
from src.predizioni import genera_predizioni
from src.subgraph import assicura_connessione, extract_subgraph_ellipse


def confronta_wallclock_subgraph_interpolazione(
    G_completo: nx.MultiDiGraph,
    G_sottografo: nx.MultiDiGraph,
    model,
    source,
    target,
    bcf_bin: str,
    bcf_input_path: str,
    weight_attr: str = "travel_time_d",
    sample_ratio: float = 0.1,
    seed: int = 42,
) -> dict:
    """
    Esegue e cronometra tre scenari sulla stessa coppia (source, target):

      1. Dijkstra vanilla puro su G_completo (nessuna predizione) — riferimento
      2. Pipeline completa: predizioni + BCF + sanazione su TUTTO G_completo
      3. Ritaglio + interpolazione: predizioni solo sulle ancore di
         G_sottografo (già ritagliato, es. con src.subgraph.extract_subgraph_ellipse),
         BCF solo sul sottografo

    Restituisce un dict con tempi e nodi esplorati per ciascuno scenario,
    più lo speedup tra pipeline completa e ritaglio+interpolazione.
    """
    # 1. Riferimento: Dijkstra vanilla
    t0 = time.time()
    _, n_nodi_vanilla = dijkstra_benchmark(G_completo, source, target, weight=weight_attr)
    t_vanilla = time.time() - t0

    # 2. Pipeline completa sul grafo intero
    t0 = time.time()
    y_hat_full, y_hat_int_full = genera_predizioni(G_completo, model, target)
    archi_full, nodo_to_idx_full, art_idx_full, _ = costruisci_archi_ridotti(
        G_completo, y_hat_int_full, weight_attr=weight_attr
    )
    esporta_per_bcf(archi_full, art_idx_full, bcf_input_path)
    phi_full, _ = esegui_bcf(bcf_bin, bcf_input_path, art_idx_full, len(G_completo.nodes()))
    G_san_full = sanifica_grafo(
        G_completo, y_hat_int_full, phi_full, nodo_to_idx_full, weight_attr=weight_attr
    )
    _, n_nodi_full = dijkstra_benchmark(G_san_full, source, target, weight=weight_attr)
    t_full = time.time() - t0

    # 3. Ritaglio + interpolazione sul sottografo già estratto
    t0 = time.time()
    y_hat_interp, y_hat_int_interp = genera_predizioni_interpolate(
        G_sottografo, model, target, sample_ratio=sample_ratio, seed=seed
    )
    archi_sub, nodo_to_idx_sub, art_idx_sub, _ = costruisci_archi_ridotti(
        G_sottografo, y_hat_int_interp, weight_attr=weight_attr
    )
    esporta_per_bcf(archi_sub, art_idx_sub, bcf_input_path)
    phi_sub, _ = esegui_bcf(bcf_bin, bcf_input_path, art_idx_sub, len(G_sottografo.nodes()))
    G_san_sub = sanifica_grafo(
        G_sottografo, y_hat_int_interp, phi_sub, nodo_to_idx_sub, weight_attr=weight_attr
    )
    _, n_nodi_sub = dijkstra_benchmark(G_san_sub, source, target, weight=weight_attr)
    t_interp = time.time() - t0

    risultato = {
        "t_vanilla_s": t_vanilla, "n_nodi_vanilla": n_nodi_vanilla,
        "t_pipeline_completa_s": t_full, "n_nodi_pipeline_completa": n_nodi_full,
        "t_ritaglio_interp_s": t_interp, "n_nodi_ritaglio_interp": n_nodi_sub,
        "speedup_completa_su_interp": t_full / t_interp if t_interp > 0 else float("inf"),
        "rapporto_interp_su_vanilla": t_interp / t_vanilla if t_vanilla > 0 else float("inf"),
    }

    print(f"=== Confronto wall-clock ===\n")
    print(f"  Dijkstra vanilla (riferimento):       {t_vanilla:.4f}s  | {n_nodi_vanilla} nodi esplorati")
    print(f"  Pipeline completa (tutto il grafo):    {t_full:.4f}s  | {n_nodi_full} nodi esplorati")
    print(f"  Ritaglio ellisse + interpolazione:      {t_interp:.4f}s  | {n_nodi_sub} nodi esplorati")
    print(f"\n  Speedup pipeline completa -> ritaglio+interp: {risultato['speedup_completa_su_interp']:.2f}x")
    print(
        f"  Rapporto rispetto a Dijkstra vanilla:        "
        f"{risultato['rapporto_interp_su_vanilla']:.2f}x (1.0x = stessa velocità di Dijkstra puro)"
    )

    return risultato


def confronta_nodi_esplorati_con_subgraph(
    G_completo: nx.MultiDiGraph,
    model,
    coppie: list[tuple],
    bcf_bin: str,
    bcf_input_path: str,
    padding_km: float = 5.0,
    weight_attr: str = "travel_time_d",
    scale_factor: float = 10.0,
    progress_ogni: int = 5,
) -> pd.DataFrame:
    """
    Versione di confronta_nodi_esplorati (vedi
    valutazione.benchmark_nodi_esplorati) che, per ciascuna coppia, ritaglia
    prima un sottografo ellittico tra source e target (vedi
    src.subgraph.extract_subgraph_ellipse) ed esegue predizioni + BCF solo
    su quel sottografo, invece che sull'intero grafo completo.

    Pensata per grafi grandi (es. il Veneto, ~240k nodi) dove BCF sembra
    scalare peggio del lineare con la dimensione del grafo: ritagliare
    riduce drasticamente il costo di ogni singola chiamata, a fronte di un
    piccolo overhead aggiuntivo per il ritaglio stesso (che e' O(N) sul
    grafo completo, molto più leggero di un Dijkstra o di BCF).

    Il numero di nodi esplorati da Dijkstra vanilla resta calcolato sul
    grafo COMPLETO (per un confronto equo — Dijkstra vanilla non beneficia
    di nessun ritaglio), mentre A* viene eseguito solo sul sottografo.

    Restituisce un DataFrame con le stesse colonne di confronta_nodi_esplorati,
    più la dimensione del sottografo per ciascuna coppia (utile per
    verificare quanto effettivamente si riduce il problema).
    """
    risultati = []

    for i, dati_coppia in enumerate(coppie):
        nome = dati_coppia[0]
        source = dati_coppia[1]
        target = dati_coppia[2]
        fascia = dati_coppia[3] if len(dati_coppia) > 3 else "N/A"

        try:
            n_baseline = len(
                dijkstra_con_nodi_visitati(G_completo, source, target, weight=weight_attr)
            )

            G_sub = extract_subgraph_ellipse(G_completo, source, target, padding_km=padding_km)
            G_sub = assicura_connessione(G_sub, G_completo, source, target)
            n_nodi_sub = len(G_sub.nodes())

            y_hat, y_hat_int = genera_predizioni(G_sub, model, target, scale_factor=scale_factor)
            archi, nodo_to_idx, art_idx, n_neg = costruisci_archi_ridotti(
                G_sub, y_hat_int, weight_attr=weight_attr
            )
            esporta_per_bcf(archi, art_idx, bcf_input_path)
            phi, _ = esegui_bcf(bcf_bin, bcf_input_path, art_idx, len(G_sub.nodes()))
            G_san = sanifica_grafo(G_sub, y_hat_int, phi, nodo_to_idx, weight_attr=weight_attr)

            n_sanato = len(
                dijkstra_con_nodi_visitati(G_san, source, target, weight=weight_attr)
            )
            riduzione_pct = (1 - n_sanato / n_baseline) * 100 if n_baseline > 0 else 0

            risultati.append({
                "coppia": nome, "fascia": fascia,
                "nodi_baseline": n_baseline, "nodi_sanato": n_sanato,
                "n_nodi_sottografo": n_nodi_sub,
                "pct_grafo_originale": (n_nodi_sub / len(G_completo.nodes())) * 100,
                "riduzione_pct": riduzione_pct, "n_negativi": n_neg, "trovato": True,
            })

            if len(coppie) < 20:
                print(
                    f"{nome}: Dijkstra={n_baseline}, Sanato={n_sanato} ({riduzione_pct:+.1f}%) "
                    f"| sottografo={n_nodi_sub} nodi ({(n_nodi_sub/len(G_completo.nodes()))*100:.1f}% del totale)"
                )

        except Exception as ex:
            if len(coppie) < 20:
                print(f"❌ {nome}: {ex}")
            risultati.append({
                "coppia": nome, "fascia": fascia, "trovato": False, "errore": str(ex)
            })

        if len(coppie) >= 20 and (i + 1) % progress_ogni == 0:
            print(f"  {i + 1}/{len(coppie)} coppie processate...")

    df = pd.DataFrame(risultati)
    n_falliti = (~df["trovato"]).sum() if "trovato" in df.columns and len(df) > 0 else 0
    if n_falliti > 0:
        print(f"\n⚠️  {n_falliti} coppie fallite (escluse dall'analisi).")

    return df
