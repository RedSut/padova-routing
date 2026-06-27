"""
valutazione/benchmark_nodi_esplorati.py
==========================================
Confronto tra Dijkstra vanilla e Dijkstra eseguito sul grafo sanato con i
potenziali predetti.

Questa versione unificata accetta sia coppie manuali che coppie generate 
dinamicamente in base alle fasce geografiche (campionamento stratificato).
Supporta la valutazione dei modelli sul traffico tramite il parametro 
`periodo_giorno`.
"""

import random

import networkx as nx
import pandas as pd

from src.algoritmi import dijkstra_con_nodi_visitati
from src.bcf import esegui_bcf, esporta_per_bcf
from src.grafo import costruisci_archi_ridotti, sanifica_grafo
from src.predizioni import genera_predizioni


def genera_coppie_stratificate(
    nodi_per_fascia: dict[int, list],
    nomi_fasce: list[str],
    tutti_nodi: list,
    n_coppie_per_fascia: int = 100,
    seed: int = 123,
) -> list[tuple[str, object, object, str]]:
    """
    Genera coppie (nome, source, target, fascia) casuali, stratificate per
    fascia del TARGET (il source può essere ovunque nel grafo, simulando
    query realistiche "qualcuno, da qualche parte, verso quella zona").
    """
    random.seed(seed)
    coppie = []

    for fascia_idx, nome_fascia in enumerate(nomi_fasce):
        nodi_disponibili = nodi_per_fascia[fascia_idx]
        if not nodi_disponibili:
            continue
            
        n_generate = 0
        while n_generate < n_coppie_per_fascia:
            tgt = random.choice(nodi_disponibili)
            src = random.choice(tutti_nodi)
            if src == tgt:
                continue
            coppie.append((f"{nome_fascia} #{n_generate + 1}", src, tgt, nome_fascia))
            n_generate += 1

    return coppie


def genera_coppie_per_tipologia(
    nodi_per_fascia: dict[int, list],
    n_coppie_per_tipo: int = 100,
    seed: int = 123,
) -> list[tuple[str, object, object, str]]:
    """
    Genera coppie classificandole per "Tipologia Topologica" del viaggio:
    - Tratte Urbane: Centro -> Centro (Fascia 0 -> Fascia 0)
    - Tratte Miste: Coinvolgono la Periferia (Fascia 1) ma non la Provincia
    - Tratte Lunghe: Coinvolgono la Provincia (Fascia 2)
    """
    random.seed(seed)
    coppie = []
    
    nodi_centro = nodi_per_fascia.get(0, [])
    nodi_periferia = nodi_per_fascia.get(1, [])
    nodi_provincia = nodi_per_fascia.get(2, [])
    tutti_nodi = nodi_centro + nodi_periferia + nodi_provincia
    
    if not nodi_centro or not tutti_nodi:
        return []

    # 1. Tratte Urbane (Centro -> Centro)
    for i in range(n_coppie_per_tipo):
        src = random.choice(nodi_centro)
        tgt = random.choice(nodi_centro)
        while src == tgt: tgt = random.choice(nodi_centro)
        coppie.append((f"Urbana #{i + 1}", src, tgt, "Tratte Urbane"))
        
    # 2. Tratte Miste (Almeno uno in Periferia, l'altro in Centro/Periferia)
    nodi_misti = nodi_centro + nodi_periferia
    for i in range(n_coppie_per_tipo):
        if random.random() > 0.5:
            src = random.choice(nodi_periferia)
            tgt = random.choice(nodi_misti)
        else:
            src = random.choice(nodi_misti)
            tgt = random.choice(nodi_periferia)
        while src == tgt: tgt = random.choice(nodi_misti)
        coppie.append((f"Mista #{i + 1}", src, tgt, "Tratte Miste"))
        
    # 3. Tratte Lunghe (Almeno uno in Provincia)
    for i in range(n_coppie_per_tipo):
        if random.random() > 0.5:
            src = random.choice(nodi_provincia)
            tgt = random.choice(tutti_nodi)
        else:
            src = random.choice(tutti_nodi)
            tgt = random.choice(nodi_provincia)
        while src == tgt: tgt = random.choice(tutti_nodi)
        coppie.append((f"Lunga #{i + 1}", src, tgt, "Tratte Lunghe"))
        
    return coppie


def confronta_nodi_esplorati(
    G: nx.MultiDiGraph,
    model,
    coppie: list[tuple],
    bcf_bin: str,
    bcf_input_path: str,
    weight_attr: str = "travel_time_d",
    scale_factor: float = 10.0,
    periodo_giorno: float | None = None,
    progress_ogni: int = 50,
) -> pd.DataFrame:
    """
    Esegue l'intera pipeline (predizioni -> BCF -> sanazione -> Dijkstra)
    e confronta i nodi esplorati con Dijkstra vanilla per tutte le `coppie`.
    
    Supporta tuple a 3 elementi (nome, source, target) o a 4 elementi 
    (nome, source, target, fascia).
    Il parametro `periodo_giorno` permette di testare i modelli basati sul traffico.
    """
    risultati = []

    for i, dati_coppia in enumerate(coppie):
        nome = dati_coppia[0]
        source = dati_coppia[1]
        target = dati_coppia[2]
        fascia = dati_coppia[3] if len(dati_coppia) > 3 else "N/A"
        
        try:
            n_baseline = len(
                dijkstra_con_nodi_visitati(G, source, target, weight=weight_attr)
            )

            # Usiamo genera_predizioni passando il periodo_giorno in modo dinamico
            y_hat, y_hat_int = genera_predizioni(
                G, model, target, scale_factor=scale_factor, periodo_giorno=periodo_giorno
            )
            
            archi, nodo_to_idx, art_idx, n_neg = costruisci_archi_ridotti(
                G, y_hat_int, weight_attr=weight_attr
            )
            esporta_per_bcf(archi, art_idx, bcf_input_path)
            phi, _ = esegui_bcf(bcf_bin, bcf_input_path, art_idx, len(G.nodes()))
            G_san = sanifica_grafo(G, y_hat_int, phi, nodo_to_idx, weight_attr=weight_attr)

            n_sanato = len(
                dijkstra_con_nodi_visitati(G_san, source, target, weight=weight_attr)
            )
            riduzione_pct = (1 - n_sanato / n_baseline) * 100 if n_baseline > 0 else 0

            risultati.append({
                "coppia": nome, 
                "fascia": fascia,
                "nodi_baseline": n_baseline, 
                "nodi_sanato": n_sanato,
                "riduzione_pct": riduzione_pct, 
                "n_negativi": n_neg, 
                "trovato": True,
            })
            
            # Stampa live per le piccole batch, o ogni N per i campionamenti larghi
            if len(coppie) < 20:
                print(f"{nome}: Dijkstra={n_baseline}, Sanato={n_sanato} ({riduzione_pct:+.1f}%)")
                
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


def aggrega_per_fascia(df_risultati: pd.DataFrame, nomi_fasce: list[str]) -> pd.DataFrame:
    """
    Aggrega i risultati di confronta_nodi_esplorati per fascia, calcolando media, 
    mediana, deviazione standard e percentuale di coppie con riduzione positiva.
    """
    if "fascia" not in df_risultati.columns or all(df_risultati["fascia"] == "N/A"):
        print("Nessuna informazione sulle fasce trovata per l'aggregazione.")
        return pd.DataFrame()
        
    df_ok = df_risultati[df_risultati["trovato"]]

    agg = df_ok.groupby("fascia")["riduzione_pct"].agg(
        media="mean", mediana="median", std="std",
        pct_positive=lambda x: (x > 0).mean() * 100,
        n="count",
    )
    agg = agg.loc[[f for f in nomi_fasce if f in agg.index]]

    print("\n=== RISULTATI AGGREGATI PER FASCIA ===\n")
    print(agg.round(1))

    media_globale = df_ok["riduzione_pct"].mean()
    pct_positive_globale = (df_ok["riduzione_pct"] > 0).mean() * 100
    print(f"\nMedia globale su {len(df_ok)} coppie: {media_globale:+.1f}%")
    print(f"Percentuale di coppie con riduzione positiva: {pct_positive_globale:.1f}%")

    return agg
