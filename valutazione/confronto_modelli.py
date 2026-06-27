"""
valutazione/confronto_modelli.py
===================================
Confronto a runtime tra diverse versioni del modello ML.
Supporta l'esportazione dei risultati sia aggregati per fascia (con deviazione standard) 
sia visualizzati coppia per coppia per piccole demo.
"""

import networkx as nx
import numpy as np
import pandas as pd

from src.algoritmi import dijkstra_con_nodi_visitati
from src.bcf import esegui_bcf, esporta_per_bcf
from src.grafo import costruisci_archi_ridotti, sanifica_grafo
from src.predizioni import genera_predizioni


def confronta_modelli_runtime(
    G: nx.MultiDiGraph,
    modelli: dict[str, object],
    coppie: list[tuple],
    bcf_bin: str,
    bcf_input_path: str,
    weight_attr: str = "travel_time_d",
    scale_factor: float = 10.0,
    periodo_giorno: float | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Esegue l'intera pipeline per ciascun modello.
    Restituisce un DataFrame in formato 'tidy' (long format):
    [modello, coppia, fascia, riduzione_pct, trovato]
    """
    print(f"Calcolo baseline Dijkstra su {len(coppie)} coppie...")
    from src.algoritmi import dijkstra_benchmark
    nodi_esplorati_baseline = {}
    distanze_baseline = {}
    for i, dati_coppia in enumerate(coppie):
        nome = dati_coppia[0]
        source = dati_coppia[1]
        target = dati_coppia[2]
        
        distanza, n_nodi = dijkstra_benchmark(G, source, target, weight=weight_attr)
        nodi_esplorati_baseline[nome] = n_nodi
        distanze_baseline[nome] = distanza
        if verbose:
            print(f"Dijkstra vanilla — {nome}: {n_nodi} nodi esplorati, costo {distanza}")
        elif (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(coppie)} baseline completate.")
    print()

    risultati = []

    for nome_modello, modello_corrente in modelli.items():
        print(f"=== Modello: {nome_modello} ===")

        for i, dati_coppia in enumerate(coppie):
            nome = dati_coppia[0]
            source = dati_coppia[1]
            target = dati_coppia[2]
            fascia = dati_coppia[3] if len(dati_coppia) > 3 else "Globale"
            
            try:
                y_hat, y_hat_int = genera_predizioni(
                    G, modello_corrente, target, scale_factor=scale_factor, periodo_giorno=periodo_giorno
                )
                archi, nodo_to_idx, art_idx, n_neg = costruisci_archi_ridotti(
                    G, y_hat_int, weight_attr=weight_attr
                )
                esporta_per_bcf(archi, art_idx, bcf_input_path)
                phi, tempo_bcf_s = esegui_bcf(bcf_bin, bcf_input_path, art_idx, len(G.nodes()))
                G_san = sanifica_grafo(
                    G, y_hat_int, phi, nodo_to_idx, weight_attr=weight_attr
                )

                visitati_sanato = dijkstra_con_nodi_visitati(
                    G_san, source, target, weight=weight_attr
                )
                n_sanato = len(visitati_sanato)
                n_baseline = nodi_esplorati_baseline[nome]
                riduzione_pct = (1 - n_sanato / n_baseline) * 100 if n_baseline > 0 else 0
                
                risultati.append({
                    "modello": nome_modello,
                    "coppia": nome,
                    "fascia": fascia,
                    "riduzione_pct": riduzione_pct,
                    "n_negativi": n_neg,
                    "tempo_bcf_s": tempo_bcf_s,
                    "trovato": True
                })
                if verbose:
                    print(f"  {nome}: {n_sanato} nodi (baseline {n_baseline}) → {riduzione_pct:+.1f}%")
                elif (i + 1) % 100 == 0:
                    print(f"  {i + 1}/{len(coppie)} inferenze completate.")

            except Exception as ex:
                if verbose:
                    print(f"  ❌ {nome}: {ex}")
                risultati.append({
                    "modello": nome_modello,
                    "coppia": nome,
                    "fascia": "N/A",
                    "riduzione_pct": np.nan,
                    "n_negativi": np.nan,
                    "tempo_bcf_s": np.nan,
                    "trovato": False
                })
        print()

    return pd.DataFrame(risultati)


def aggrega_confronto_modelli(df_tidy: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola Media e Deviazione Standard della riduzione percentuale, 
    raggruppate per Modello e Fascia.
    """
    df_ok = df_tidy[df_tidy["trovato"]].copy()
    
    agg = df_ok.groupby(["fascia", "modello"])["riduzione_pct"].agg(
        media="mean",
        std="std",
        n_coppie="count"
    ).reset_index()
    
    return agg


def plot_confronto_modelli_singole_coppie(df_tidy: pd.DataFrame, output_path: str = "confronto_modelli_singoli.png"):
    """
    Grafico originale (legacy): una barra per modello, raggruppata per singola coppia.
    Ottimo per visualizzazioni visive con POCHE coppie manuali (es. < 10).
    Sconsigliato per benchmark stratificati su larga scala.
    """
    import matplotlib.pyplot as plt

    df_ok = df_tidy[df_tidy["trovato"]]
    df_plot = df_ok.pivot(index="coppia", columns="modello", values="riduzione_pct")

    coppie_nomi = list(df_plot.index)
    modelli_nomi = list(df_plot.columns)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(coppie_nomi))
    width = 0.8 / max(len(modelli_nomi), 1)
    colori = plt.cm.Set2(np.linspace(0, 1, len(modelli_nomi)))

    for j, (modello_nome, colore) in enumerate(zip(modelli_nomi, colori)):
        valori = df_plot[modello_nome].values
        ax.bar(x + j * width, valori, width, label=modello_nome, color=colore)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x + width * (len(modelli_nomi) - 1) / 2)
    ax.set_xticklabels(coppie_nomi, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Riduzione % nodi esplorati")
    ax.set_title("Confronto tra Modelli (Singole Coppie)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"Grafico salvato come '{output_path}'")


def plot_confronto_modelli_aggregato(df_tidy: pd.DataFrame, output_path: str = "confronto_modelli_aggregato.png"):
    """
    Nuovo Grafico Analitico: asse X diviso per Fasce Geografiche, 
    barre rappresentano la Media, con linee verticali per la Deviazione Standard (yerr).
    Ideale per benchmark stratificati con 100+ coppie.
    """
    import matplotlib.pyplot as plt

    df_agg = aggrega_confronto_modelli(df_tidy)
    
    fasce_nomi = df_agg["fascia"].unique()
    modelli_nomi = df_agg["modello"].unique()

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(fasce_nomi))
    width = 0.8 / max(len(modelli_nomi), 1)
    colori = plt.cm.Set2(np.linspace(0, 1, len(modelli_nomi)))

    for j, (modello_nome, colore) in enumerate(zip(modelli_nomi, colori)):
        # Estraiamo i valori per il modello corrente nell'ordine delle fasce
        medie = []
        stds = []
        for fascia in fasce_nomi:
            row = df_agg[(df_agg["modello"] == modello_nome) & (df_agg["fascia"] == fascia)]
            if not row.empty:
                medie.append(row["media"].values[0])
                # Riempi i NaN della deviazione standard con 0 (succede se c'è solo 1 coppia)
                std_val = row["std"].values[0]
                stds.append(0 if pd.isna(std_val) else std_val)
            else:
                medie.append(0)
                stds.append(0)
                
        ax.bar(
            x + j * width, medie, width, 
            yerr=stds, capsize=5, 
            label=modello_nome, color=colore, alpha=0.9
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x + width * (len(modelli_nomi) - 1) / 2)
    ax.set_xticklabels(fasce_nomi, fontsize=10)
    ax.set_ylabel("Riduzione Media % ± STD")
    ax.set_title("Confronto tra Modelli (Medie e Deviazioni Standard per Fascia)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"Grafico salvato come '{output_path}'")


def plot_confronto_modelli_boxplot(df_tidy: pd.DataFrame, output_path: str = "confronto_modelli_boxplot.png", y_limit: tuple = None):
    """
    Grafico tramite Boxplot: mostra Mediana, Quartili e Outlier (i "pallini").
    Perfetto per dati con altissima varianza dove la deviazione standard esplode.
    Richiede la libreria 'seaborn'.
    """
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
    except ImportError:
        print("La libreria 'seaborn' non è installata. Usa pip install seaborn.")
        return

    df_ok = df_tidy[df_tidy["trovato"]].copy()

    plt.figure(figsize=(12, 6))
    
    # Crea il boxplot raggruppato per Fascia (asse X) e Modello (colori)
    sns.boxplot(
        data=df_ok,
        x="fascia",
        y="riduzione_pct",
        hue="modello",
        palette="Set2",
        showfliers=True # Mostra gli outlier
    )

    if y_limit is not None:
        plt.ylim(y_limit)

    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.ylabel("Riduzione % nodi esplorati")
    plt.xlabel("")
    plt.title("Confronto tra Modelli - Distribuzione Completa (Boxplot)")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(title="Modello", loc="lower right")
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"Boxplot salvato come '{output_path}'")

def plot_confronto_modelli_archi_negativi(df_tidy: pd.DataFrame, output_path: str = "confronto_modelli_archi_negativi.png"):
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
    except ImportError:
        print("La libreria 'seaborn' non è installata. Usa pip install seaborn.")
        return

    df_ok = df_tidy[df_tidy["trovato"]].copy()

    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=df_ok,
        x="fascia",
        y="n_negativi",
        hue="modello",
        palette="Set2",
        showfliers=False # Meglio omettere outlier giganti per scalare bene
    )
    # Aggiungiamo scala logaritmica perché la differenza è ordini di grandezza
    plt.yscale("symlog", linthresh=10)
    
    plt.ylabel("Numero di Archi Negativi Generati (Log Scale)")
    plt.xlabel("")
    plt.title("Archi Negativi pre-Sanazione (Misura della Consistenza)")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(title="Modello", loc="upper left")
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"Grafico salvato come '{output_path}'")

def plot_confronto_modelli_tempo_bcf(df_tidy: pd.DataFrame, output_path: str = "confronto_modelli_tempo_bcf.png"):
    import matplotlib.pyplot as plt
    try:
        import seaborn as sns
    except ImportError:
        print("La libreria 'seaborn' non è installata. Usa pip install seaborn.")
        return

    df_ok = df_tidy[df_tidy["trovato"]].copy()

    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=df_ok,
        x="fascia",
        y="tempo_bcf_s",
        hue="modello",
        palette="Set2",
        showfliers=False
    )
    plt.yscale("log")
    
    plt.ylabel("Tempo di Sanazione C++ BCF (secondi, Log Scale)")
    plt.xlabel("")
    plt.title("Tempi di Sanazione (Costo degli errori del modello ML)")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(title="Modello", loc="upper left")
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"Grafico salvato come '{output_path}'")
