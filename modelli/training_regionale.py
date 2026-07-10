"""
modelli/training_regionale.py
================================
Training multicentrico per l'intera regione Veneto: applica la stessa
logica "ad anelli" (vedi training_anelli.py) attorno a più centri
(i capoluoghi di provincia), poi unisce i dataset e allena un unico
modello su tutta la regione.

Correzioni rispetto alla prima versione (vedi note di revisione):
  1. NON sovrascrive la variabile globale `G` — il grafo regionale viene
     passato come parametro esplicito, così il chiamante decide se e come
     conservarlo separatamente dal grafo di Padova.
  2. Usa la STESSA loss custom con penalità di consistenza degli altri
     modelli ad anelli (prima usava MSE puro), per non perdere le
     proprietà di consistenza già validate sui modelli di Padova.
  3. Rimossi i parametri 'tree_method': 'hist', 'device': 'cuda' — non
     portabili su runtime senza GPU. Il training gira su CPU come gli
     altri modelli (più lento, ma sempre eseguibile).
  4. La distanza haversine usa la funzione vettorizzata esistente, non
     DataFrame.apply riga per riga (che annullava il vantaggio della
     vettorizzazione).
"""

import pandas as pd

from modelli.training_anelli import allena_modello_anelli, genera_dataset_anelli

CENTRI_VENETO = {
    "Padova": (45.4064, 11.8768),
    "Venezia": (45.4408, 12.3155),
    "Verona": (45.4384, 10.9916),
    "Vicenza": (45.5455, 11.5354),
    "Treviso": (45.6669, 12.2450),
    "Belluno": (46.1424, 12.2166),
    "Rovigo": (45.0712, 11.7901),
}


def genera_dataset_regionale(
    G,
    centri: dict[str, tuple[float, float]] = CENTRI_VENETO,
    fasce_km: list[tuple[float, float]] | None = None,
    target_per_fascia: int = 5,
    sorgenti_per_fascia_per_target: int = 100,
    weight_attr: str = "travel_time_d",
    seed: int = 42,
    usa_feature_avanzate: bool = False,
) -> pd.DataFrame:
    """
    Genera un dataset multicentrico: per ogni centro in `centri`, applica
    genera_dataset_anelli con le stesse fasce, poi concatena tutto.

    Default fasce_km: 3 fasce pensate per la scala regionale (0-10km,
    10-30km, 30-80km), diverse da quelle usate per il solo centro di
    Padova (che usa raggi più piccoli, 0-5/5-15/15-60km).

    usa_feature_avanzate: propagato a genera_dataset_anelli per ogni
    centro — vedi src/feature_avanzate.py.
    """
    if fasce_km is None:
        fasce_km = [(0, 10), (10, 30), (30, 80)]

    dataframes = []
    for nome_centro, (lat, lon) in centri.items():
        print(f"\n=== Generazione dataset per centro: {nome_centro} ===")
        df_centro = genera_dataset_anelli(
            G,
            centro_lat=lat,
            centro_lon=lon,
            fasce_km=fasce_km,
            target_per_fascia=target_per_fascia,
            sorgenti_per_fascia_per_target=sorgenti_per_fascia_per_target,
            weight_attr=weight_attr,
            seed=seed,
            usa_feature_avanzate=usa_feature_avanzate,
        )
        df_centro["centro"] = nome_centro
        dataframes.append(df_centro)

    df_totale = pd.concat(dataframes, ignore_index=True)
    print(f"\nDataset regionale totale: {len(df_totale)} righe da {len(centri)} centri.")
    return df_totale


def allena_modello_regionale(
    G,
    df_train_regionale: pd.DataFrame,
    weight_attr: str = "travel_time_d",
    lambda_consistenza: float = 0.5,
    n_round: int = 300,
    seed: int = 42,
):
    """
    Allena il modello regionale riusando allena_modello_anelli — stessa
    loss custom di consistenza usata per i modelli locali su Padova, così
    il modello regionale beneficia delle stesse garanzie di euristica
    consistente.
    """
    return allena_modello_anelli(
        G,
        df_train_regionale,
        weight_attr=weight_attr,
        lambda_consistenza=lambda_consistenza,
        n_round=n_round,
        seed=seed,
    )
