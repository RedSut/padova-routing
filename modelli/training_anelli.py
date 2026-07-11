"""
modelli/training_anelli.py
============================
Training generico "ad anelli concentrici": genera un dataset stratificato
per fasce di distanza dal centro, e allena un modello XGBoost con la loss
custom di consistenza (vedi modelli/base.py).

Questa funzione UNIFICA quello che nel notebook originale erano tre celle
quasi identiche (modello a 3 anelli su Padova, a 6 anelli su Padova, e il
modello regionale multicentrico sul Veneto): cambiano solo i parametri
(centro, fasce, target/sorgenti per fascia), non la logica.

Per il modello regionale (multi-centro), vedi training_regionale.py, che
riusa questa stessa funzione iterando su più centri.
"""

import random
import time

import networkx as nx
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from modelli.base import crea_loss_consistenza
from src.grafo import classifica_per_fasce
from src.predizioni import _haversine_vettoriale
from src.feature_avanzate import (
    FEATURE_COLS_AVANZATE,
    estrai_feature_avanzate_vettoriale,
    precalcola_attributi_stradali,
)


def genera_dataset_anelli(
    G: nx.MultiDiGraph,
    centro_lat: float,
    centro_lon: float,
    fasce_km: list[tuple[float, float]],
    target_per_fascia: int = 20,
    sorgenti_per_fascia_per_target: int = 150,
    weight_attr: str = "travel_time_d",
    seed: int = 42,
    nomi_fasce: list[str] | None = None,
    usa_feature_avanzate: bool = False,
) -> pd.DataFrame:
    """
    Genera un dataset di training stratificato su `fasce_km` fasce
    concentriche attorno a (centro_lat, centro_lon).

    Per ogni fascia, sceglie `target_per_fascia` target casuali in quella
    fascia; per ogni target, calcola un singolo Dijkstra single-source
    (invertendo il grafo, per correttezza su archi direzionati) e campiona
    `sorgenti_per_fascia_per_target` sorgenti da OGNI fascia (non solo
    dalla fascia del target) — così il modello vede combinazioni target/
    source a tutte le combinazioni di distanza.

    Restituisce un DataFrame con colonne:
        node_id, target_id, node_lat, node_lon, target_lat, target_lon,
        haversine_dist_m, tempo_reale_s
        (+ node_degree, target_degree, node_road_score, target_road_score,
        heading_deviation_deg se usa_feature_avanzate=True — vedi
        src/feature_avanzate.py)

    usa_feature_avanzate: se True, aggiunge le 5 feature di
        src.feature_avanzate (grado, gerarchia stradale, deviazione di
        heading) oltre alle 5 base. allena_modello_anelli le rileva
        automaticamente dalle colonne presenti e le usa di conseguenza.
    """
    random.seed(seed)
    np.random.seed(seed)

    if nomi_fasce is None:
        nomi_fasce = [f"fascia_{i}" for i in range(len(fasce_km))]

    print("Classificazione dei nodi per fasce concentriche...")
    nodi_per_fascia = classifica_per_fasce(G, centro_lat, centro_lon, fasce_km)
    for i, nome in enumerate(nomi_fasce):
        print(f"  {nome}: {len(nodi_per_fascia[i])} nodi")

    target_selezionati = []
    for i in range(len(fasce_km)):
        nodi_disp = nodi_per_fascia[i]
        campione = random.sample(nodi_disp, min(target_per_fascia, len(nodi_disp)))
        target_selezionati.extend(campione)

    print(f"\nGenerazione dataset con {len(target_selezionati)} target totali...")

    nodes_data = G.nodes(data=True)
    righe_dataset = []
    n_fasce_vuote = 0

    G_rev = G.reverse(copy=False)
    for idx, target in enumerate(target_selezionati):
        distanze = nx.single_source_dijkstra_path_length(
            G_rev, target, weight=weight_attr
        )
        target_lat, target_lon = nodes_data[target]["y"], nodes_data[target]["x"]

        for i in range(len(fasce_km)):
            # Escludiamo il target stesso: se appartiene a questa fascia,
            # finirebbe tra le sorgenti con tempo_reale ≈ 0.
            nodi_disp = [
                n for n in nodi_per_fascia[i] if n in distanze and n != target
            ]

            if not nodi_disp:
                n_fasce_vuote += 1
                continue

            sorgenti = random.sample(
                nodi_disp, min(sorgenti_per_fascia_per_target, len(nodi_disp))
            )

            divisore = {
                "travel_time_s": 1.0, "travel_time_d": 10.0,
                "travel_time_c": 100.0, "travel_time_m": 1000.0,
            }.get(weight_attr, 10.0)

            if usa_feature_avanzate:
                df_feat = estrai_feature_avanzate_vettoriale(G, sorgenti, target, nodes_data)
            else:
                node_lat_arr = np.array([nodes_data[n]["y"] for n in sorgenti])
                node_lon_arr = np.array([nodes_data[n]["x"] for n in sorgenti])
                hav_arr = _haversine_vettoriale(node_lat_arr, node_lon_arr, target_lat, target_lon)
                df_feat = pd.DataFrame(
                    {
                        "node_lat": node_lat_arr, "node_lon": node_lon_arr,
                        "target_lat": target_lat, "target_lon": target_lon,
                        "haversine_dist_m": hav_arr,
                    }
                )

            for idx_s, nodo in enumerate(sorgenti):
                riga = {
                    "node_id": nodo,
                    "target_id": target,
                    "tempo_reale_s": distanze[nodo] / divisore,
                }
                riga.update(df_feat.iloc[idx_s].to_dict())
                righe_dataset.append(riga)

        if (idx + 1) % 10 == 0:
            print(f"  {idx + 1}/{len(target_selezionati)} target completati.")

    df_train = pd.DataFrame(righe_dataset)
    print(f"Dataset generato: {len(df_train)} righe.")
    if n_fasce_vuote > 0:
        print(
            f"⚠️  {n_fasce_vuote} combinazioni target/fascia senza nodi "
            f"disponibili (saltate)."
        )

    return df_train


def _prepara_struttura_vicini(
    G: nx.MultiDiGraph, df_train: pd.DataFrame, weight_attr: str,
    usa_feature_avanzate: bool = False,
):
    """
    Per ogni riga di df_train (identificata da node_id), trova TUTTI gli
    archi reali uscenti (node_id -> vicino) e ne registra peso e feature.
    Necessario per calcolare la penalità di consistenza nella loss custom
    (vedi modelli/base.py) — la loss confronta h(nodo) con h(vicino) per
    OGNI arco uscente, non solo uno a caso: controllare un solo vicino
    (il primo per ordine di inserimento OSM, senza significato) lascia la
    maggior parte degli archi di un nodo di grado >1 mai verificati durante
    il training, qualunque sia lambda_consistenza.

    Restituisce in formato "lungo" (una riga per ARCO, non per nodo):
        df_train        : invariato (nessuna colonna peso_arco_s aggiunta
                           qui, la trovi indirettamente tramite row_id_arr)
        x_vicini_lungo   : DataFrame (n_archi_totali, n_feature) feature
                           del vicino per ciascun arco
        peso_archi_lungo : array (n_archi_totali,) peso reale di ciascun
                           arco
        row_id_arr       : array (n_archi_totali,) indice POSIZIONALE
                           (0..len(df_train)-1) della riga di df_train a
                           cui ciascun arco appartiene

    Nodi senza archi uscenti (vicoli ciechi) non compaiono in row_id_arr:
    la loss li tratta correttamente (nessuna penalità di consistenza per
    loro, dato che non hanno da chi essere "spinti" verso la consistenza).

    NOTA: usa direttamente df_train['node_id'] (e non
    ox.distance.nearest_nodes, che costa ~1s per chiamata ed e' impraticabile
    su migliaia di righe).
    """
    divisore = {
        "travel_time_s": 1.0, "travel_time_d": 10.0,
        "travel_time_c": 100.0, "travel_time_m": 1000.0,
    }.get(weight_attr, 10.0)

    nodes_data = G.nodes(data=True)
    node_ids = df_train["node_id"].to_numpy()
    target_ids = df_train["target_id"].to_numpy()

    # --- Passo 1: per ogni riga di training, elenco TUTTI gli archi uscenti ---
    row_id_list, vicino_list, peso_list = [], [], []
    for row_id, (nodo_id, target_id) in enumerate(zip(node_ids, target_ids)):
        for _, vicino, key, data in G.edges(nodo_id, keys=True, data=True):
            row_id_list.append(row_id)
            vicino_list.append(vicino)
            peso_list.append(data.get(weight_attr, 0) / divisore)

    row_id_arr = np.array(row_id_list, dtype=int)
    peso_archi_lungo = np.array(peso_list, dtype=float)

    if len(row_id_arr) == 0:
        # nessun nodo del dataset ha archi uscenti (caso limite/degenerato)
        colonne = FEATURE_COLS_AVANZATE if usa_feature_avanzate else FEATURE_COLS_BASE
        return df_train, pd.DataFrame(columns=colonne), peso_archi_lungo, row_id_arr

    # --- Passo 2: feature dei vicini, raggruppate per target (il target
    #     cambia da riga a riga in df_train) ---
    if usa_feature_avanzate:
        target_per_arco = target_ids[row_id_arr]
        gruppi: dict = {}
        for idx, (v, t) in enumerate(zip(vicino_list, target_per_arco)):
            gruppi.setdefault(t, []).append((idx, v))

        righe_vicino: list = [None] * len(vicino_list)
        for t, lista in gruppi.items():
            indici, vicini_gruppo = zip(*lista)
            df_feat_gruppo = estrai_feature_avanzate_vettoriale(
                G, list(vicini_gruppo), t, nodes_data
            )
            for pos, idx in enumerate(indici):
                righe_vicino[idx] = df_feat_gruppo.iloc[pos].to_dict()
        x_vicini_lungo = pd.DataFrame(righe_vicino)
    else:
        vicino_lat = np.array([nodes_data[v]["y"] for v in vicino_list])
        vicino_lon = np.array([nodes_data[v]["x"] for v in vicino_list])
        target_lat_per_arco = df_train["target_lat"].to_numpy()[row_id_arr]
        target_lon_per_arco = df_train["target_lon"].to_numpy()[row_id_arr]
        haversine_vicino = _haversine_vettoriale(
            vicino_lat, vicino_lon, target_lat_per_arco, target_lon_per_arco
        )
        x_vicini_lungo = pd.DataFrame(
            {
                "node_lat": vicino_lat, "node_lon": vicino_lon,
                "target_lat": target_lat_per_arco, "target_lon": target_lon_per_arco,
                "haversine_dist_m": haversine_vicino,
            }
        )

    return df_train, x_vicini_lungo, peso_archi_lungo, row_id_arr


def allena_modello_anelli(
    G: nx.MultiDiGraph,
    df_train: pd.DataFrame,
    weight_attr: str = "travel_time_d",
    lambda_consistenza: float = 0.5,
    lambda_iniziale: float | None = None,
    lambda_finale: float | None = None,
    n_round: int = 300,
    test_size: float = 0.2,
    seed: int = 42,
    ogni_n_round_print: int = 20,
) -> tuple[xgb.Booster, list[str], dict]:
    """
    Allena un modello XGBoost con loss custom di consistenza, su un
    dataset già generato da genera_dataset_anelli.

    Se lambda_iniziale e lambda_finale sono entrambi forniti, usa lo
    schema di penalità CRESCENTE (interpolazione lineare tra i due lungo
    n_round) invece di lambda_consistenza fissa -- vedi la docstring di
    crea_loss_consistenza per il perché.

    Restituisce:
        booster        : il modello XGBoost allenato
        feature_cols   : lista delle feature usate (per costruire poi un
                         WrapperXGBoost)
        metriche       : dict con MAE/R² su train e test, per log/confronto
    """
    # Auto-rilevato dalle colonne presenti in df_train (prodotto da
    # genera_dataset_anelli con usa_feature_avanzate=True/False): evita di
    # dover ripetere il flag qui e rischiare un disallineamento.
    from src.feature_avanzate import FEATURE_COLS_EXTRA
    usa_feature_avanzate = all(c in df_train.columns for c in FEATURE_COLS_EXTRA)

    feature_cols = FEATURE_COLS_AVANZATE if usa_feature_avanzate else [
        "node_lat", "node_lon", "target_lat", "target_lon", "haversine_dist_m"
    ]
    print(f"Feature usate ({len(feature_cols)}): {feature_cols}")

    idx_train, idx_test = train_test_split(
        df_train.index, test_size=test_size, random_state=seed
    )

    # Split PRIMA di calcolare la struttura dei vicini: ci serve solo per
    # la parte di training (la loss di consistenza si applica solo li'),
    # e l'indice va resettato a 0..n_train-1 perche' row_id_arr (in
    # _prepara_struttura_vicini) e' posizionale, non basato sull'indice
    # originale del dataset completo.
    df_train_split = df_train.loc[idx_train].reset_index(drop=True)
    df_test_split = df_train.loc[idx_test].reset_index(drop=True)

    print("Recupero vicini (TUTTI gli archi uscenti) per calcolo consistenza...")
    df_train_split, x_vicini_lungo, peso_archi_lungo, row_id_arr = _prepara_struttura_vicini(
        G, df_train_split, weight_attr, usa_feature_avanzate=usa_feature_avanzate
    )
    n_nodi_con_vicini = len(np.unique(row_id_arr)) if len(row_id_arr) else 0
    print(
        f"Vicini trovati per {n_nodi_con_vicini}/{len(df_train_split)} nodi di training "
        f"({len(row_id_arr)} archi totali controllati, media "
        f"{len(row_id_arr)/max(n_nodi_con_vicini,1):.1f} archi/nodo)."
    )

    X_train = df_train_split[feature_cols]
    y_train = df_train_split["tempo_reale_s"]
    X_test = df_test_split[feature_cols]
    y_test = df_test_split["tempo_reale_s"]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    loss_fn, callback_cls = crea_loss_consistenza(
        peso_archi_lungo, x_vicini_lungo, lambda_consistenza=lambda_consistenza,
        row_id_arr=row_id_arr, n_esempi=len(df_train_split),
        lambda_iniziale=lambda_iniziale, lambda_finale=lambda_finale,
        n_round_totali=n_round,
    )

    print(f"\nTraining XGBoost con loss custom (lambda={lambda_consistenza})...")
    booster = xgb.train(
        params={"max_depth": 8, "eta": 0.05, "disable_default_eval_metric": 1},
        dtrain=dtrain,
        num_boost_round=n_round,
        obj=loss_fn,
        callbacks=[callback_cls(ogni_n_round=ogni_n_round_print, n_round_totali=n_round)],
        verbose_eval=False,
    )

    y_pred_train = booster.predict(dtrain)
    y_pred_test = booster.predict(dtest)

    metriche = {
        "mae_train": mean_absolute_error(y_train, y_pred_train),
        "mae_test": mean_absolute_error(y_test, y_pred_test),
        "r2_train": r2_score(y_train, y_pred_train),
        "r2_test": r2_score(y_test, y_pred_test),
    }

    print("\n=== Valutazione modello ===")
    print(f"  MAE  train: {metriche['mae_train']:.1f}s   |  MAE  test: {metriche['mae_test']:.1f}s")
    print(f"  R²   train: {metriche['r2_train']:.3f}     |  R²   test: {metriche['r2_test']:.3f}")

    return booster, feature_cols, metriche
