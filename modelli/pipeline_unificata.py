"""
modelli/pipeline_unificata.py
================================
Master Pipeline universale per la generazione di dataset e addestramento 
dei modelli ML per il routing SSSP.

Sostituisce e unifica le logiche di:
- training_anelli.py (stratificazione spaziale)
- training_regionale.py (multicentricità)
- dataset_traffico_anelli.py (variabilità temporale e di traffico)

Tutta l'infrastruttura di campionamento e calcolo della consistenza topologica 
è racchiusa qui, permettendo di passare dal modello base a quello avanzato
cambiando semplicemente un paio di flag (es. usa_traffico=True).
"""

import random

import networkx as nx
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from src.grafo import classifica_per_fasce, genera_traffico_realistico
from src.predizioni import _haversine_vettoriale


def genera_dataset_unificato(
    G: nx.MultiDiGraph,
    centri_dict: dict[str, tuple[float, float]],
    fasce_km: list[tuple[float, float]],
    target_per_fascia: int = 15,
    sorgenti_per_fascia_per_target: int = 50,
    weight_attr: str = "travel_time_d",
    usa_traffico: bool = False,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """
    Genera un dataset bilanciato spazialmente (ad anelli) su uno o più centri.
    
    Se usa_traffico=True, pre-genera i grafi per Mattina, Pomeriggio e Sera, 
    ed espande il campionamento includendo la feature 'periodo_giorno' per il ML.
    Se usa_traffico=False, lavora sul grafo standard senza feature temporali.
    
    Restituisce:
        df_train : il DataFrame pronto per XGBoost
        G_periodi: dict mappante periodo -> Grafo (necessario poi per allena_modello)
    """
    random.seed(seed)
    np.random.seed(seed)

    # 1. Setup dei grafi/periodi
    if usa_traffico:
        print("\n[Traffico=ON] Pre-generazione varianti di traffico...")
        centri_list = list(centri_dict.values())
        # FIX: seed propagato esplicitamente a ciascuna chiamata. Senza questo,
        # genera_traffico_realistico riceveva seed=None (il suo default), quindi
        # il rumore casuale (np.random.uniform(0.95, 1.05) per arco) cambiava
        # ad ogni esecuzione del notebook — compromettendo la riproducibilità
        # del dataset di training con traffico. Usiamo seed+1/+2 per i periodi
        # successivi così Mattina/Pomeriggio/Sera non condividono lo stesso
        # identico rumore arco-per-arco.
        G_periodi = {
            0: genera_traffico_realistico(G, centri_list, fattore_centro=2.0, seed=seed),       # Mattina
            1: genera_traffico_realistico(G, centri_list, fattore_centro=1.1, seed=seed + 1),   # Pomeriggio
            2: genera_traffico_realistico(G, centri_list, fattore_centro=1.8, seed=seed + 2),   # Sera
        }
    else:
        print("\n[Traffico=OFF] Uso del grafo standard...")
        G_periodi = {None: G}

    # Pre-computiamo i reverse per velocizzare Dijkstra
    G_rev_dict = {k: v.reverse(copy=False) for k, v in G_periodi.items()}

    dataframes_centri = []
    
    # Le coordinate topologiche (sempre identiche) le prendiamo dal primo grafo disponibile
    primo_grafo = list(G_periodi.values())[0]
    nodes_data = primo_grafo.nodes(data=True)

    # 2. Iterazione sui Centri Geografici
    for nome_centro, (lat_c, lon_c) in centri_dict.items():
        print(f"\n=== Elaborazione Centro: {nome_centro} ===")
        nodi_per_fascia = classifica_per_fasce(primo_grafo, lat_c, lon_c, fasce_km)
        
        target_selezionati = []
        for i in range(len(fasce_km)):
            nodi_disp = nodi_per_fascia[i]
            campione = random.sample(nodi_disp, min(target_per_fascia, len(nodi_disp)))
            target_selezionati.extend(campione)

        righe_dataset = []
        
        # 3. Campionamento per ogni target
        for idx, target in enumerate(target_selezionati):
            target_lat, target_lon = nodes_data[target]["y"], nodes_data[target]["x"]
            
            # Iterazione sui periodi (1 se Traffico=OFF, 3 se Traffico=ON)
            for p_idx, G_rev in G_rev_dict.items():
                distanze = nx.single_source_dijkstra_path_length(G_rev, target, weight=weight_attr)

                # Campionamento sorgenti stratificato per fasce
                for i in range(len(fasce_km)):
                    nodi_disp = [n for n in nodi_per_fascia[i] if n in distanze and n != target]
                    if not nodi_disp:
                        continue

                    sorgenti = random.sample(nodi_disp, min(sorgenti_per_fascia_per_target, len(nodi_disp)))

                    for nodo in sorgenti:
                        node_lat, node_lon = nodes_data[nodo]["y"], nodes_data[nodo]["x"]
                        haversine_dist_m = _haversine_vettoriale(
                            np.array([node_lat]), np.array([node_lon]), target_lat, target_lon
                        )[0]
                        
                        divisore = {
                            "travel_time_s": 1.0, "travel_time_d": 10.0,
                            "travel_time_c": 100.0, "travel_time_m": 1000.0,
                        }.get(weight_attr, 10.0)
                        
                        riga = {
                            "node_id": nodo,
                            "target_id": target,
                            "centro_riferimento": nome_centro,
                            "node_lat": node_lat,
                            "node_lon": node_lon,
                            "target_lat": target_lat,
                            "target_lon": target_lon,
                            "haversine_dist_m": haversine_dist_m,
                            "tempo_reale_s": distanze[nodo] / divisore,
                        }
                        
                        if usa_traffico:
                            riga["periodo_giorno"] = float(p_idx)
                            
                        righe_dataset.append(riga)

            if (idx + 1) % 30 == 0:
                print(f"  {idx + 1}/{len(target_selezionati)} target completati.")

        df_centro = pd.DataFrame(righe_dataset)
        dataframes_centri.append(df_centro)

    # Concatenazione finale di tutti i centri
    df_totale = pd.concat(dataframes_centri, ignore_index=True)
    print(f"\nGenerazione completata: {len(df_totale)} righe totali da {len(centri_dict)} centri.")
    
    return df_totale, G_periodi


def _prepara_struttura_vicini_universale(G_periodi: dict, df_train: pd.DataFrame, weight_attr: str, k_vicini: int = 4):
    """
    Estrae le features dei vicini per calcolare la loss di consistenza.
    Valuta fino a K_VICINI contemporaneamente per applicare la Bellman Equation (Minimo).
    """
    divisore = {
        "travel_time_s": 1.0, "travel_time_d": 10.0,
        "travel_time_c": 100.0, "travel_time_m": 1000.0,
    }.get(weight_attr, 10.0)

    usa_traffico = "periodo_giorno" in df_train.columns

    vicino_per_riga = {k: [] for k in range(k_vicini)}
    peso_arco_per_riga = {k: [] for k in range(k_vicini)}

    for row in df_train.itertuples():
        nodo_id = row.node_id

        periodo = int(row.periodo_giorno) if usa_traffico else None
        G_p = G_periodi[periodo]

        vicini = list(G_p.successors(nodo_id))
        for k in range(k_vicini):
            if k < len(vicini):
                v = vicini[k]
                data_arco = G_p.get_edge_data(nodo_id, v)
                key = list(data_arco.keys())[0]
                peso = data_arco[key].get(weight_attr, 0) / divisore
                vicino_per_riga[k].append(v)
                peso_arco_per_riga[k].append(peso)
            else:
                vicino_per_riga[k].append(None)
                peso_arco_per_riga[k].append(np.inf)

    primo_grafo = list(G_periodi.values())[0]
    nodes_data = primo_grafo.nodes(data=True)

    dati_x_vicino = {}
    peso_arco_matrice = []

    for k in range(k_vicini):
        vicino_lat = np.array([nodes_data[v]["y"] if v is not None else np.nan for v in vicino_per_riga[k]])
        vicino_lon = np.array([nodes_data[v]["x"] if v is not None else np.nan for v in vicino_per_riga[k]])

        haversine_vicino = _haversine_vettoriale(
            vicino_lat, vicino_lon,
            df_train["target_lat"].values, df_train["target_lon"].values,
        )

        dati_x_vicino[f"node_lat_{k}"] = vicino_lat
        dati_x_vicino[f"node_lon_{k}"] = vicino_lon
        dati_x_vicino[f"target_lat_{k}"] = df_train["target_lat"].values
        dati_x_vicino[f"target_lon_{k}"] = df_train["target_lon"].values
        dati_x_vicino[f"haversine_dist_m_{k}"] = haversine_vicino

        if usa_traffico:
            dati_x_vicino[f"periodo_giorno_{k}"] = df_train["periodo_giorno"].values

        peso_arco_matrice.append(peso_arco_per_riga[k])

    x_vicino = pd.DataFrame(dati_x_vicino)
    peso_arco_arr = np.array(peso_arco_matrice).T  # Shape: (N, k_vicini)

    df_train = df_train.copy()

    return df_train, x_vicino, peso_arco_arr


def allena_modello_unificato(
    G_periodi: dict,
    df_train: pd.DataFrame,
    weight_attr: str = "travel_time_d",
    lambda_consistenza: float = 0.5,
    n_round: int = 300,
    test_size: float = 0.2,
    seed: int = 42,
    ogni_n_round_print: int = 50,
    k_vicini: int = 4,
) -> tuple[xgb.Booster, list[str], dict]:
    """
    Allena il modello XGBoost universale.
    L'architettura delle feature si adatta automaticamente ai dati generati.
    """
    print(f"Recupero vicini per calcolo consistenza topologica (k_vicini={k_vicini})...")
    df_train, x_vicino, peso_arco_matrice = _prepara_struttura_vicini_universale(G_periodi, df_train, weight_attr, k_vicini)

    # Auto-rilevamento delle feature
    feature_cols = ["node_lat", "node_lon", "target_lat", "target_lon", "haversine_dist_m"]
    if "periodo_giorno" in df_train.columns:
        feature_cols.append("periodo_giorno")
        print("Feature temporale rilevata: modalità TRAFFICO abilitata.")

    X_full = df_train[feature_cols]
    y_full = df_train["tempo_reale_s"]

    idx_train, idx_test = train_test_split(df_train.index, test_size=test_size, random_state=seed)

    X_train, X_test = X_full.loc[idx_train], X_full.loc[idx_test]
    y_train, y_test = y_full.loc[idx_train], y_full.loc[idx_test]

    pos_train = X_full.index.get_indexer(idx_train)
    peso_arco_arr = peso_arco_matrice[pos_train]

    x_vicino_train = x_vicino.loc[idx_train].reset_index(drop=True)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    loss_fn, callback_cls = crea_loss_consistenza(
        peso_arco_arr, x_vicino_train, lambda_consistenza=lambda_consistenza, k_vicini=k_vicini
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


class WrapperXGBoost:
    """
    Rende un Booster XGBoost compatibile con l'interfaccia scikit-learn,
    cosi' può essere usato come `model` in predizioni.genera_predizioni
    senza modifiche a quella funzione.
    """

    def __init__(self, booster: xgb.Booster, feature_names: list[str]):
        self.booster = booster
        self.feature_names_in_ = np.array(feature_names)

    def predict(self, X):
        X_ordinato = X[self.feature_names_in_] if hasattr(X, "columns") else X
        return self.booster.predict(xgb.DMatrix(X_ordinato))


def crea_loss_consistenza(
    peso_arco_arr: np.ndarray, x_vicino_train, lambda_consistenza: float = 0.5, k_vicini: int = 4
):
    """
    Factory che costruisce una loss custom per XGBoost basata sull'Equazione di Bellman.
    Valuta la disuguaglianza triangolare sul MINIMO tra i vicini (fino a k_vicini).
    """
    import time

    state = {"booster_attuale": None}

    def loss_fn(y_pred, dtrain):
        y_true = dtrain.get_label()

        booster_corrente = state["booster_attuale"]

        soglia_minima = np.inf * np.ones_like(y_pred)

        for k in range(k_vicini):
            peso_k = peso_arco_arr[:, k]

            # Estrai colonne del vicino k e togli il suffisso _k
            cols_k = [c for c in x_vicino_train.columns if c.endswith(f"_{k}")]
            df_k = x_vicino_train[cols_k].rename(columns=lambda x: x[:-2]).fillna(0)

            if booster_corrente is not None:
                d_vicino_k = xgb.DMatrix(df_k)
                h_vicino_k = booster_corrente.predict(d_vicino_k)
            else:
                h_vicino_k = np.zeros_like(y_pred)

            soglia_k = peso_k + h_vicino_k
            soglia_minima = np.minimum(soglia_minima, soglia_k)

        violazione = np.maximum(0, y_pred - soglia_minima)
        violazione[np.isinf(soglia_minima)] = 0  # ignora righe senza alcun vicino

        grad = (y_pred - y_true) + 2 * lambda_consistenza * violazione
        hess = np.ones_like(y_pred) + 2 * lambda_consistenza * (violazione > 0).astype(float)

        return grad, hess

    class AggiornaBoosterCallback(xgb.callback.TrainingCallback):
        """
        Aggiorna il riferimento al booster corrente dopo ogni round di
        boosting, cosi' la loss custom puo' calcolare h(v) con le
        predizioni piu' recenti. Stampa anche un avanzamento periodico,
        perché con la loss custom (che richiama predict() ad ogni round)
        il training può richiedere diversi minuti.
        """

        def __init__(self, ogni_n_round: int = 20, n_round_totali: int | None = None):
            self.ogni_n_round = ogni_n_round
            self.n_round_totali = n_round_totali
            self.t0 = time.time()

        def after_iteration(self, model, epoch, evals_log):
            state["booster_attuale"] = model
            if epoch == 0 or (epoch + 1) % self.ogni_n_round == 0:
                trascorso = time.time() - self.t0
                totale_str = f"/{self.n_round_totali}" if self.n_round_totali else ""
                print(f"  Round {epoch + 1}{totale_str}  ({trascorso:.1f}s trascorsi)")
            return False  # False = continua il training

    return loss_fn, AggiornaBoosterCallback
