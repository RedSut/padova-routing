"""
modelli/base.py
================
Componenti condivisi da tutte le strategie di training: il wrapper che
rende un booster XGBoost compatibile con l'interfaccia scikit-learn
(predict + feature_names_in_), e la FACTORY della loss custom con
penalità di consistenza.

Perché una factory e non una funzione fissa: la loss custom deve "vedere"
i dati di un dataset specifico (X_vicino, peso_arco_arr), che cambiano da
un training all'altro (standard, anelli, 6 anelli, regionale...). Nel
notebook originale questo veniva risolto duplicando la funzione con un
suffisso diverso per ogni training (loss_consistenza_xgb_anelli,
loss_consistenza_xgb_6anelli, ...) — qui usiamo una closure, che crea una
nuova funzione "su misura" senza duplicare codice.
"""

import time

import numpy as np
import xgboost as xgb


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
    peso_arco_arr: np.ndarray, x_vicino_train, lambda_consistenza: float = 0.5,
    row_id_arr: np.ndarray | None = None, n_esempi: int | None = None,
    lambda_iniziale: float | None = None, lambda_finale: float | None = None,
    n_round_totali: int | None = None,
):
    """
    Factory che costruisce una loss custom per XGBoost, parametrizzata sui
    dati del training corrente. Restituisce (loss_fn, callback_class) pronti
    da passare a xgb.train(obj=loss_fn, callbacks=[callback_class()]).

    Formulazione (per ogni esempio i, un nodo u_i rispetto a un target):

        L_i = 0.5*(h(u_i) - tempo_reale_i)^2
            + lambda(t) * media_su_j[ max(0, h(u_i) - w(u_i,v_ij) - h(v_ij)) ]^2

    dove v_i1..v_id sono TUTTI i vicini (successori reali nel grafo) di
    u_i, non solo il primo, e lambda(t) può essere fissa oppure CRESCENTE
    nel corso del training (round t).

    Perché lambda crescente: con lambda fissa alta fin dall'inizio,
    abbiamo misurato empiricamente che le violazioni di consistenza
    PEGGIORANO invece di migliorare (0.5->10.4%, 2.0->11.4%, 10.0->14.9%
    su Padova reale) — sintomo di instabilità da retroazione ritardata:
    h(vicino) nella penalità è la predizione del round PRECEDENTE, quindi
    è un bersaglio che si sposta; inseguirlo aggressivamente fin da subito
    (lambda alta) causa oscillazione invece di convergenza. Partire con
    lambda bassa lascia che le predizioni base si stabilizzino prima
    (l'errore di posizione del bersaglio si riduce), poi si alza
    gradualmente la pressione di consistenza quando il bersaglio si muove
    già molto meno.

    Parametri specifici dello schema adattivo (se lambda_iniziale è
    fornito, sovrascrive lambda_consistenza fissa):
        lambda_iniziale : valore di lambda al round 0
        lambda_finale   : valore di lambda all'ultimo round
                          (interpolazione LINEARE tra i due)
        n_round_totali  : numero totale di round del training, necessario
                          per calcolare l'interpolazione

    Parametri (ereditati, invariati):
        peso_arco_arr  : array (n_archi_totali,) peso reale di CIASCUN
                         arco u_i->v_ij nel formato "lungo"
        x_vicino_train : DataFrame (n_archi_totali, n_feature) feature del
                         vicino v_ij per ciascun arco
        row_id_arr     : array (n_archi_totali,) indice posizionale
                         (0..n_esempi-1) della riga di training a cui
                         ciascun arco appartiene
        n_esempi       : numero di righe di training originali

    Se row_id_arr è None, ricade sul comportamento a un solo vicino per
    riga (retrocompatibilità).

    Restituisce:
        (loss_fn, AggiornaBoosterCallback)
    """
    import pandas as pd

    usa_aggregazione = row_id_arr is not None
    usa_schedule = lambda_iniziale is not None

    if usa_schedule:
        assert lambda_finale is not None and n_round_totali is not None, (
            "lambda_iniziale richiede anche lambda_finale e n_round_totali"
        )

    state = {"booster_attuale": None, "lambda_attuale": lambda_iniziale if usa_schedule else lambda_consistenza}

    def loss_fn(y_pred, dtrain):
        y_true = dtrain.get_label()
        lam = state["lambda_attuale"]

        booster_corrente = state["booster_attuale"]
        if booster_corrente is not None:
            d_vicino = xgb.DMatrix(x_vicino_train.fillna(0))
            h_vicino = booster_corrente.predict(d_vicino)
        else:
            n_archi = len(peso_arco_arr)
            h_vicino = np.zeros(n_archi)

        if not usa_aggregazione:
            soglia = peso_arco_arr + h_vicino
            violazione = np.maximum(0, y_pred - soglia)
            violazione[np.isinf(soglia)] = 0

            grad = (y_pred - y_true) + 2 * lam * violazione
            hess = np.ones_like(y_pred) + 2 * lam * (violazione > 0).astype(float)
            return grad, hess

        h_u_per_arco = y_pred[row_id_arr]
        soglia = peso_arco_arr + h_vicino
        violazione_per_arco = np.maximum(0, h_u_per_arco - soglia)
        violazione_per_arco[np.isinf(soglia)] = 0
        indicatore_per_arco = (violazione_per_arco > 0).astype(float)

        df_agg = pd.DataFrame({
            "row_id": row_id_arr,
            "violazione": violazione_per_arco,
            "indicatore": indicatore_per_arco,
        })
        medie = df_agg.groupby("row_id").mean()

        media_violazione = np.zeros(n_esempi)
        media_indicatore = np.zeros(n_esempi)
        media_violazione[medie.index.to_numpy()] = medie["violazione"].to_numpy()
        media_indicatore[medie.index.to_numpy()] = medie["indicatore"].to_numpy()

        grad = (y_pred - y_true) + 2 * lam * media_violazione
        hess = np.ones_like(y_pred) + 2 * lam * media_indicatore
        return grad, hess

    class AggiornaBoosterCallback(xgb.callback.TrainingCallback):
        """
        Aggiorna il riferimento al booster corrente dopo ogni round di
        boosting, e (se attivo lo schema adattivo) il valore corrente di
        lambda per interpolazione lineare tra lambda_iniziale e
        lambda_finale. Stampa anche un avanzamento periodico.
        """

        def __init__(self, ogni_n_round: int = 20, n_round_totali: int | None = n_round_totali):
            self.ogni_n_round = ogni_n_round
            self.n_round_totali = n_round_totali
            self.t0 = time.time()

        def after_iteration(self, model, epoch, evals_log):
            state["booster_attuale"] = model
            if usa_schedule and self.n_round_totali:
                frazione = min(1.0, (epoch + 1) / self.n_round_totali)
                state["lambda_attuale"] = lambda_iniziale + frazione * (lambda_finale - lambda_iniziale)
            if epoch == 0 or (epoch + 1) % self.ogni_n_round == 0:
                trascorso = time.time() - self.t0
                totale_str = f"/{self.n_round_totali}" if self.n_round_totali else ""
                lam_str = f", lambda={state['lambda_attuale']:.2f}" if usa_schedule else ""
                print(f"  Round {epoch + 1}{totale_str}  ({trascorso:.1f}s trascorsi{lam_str})")
            return False

    return loss_fn, AggiornaBoosterCallback
