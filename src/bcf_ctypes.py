"""
src/bcf_ctypes.py
====================
Binding diretto (ctypes) alla libreria condivisa libbcf_shared.so, come
alternativa a src/bcf.py (che usa subprocess + file su disco).

Elimina l'overhead di I/O misurato empiricamente: scrittura file, avvio
processo, lettura file — che pesava fino al 66% del tempo BCF totale sui
sottografi piccoli del Sub-Graph Routing (vedi verifica dedicata).

Richiede che libbcf_shared.so sia stato compilato secondo le istruzioni
in bcf_binding.cpp (vedi quel file per i dettagli di build su Colab, dove
il sorgente C++ di negative_weight_SSSP è disponibile — qui in questo
repository Python non c'è, va clonato/compilato a parte).

IMPORTANTE — differenza di comportamento rispetto a src/bcf.py:
bcf_shortest_path() nella libreria C++ restituisce GIA' le distanze finali
sanate (fa internamente sia il calcolo dei potenziali sia il Dijkstra),
non i soli potenziali. Questo significa che con questo binding NON serve
più chiamare src.grafo.sanifica_grafo() né un secondo Dijkstra in Python:
il risultato di esegui_bcf_ctypes() è già la risposta finale.
"""

import ctypes
import os

import numpy as np


class _BCFResult(ctypes.Structure):
    pass  # placeholder, non usiamo struct: gli array sono passati come puntatori


def carica_libreria_bcf(percorso_libreria: str) -> ctypes.CDLL:
    """
    Carica la libreria condivisa compilata (vedi bcf_binding.cpp per le
    istruzioni di build). Da chiamare una sola volta per sessione,
    tipicamente subito dopo compila_bcf() in src/bcf.py.
    """
    if not os.path.exists(percorso_libreria):
        raise FileNotFoundError(
            f"Libreria non trovata: {percorso_libreria}\n"
            "Compilarla seguendo le istruzioni in bcf_binding.cpp "
            "(richiede il sorgente C++ di negative_weight_SSSP)."
        )

    lib = ctypes.CDLL(percorso_libreria)

    # Firma della funzione C esportata:
    # int bcf_shortest_path(int64_t n_nodes, const int64_t* edge_heads,
    #                        const int64_t* edge_tails, const int64_t* edge_weights,
    #                        int64_t n_edges, int64_t source, int64_t* out_distances)
    lib.bcf_shortest_path.argtypes = [
        ctypes.c_int64,                                   # n_nodes
        ctypes.POINTER(ctypes.c_int64),                   # edge_heads
        ctypes.POINTER(ctypes.c_int64),                   # edge_tails
        ctypes.POINTER(ctypes.c_int64),                   # edge_weights
        ctypes.c_int64,                                   # n_edges
        ctypes.c_int64,                                   # source
        ctypes.POINTER(ctypes.c_int64),                   # out_distances
    ]
    lib.bcf_shortest_path.restype = ctypes.c_int

    return lib


def esegui_bcf_ctypes(
    lib: ctypes.CDLL,
    n_nodi_totali: int,
    edge_heads: np.ndarray,
    edge_tails: np.ndarray,
    edge_weights: np.ndarray,
    source: int,
) -> dict:
    """
    Esegue BCF direttamente in memoria via ctypes, senza file né subprocess.

    Parametri:
        lib             : libreria caricata da carica_libreria_bcf()
        n_nodi_totali   : numero di nodi nel grafo (incluso il super-nodo)
        edge_heads      : array numpy int64, nodo di partenza per ogni arco
        edge_tails      : array numpy int64, nodo di arrivo per ogni arco
        edge_weights    : array numpy int64, peso per ogni arco (può essere negativo)
        source          : nodo sorgente (tipicamente il super-nodo artificiale)

    Restituisce:
        dict {nodo_idx -> distanza}, con le distanze finali GIA' sanate
        (equivalenti a quanto si otterrebbe con sanifica_grafo + Dijkstra
        in src/grafo.py, ma calcolate qui internamente da BCF).

    Raises:
        RuntimeError se BCF rileva un ciclo di peso negativo (nessuna
        soluzione valida — non dovrebbe accadere con pesi reali di rete
        stradale, ma è una condizione che la libreria segnala esplicitamente).
    """
    n_edges = len(edge_heads)
    assert len(edge_tails) == n_edges and len(edge_weights) == n_edges, (
        "edge_heads, edge_tails, edge_weights devono avere la stessa lunghezza"
    )

    edge_heads_c = np.ascontiguousarray(edge_heads, dtype=np.int64)
    edge_tails_c = np.ascontiguousarray(edge_tails, dtype=np.int64)
    edge_weights_c = np.ascontiguousarray(edge_weights, dtype=np.int64)
    out_distances = np.zeros(n_nodi_totali, dtype=np.int64)

    ok = lib.bcf_shortest_path(
        ctypes.c_int64(n_nodi_totali),
        edge_heads_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        edge_tails_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        edge_weights_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_int64(n_edges),
        ctypes.c_int64(source),
        out_distances.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )

    if ok == 0:
        raise RuntimeError(
            "BCF ha rilevato un ciclo di peso negativo: nessuna soluzione valida. "
            "Verificare i pesi degli archi passati (edge_weights)."
        )

    INT64_MAX = np.iinfo(np.int64).max
    distanze = {
        i: int(out_distances[i])
        for i in range(n_nodi_totali)
        if out_distances[i] != INT64_MAX
    }
    return distanze


def archi_to_numpy(archi: list) -> tuple:
    """
    Converte la lista di stringhe "u v w\\n" (formato usato da
    src.grafo.costruisci_archi_ridotti / src.bcf.esporta_per_bcf) in tre
    array numpy separati, pronti per esegui_bcf_ctypes().

    Evita completamente il passaggio per file di testo.
    """
    heads, tails, weights = [], [], []
    for riga in archi:
        parti = riga.split()
        if len(parti) == 3:
            heads.append(int(parti[0]))
            tails.append(int(parti[1]))
            weights.append(int(parti[2]))

    return (
        np.array(heads, dtype=np.int64),
        np.array(tails, dtype=np.int64),
        np.array(weights, dtype=np.int64),
    )


def costruisci_archi_per_ctypes(G, y_hat_int: dict, weight_attr: str = "travel_time_d"):
    """
    Variante di src.grafo.costruisci_archi_ridotti pensata per il binding
    diretto (esegui_bcf_ctypes): NON aggiunge il super-nodo artificiale.

    Perché: bcf_shortest_path() nella libreria C++ calcola già un
    single-source shortest path dal source passato esplicitamente — non
    serve più un super-nodo con archi entranti nulli verso tutti i nodi
    per ottenere i potenziali "in un colpo solo" (quello serviva nella
    vecchia pipeline subprocess, che poi doveva rifare un secondo Dijkstra
    separato in Python).

    IMPORTANTE: il super-nodo, se usato come `source` in
    bcf_shortest_path(), produce risultati SBAGLIATI (verificato
    empiricamente: un nodo con grado entrante zero finisce isolato nella
    propria componente durante la decomposizione in SCC interna a BCF,
    con un potenziale calcolato in modo degenere). Con questa funzione,
    passare invece l'indice del nodo SOURCE REALE della query come
    `source` a esegui_bcf_ctypes().

    Restituisce:
        archi       : lista di stringhe "u v w\\n" (SENZA super-nodo)
        nodo_to_idx : {nodo OSM -> indice intero 0-based}
        n_nodi      : numero di nodi reali nel grafo (= len(archi_totali) per out_distances)
        n_negativi  : numero di archi con costo ridotto < 0
    """
    nodi_lista = list(G.nodes())
    nodo_to_idx = {n: i for i, n in enumerate(nodi_lista)}
    n_nodi = len(nodi_lista)

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

    return archi, nodo_to_idx, n_nodi, n_negativi
