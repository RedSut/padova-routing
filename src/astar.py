"""
src/astar.py
============
A* con predizioni ML come euristica, con un parametro `modalita` esplicito
per scegliere tra TRE varianti, con garanzie di correttezza diverse e
costi diversi:

- modalita="chiusa" (VELOCE, NON garantita corretta): il classico A* "da
  manuale" — un nodo, una volta estratto dalla coda, non viene mai più
  riconsiderato. Corretto SOLO se l'euristica è genuinamente ammissibile
  E consistente. Qui NON garantito (la combinazione max(h_euclidea,
  y_hat/fattore) può non essere consistente anche quando è ammissibile —
  abbiamo provato quattro strategie di training per renderla consistente
  e nessuna ha portato le violazioni sotto il 9-10% sul grafo reale di
  Padova). Complessità nel caso peggiore: O((V+E) log V), come Dijkstra.

- modalita="riapertura" (default, SICURA SE AMMISSIBILE): un nodo può
  essere riaperto se si trova un g migliore dopo essere già stato
  estratto. Garantisce la correttezza quando l'euristica è AMMISSIBILE
  (non sovrastima mai il costo vero) anche se non consistente — risolve
  esattamente il problema di "chiusa". ATTENZIONE: se il fattore di
  sicurezza fosse mal calibrato e la componente ML arrivasse a
  sovrastimare il costo vero (euristica francamente NON ammissibile, non
  solo incoerente), nemmeno questa modalità garantisce più la
  correttezza — è un rischio distinto e più severo, verificato con un
  controesempio numerico. Complessità nel caso peggiore: degrada verso
  Bellman-Ford, O(V·E).

- modalita="esaustiva" (SEMPRE SICURA, anche con euristica inammissibile):
  come "riapertura", ma toglie ANCHE la potatura basata su `f > g_target`
  — quella regola presuppone che f sia un lower bound valido, il che
  richiede ammissibilità. Senza quella potatura, l'algoritmo continua
  finché la coda non è vuota, degenerando in una variante di
  Bellman-Ford-Moore guidata dalla coda a priorità: `h` influenza solo
  l'ORDINE di esplorazione, mai se un rilassamento viene saltato. Corretta
  SEMPRE, qualunque sia `h` (anche completamente inammissibile — l'unico
  requisito è che non ci siano cicli negativi nel grafo originale, sempre
  vero per tempi di percorrenza reali). Prezzo: nel caso peggiore la
  complessità è la stessa di "riapertura" (O(V·E)) — non c'è un ulteriore
  peggioramento asintotico rispetto a "riapertura", solo la garanzia è
  più forte.

In tutti e tre i casi: g accumula SEMPRE pesi reali, mai alterati — h
guida solo l'ordine (e in "chiusa"/"riapertura", anche la decisione di
stop). Questo è il motivo per cui g resta sempre affidabile: la domanda
non è mai "il costo calcolato è giusto?", ma "abbiamo esplorato abbastanza
per essere sicuri che non ci sia niente di meglio?".
"""

import heapq

import networkx as nx
import numpy as np

from src.predizioni import _haversine_vettoriale

MODALITA_VALIDE = ("chiusa", "riapertura", "esaustiva")


def _prepara_h_euclidea(G, target, v_max_kmh=130.0):
    """h(v) = distanza_geometrica(v,target) / v_max, in travel_time_d.
    Ammissibile per costruzione (nessuna strada reale supera v_max_kmh)."""
    nodes_data = G.nodes(data=True)
    tgt_lat, tgt_lon = nodes_data[target]["y"], nodes_data[target]["x"]
    v_max_ms = v_max_kmh / 3.6
    nodi_lista = list(G.nodes())
    lats = np.array([nodes_data[n]["y"] for n in nodi_lista])
    lons = np.array([nodes_data[n]["x"] for n in nodi_lista])
    h_arr = (_haversine_vettoriale(lats, lons, tgt_lat, tgt_lon) / v_max_ms) * 10.0
    return dict(zip(nodi_lista, h_arr))


def _astar_nucleo(G, source, target, h_arr, weight, modalita):
    """Nucleo comune alle tre modalita': cambia solo la regola di
    stop/chiusura, la struttura della ricerca e' identica."""
    dist = {source: 0}
    queue = [(h_arr.get(source, 0), 0, source)]
    esplorati = set()

    if modalita == "chiusa":
        chiusi = set()
        while queue:
            f_u, g_u, u = heapq.heappop(queue)
            if u in chiusi:
                continue
            chiusi.add(u)
            esplorati.add(u)
            if u == target:
                return g_u, esplorati
            for _, v, key, data in G.edges(u, keys=True, data=True):
                if v in chiusi:
                    continue
                nuova_g = g_u + data.get(weight, 1)
                if nuova_g < dist.get(v, float("inf")):
                    dist[v] = nuova_g
                    heapq.heappush(queue, (nuova_g + h_arr.get(v, 0), nuova_g, v))
        return dist.get(target, float("inf")), esplorati

    elif modalita == "riapertura":
        g_target = float("inf")
        while queue:
            f_u, g_u, u = heapq.heappop(queue)
            if f_u > g_target:
                break
            if g_u > dist.get(u, float("inf")):
                continue
            esplorati.add(u)
            if u == target:
                g_target = min(g_target, g_u)
                continue
            for _, v, key, data in G.edges(u, keys=True, data=True):
                nuova_g = g_u + data.get(weight, 1)
                if nuova_g < dist.get(v, float("inf")):
                    dist[v] = nuova_g
                    heapq.heappush(queue, (nuova_g + h_arr.get(v, 0), nuova_g, v))
                    if v == target:
                        g_target = min(g_target, nuova_g)
        return dist.get(target, float("inf")), esplorati

    elif modalita == "esaustiva":
        while queue:
            f_u, g_u, u = heapq.heappop(queue)
            if g_u > dist.get(u, float("inf")):
                continue
            esplorati.add(u)
            for _, v, key, data in G.edges(u, keys=True, data=True):
                nuova_g = g_u + data.get(weight, 1)
                if nuova_g < dist.get(v, float("inf")):
                    dist[v] = nuova_g
                    heapq.heappush(queue, (nuova_g + h_arr.get(v, 0), nuova_g, v))
        return dist.get(target, float("inf")), esplorati

    else:
        raise ValueError(f"modalita deve essere una di {MODALITA_VALIDE}, ricevuto: {modalita!r}")


def astar_predizioni(
    G: nx.MultiDiGraph,
    source,
    target,
    h_ml: dict,
    fattore_sicurezza: float = 1.5,
    weight: str = "travel_time_d",
    v_max_kmh: float = 130.0,
    modalita: str = "riapertura",
):
    """
    A* con euristica h(v) = max(h_euclidea(v), h_ml(v)/fattore_sicurezza).

    Parametri:
        G                 : grafo (pesi reali, mai alterati)
        source, target    : nodi di partenza e arrivo
        h_ml              : dict {nodo: predizione ML}, tempo residuo
                             stimato in travel_time_d (positivo)
        fattore_sicurezza : calibrato su un set di validazione, NON un
                             numero a caso
        v_max_kmh         : velocita' massima cautelativa per h_euclidea
        modalita          : "chiusa" | "riapertura" (default) | "esaustiva"
                             -- vedi la docstring del modulo

    Restituisce:
        (distanza_ottima_o_trovata, insieme_nodi_esplorati)
    """
    if modalita not in MODALITA_VALIDE:
        raise ValueError(f"modalita deve essere una di {MODALITA_VALIDE}, ricevuto: {modalita!r}")

    h_euclidea = _prepara_h_euclidea(G, target, v_max_kmh)
    h_arr = {
        n: max(h_euclidea[n], h_ml.get(n, 0) / fattore_sicurezza)
        for n in G.nodes()
    }
    return _astar_nucleo(G, source, target, h_arr, weight, modalita)


if __name__ == "__main__":
    import random

    random.seed(13)
    N = 40
    G = nx.MultiDiGraph()
    for i in range(N):
        G.add_node(i, y=45.0, x=11.0)
    for i in range(N):
        for j in range(N):
            if i != j and random.random() < 0.25:
                G.add_edge(i, j, key=0, travel_time_d=random.randint(1, 15))

    source, target = 0, N - 1
    d_vero = nx.shortest_path_length(G, source, target, weight="travel_time_d")

    dist_vera_a_target = nx.single_source_dijkstra_path_length(
        G.reverse(copy=False), target, weight="travel_time_d"
    )
    h_ml_ammissibile = {n: dist_vera_a_target.get(n, 0) * random.uniform(0.0, 1.0) for n in range(N)}

    print("=== Test 1: euristica ammissibile-ma-non-consistente ===")
    print(f"Distanza vera: {d_vero}")
    for modalita in MODALITA_VALIDE:
        d, e = astar_predizioni(G, source, target, h_ml_ammissibile, fattore_sicurezza=1.0, modalita=modalita)
        print(f"  {modalita:<12}: distanza={d}  nodi={len(e)}  corretto={d == d_vero}")

    print("\n=== Test 2: euristica INAMMISSIBILE (puo' sovrastimare fino a 3x) ===")
    trovato = False
    for seed2 in range(200):
        random.seed(seed2)
        G2 = nx.MultiDiGraph()
        for i in range(N):
            G2.add_node(i, y=45.0, x=11.0)
        for i in range(N):
            for j in range(N):
                if i != j and random.random() < 0.25:
                    G2.add_edge(i, j, key=0, travel_time_d=random.randint(1, 15))
        try:
            d_vero2 = nx.shortest_path_length(G2, source, target, weight="travel_time_d")
        except nx.NetworkXNoPath:
            continue
        dist_vera2 = nx.single_source_dijkstra_path_length(
            G2.reverse(copy=False), target, weight="travel_time_d"
        )
        h_ml_inammissibile = {
            n: dist_vera2.get(n, 0) * (random.uniform(1.3, 3.0) if random.random() < 0.3 else random.uniform(0.5, 0.99))
            for n in range(N)
        }
        d_riap, _ = astar_predizioni(G2, source, target, h_ml_inammissibile, fattore_sicurezza=1.0, modalita="riapertura")
        d_esau, _ = astar_predizioni(G2, source, target, h_ml_inammissibile, fattore_sicurezza=1.0, modalita="esaustiva")
        if d_riap != d_vero2 and d_esau == d_vero2:
            print(f"[seed={seed2}] Distanza vera: {d_vero2}")
            print(f"  riapertura : {d_riap}  corretto={d_riap == d_vero2}  <-- puo' sbagliare")
            print(f"  esaustiva  : {d_esau}  corretto={d_esau == d_vero2}  <-- sempre corretta")
            trovato = True
            break
    if not trovato:
        print("Nessun controesempio trovato in 200 tentativi (improbabile ma possibile)")
