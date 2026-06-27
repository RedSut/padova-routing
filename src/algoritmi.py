"""
src/algoritmi.py
=================
Algoritmi di ricerca del cammino minimo usati per il benchmark: Dijkstra
con tracciamento dei nodi esplorati (per confrontare Dijkstra vanilla con
Dijkstra eseguito sul grafo sanato).
"""

import heapq
import time

import networkx as nx


def dijkstra_con_nodi_visitati(
    G: nx.MultiDiGraph, source, target, weight: str = "travel_time_d"
) -> set:
    """
    Dijkstra con tracciamento esplicito dei nodi visitati (chiusi durante
    la ricerca), usato per misurare quanti nodi esplora effettivamente
    l'algoritmo — non solo se trova il percorso.

    Eseguito sul grafo originale = Dijkstra vanilla.
    Eseguito sul grafo sanato (dopo sanifica_grafo) = esplorazione guidata
    che sfrutta i potenziali predetti matematicamente.

    Restituisce l'insieme dei nodi visitati (incluso il target, se
    raggiunto). len(risultato) e' la metrica chiave per il confronto.
    """
    queue = [(0, source)]
    dist = {source: 0}
    visited = set()

    while queue:
        d, u = heapq.heappop(queue)
        if u in visited:
            continue
        visited.add(u)
        if u == target:
            break
        for _, v, key, data in G.edges(u, keys=True, data=True):
            if v in visited:
                continue
            costo_arco = data.get(weight, 1)
            nuova_dist = d + costo_arco
            if nuova_dist < dist.get(v, float("inf")):
                dist[v] = nuova_dist
                heapq.heappush(queue, (nuova_dist, v))

    return visited


def dijkstra_benchmark(
    G: nx.MultiDiGraph, source, target, weight: str = "travel_time_d"
) -> tuple[float, int]:
    """
    Variante di dijkstra_con_nodi_visitati che restituisce anche la
    distanza finale, oltre al conteggio dei nodi esplorati.

    Restituisce (distanza, numero_nodi_esplorati).
    """
    visited = dijkstra_con_nodi_visitati(G, source, target, weight=weight)
    # Ricalcola la distanza con networkx (più leggibile che tracciarla a mano)
    try:
        distanza = nx.shortest_path_length(G, source, target, weight=weight)
    except nx.NetworkXNoPath:
        distanza = float("inf")
    return distanza, len(visited)


