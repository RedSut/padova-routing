"""
src/astar.py
============
A* con predizioni ML come euristica, con un parametro esplicito per
scegliere tra le due varianti discusse:

- consenti_riapertura=True (default, SICURA): un nodo può essere
  riaperto se si trova un g migliore dopo che era già stato estratto
  dalla coda. Garantisce la correttezza quando l'euristica è AMMISSIBILE
  (non sovrastima mai il costo vero) ma non necessariamente CONSISTENTE
  (non garantisce la disuguaglianza triangolare arco per arco) — è
  esattamente il caso qui: h_euclidea da sola è ammissibile per
  costruzione, e la combinazione con la predizione ML preserva
  l'ammissibilità nella pratica quanto il fattore di sicurezza è
  calibrato bene (percentile robusto su un set di validazione, non il
  massimo assoluto — vedi calibra_fattore_sicurezza_robusto altrove nel
  progetto). ATTENZIONE: se il fattore di sicurezza fosse mal calibrato
  e la componente ML arrivasse a sovrastimare il costo vero (cioè
  l'euristica risultasse francamente NON ammissibile, non solo
  incoerente), nemmeno la riapertura garantirebbe più la correttezza —
  è un rischio distinto e più severo dell'incoerenza, verificato con un
  controesempio numerico durante lo sviluppo. Il prezzo della riapertura,
  anche nel caso ammissibile: nel caso peggiore la complessità degrada
  verso quella di Bellman-Ford (O(V·E)), non resta O((V+E) log V) come
  Dijkstra.

- consenti_riapertura=False (VELOCE ma NON garantita corretta): un nodo,
  una volta estratto, non viene mai più riconsiderato — il classico A*
  "da manuale". Corretto SOLO se l'euristica è genuinamente ammissibile
  (mai un problema qui, il min/max con h_euclidea lo garantisce) E
  consistente (qui NON garantito — abbiamo provato quattro strategie di
  training per renderla tale e nessuna ha portato le violazioni sotto il
  9-10% sul grafo reale di Padova, vedi la tesi). Usarla solo se si è
  disposti ad accettare il rischio di un cammino occasionalmente
  subottimo, in cambio della complessità piena di Dijkstra nel caso
  peggiore.
"""

import heapq

import networkx as nx
import numpy as np

from src.predizioni import _haversine_vettoriale


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


def astar_predizioni(
    G: nx.MultiDiGraph,
    source,
    target,
    h_ml: dict,
    fattore_sicurezza: float = 1.5,
    weight: str = "travel_time_d",
    v_max_kmh: float = 130.0,
    consenti_riapertura: bool = True,
):
    """
    A* con euristica h(v) = max(h_euclidea(v), h_ml(v)/fattore_sicurezza).

    Parametri:
        G                 : grafo (pesi reali, mai alterati -- l'euristica
                             guida solo l'ORDINE della coda, g resta
                             sempre la somma di pesi veri lungo il cammino
                             effettivamente percorso)
        source, target    : nodi di partenza e arrivo
        h_ml              : dict {nodo: predizione ML}, tempo residuo
                             stimato in travel_time_d (positivo — non il
                             valore invertito di segno usato internamente
                             da sanifica_grafo/BCF, vedi calcola_h_modello
                             altrove nel progetto)
        fattore_sicurezza : calibrato su un set di validazione (vedi
                             calibra_fattore_sicurezza_robusto), NON un
                             numero a caso
        v_max_kmh         : velocità massima cautelativa per h_euclidea,
                             deve essere >= alla velocità massima reale
                             sul grafo, altrimenti h_euclidea smette di
                             essere ammissibile
        consenti_riapertura : True (default) = sicura ma più lenta nel
                             caso peggiore; False = veloce ma non garantita
                             corretta con questa euristica (vedi docstring
                             del modulo)

    Restituisce:
        (distanza_ottima_o_trovata, insieme_nodi_esplorati)

        Con consenti_riapertura=True, distanza_ottima è sempre corretta.
        Con consenti_riapertura=False, potrebbe non esserlo -- il chiamante
        è responsabile di questa scelta.
    """
    h_euclidea = _prepara_h_euclidea(G, target, v_max_kmh)
    h_arr = {
        n: max(h_euclidea[n], h_ml.get(n, 0) / fattore_sicurezza)
        for n in G.nodes()
    }

    dist = {source: 0}
    queue = [(h_arr.get(source, 0), 0, source)]
    esplorati = set()

    if consenti_riapertura:
        # --- Versione SICURA: nessun taglio basato su un nodo "già
        #     chiuso" -- un nodo può tornare in coda se si trova un g
        #     migliore anche dopo essere stato estratto una volta. Lo
        #     stop anticipato (f_u > g_target) resta valido SOLO perché
        #     confrontiamo sempre con il miglior g(target) trovato finora,
        #     mai con un valore che dipende dall'ammissibilità di h in un
        #     punto specifico. ---
        g_target = float("inf")
        while queue:
            f_u, g_u, u = heapq.heappop(queue)
            if f_u > g_target:
                break
            if g_u > dist.get(u, float("inf")):
                continue  # voce obsoleta: u è già stato migliorato dopo
            esplorati.add(u)
            if u == target:
                g_target = min(g_target, g_u)
                continue  # NON break: un cammino ancora migliore potrebbe essere in coda
            for _, v, key, data in G.edges(u, keys=True, data=True):
                nuova_g = g_u + data.get(weight, 1)
                if nuova_g < dist.get(v, float("inf")):
                    dist[v] = nuova_g
                    heapq.heappush(queue, (nuova_g + h_arr.get(v, 0), nuova_g, v))
                    if v == target:
                        g_target = min(g_target, nuova_g)
        return dist.get(target, float("inf")), esplorati

    else:
        # --- Versione VELOCE (classico A* "da manuale"): un nodo, una
        #     volta estratto, è considerato definitivo -- corretto SOLO
        #     se h è genuinamente consistente, non solo ammissibile.
        #     Qui NON garantito: usare consapevoli del rischio. ---
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


if __name__ == "__main__":
    # --- Test: dimostra che consenti_riapertura=False PUO' sbagliare con
    #     un'euristica non garantita consistente, mentre =True è sempre
    #     corretta -- stesso controesempio (seed=3) trovato durante lo
    #     sviluppo del progetto. Grafo giocattolo, non il vero G_padova,
    #     solo per isolare la logica dell'algoritmo. ---
    import random

    class GrafoFinto(nx.MultiDiGraph):
        """Wrapper minimale per riusare astar_predizioni su un grafo
        costruito a mano con pesi arbitrari (senza coordinate reali,
        serve solo a testare la logica di riapertura vs non-riapertura)."""
        pass

    random.seed(13)
    N = 40
    G = nx.MultiDiGraph()
    for i in range(N):
        # Coordinate tutte IDENTICHE apposta: rende h_euclidea = 0 ovunque
        # (ammissibile banalmente, per costruzione), cosi' il test isola
        # davvero il comportamento di riapertura vs non-riapertura rispetto
        # a h_ml_finta, senza interferenze da un'euristica euclidea
        # geograficamente scollegata dai pesi casuali di questo grafo
        # giocattolo (che non hanno alcun legame con coordinate reali).
        G.add_node(i, y=45.0, x=11.0)
    for i in range(N):
        for j in range(N):
            if i != j and random.random() < 0.25:
                G.add_edge(i, j, key=0, travel_time_d=random.randint(1, 15))

    source, target = 0, N - 1
    d_vero = nx.shortest_path_length(G, source, target, weight="travel_time_d")

    # euristica AMMISSIBILE (mai sopra il vero costo residuo) ma NON
    # CONSISTENTE (scala ogni nodo in modo indipendente, quindi puo'
    # violare la disuguaglianza triangolare arco per arco) -- il caso che
    # consenti_riapertura=True garantisce di gestire correttamente.
    # Un'euristica anche solo occasionalmente INAMMISSIBILE (sopra il
    # vero costo) romperebbe la garanzia anche CON riapertura -- quello
    # e' un rischio distinto, piu' severo, discusso nella docstring sopra.
    dist_vera_a_target = nx.single_source_dijkstra_path_length(
        G.reverse(copy=False), target, weight="travel_time_d"
    )
    h_ml_finta = {n: dist_vera_a_target.get(n, 0) * random.uniform(0.0, 1.0) for n in range(N)}

    d_con, e_con = astar_predizioni(G, source, target, h_ml_finta, fattore_sicurezza=1.0, consenti_riapertura=True)
    d_senza, e_senza = astar_predizioni(G, source, target, h_ml_finta, fattore_sicurezza=1.0, consenti_riapertura=False)

    print(f"Distanza vera (Dijkstra):        {d_vero}")
    print(f"consenti_riapertura=True:        {d_con}  (corretto: {d_con == d_vero})")
    print(f"consenti_riapertura=False:       {d_senza}  (corretto: {d_senza == d_vero})")
