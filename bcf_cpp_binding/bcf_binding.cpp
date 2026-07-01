// bcf_binding.cpp
// ================
// Wrapper con interfaccia C attorno a bcf::BCF(Graph&, NodeID), pensato per
// essere compilato come libreria condivisa (.so su Linux/Colab) e chiamato
// da Python via ctypes, SENZA passare da file su disco o da un processo
// subprocess separato — elimina l'overhead di I/O misurato empiricamente
// (~44% del tempo BCF totale, fino al 66% sui sottografi piccoli del
// Sub-Graph Routing).
//
// Da aggiungere alla cartella src/ del repository negative_weight_SSSP e
// compilare insieme agli altri sorgenti (vedi istruzioni di build in fondo
// al file, o il file build_binding.sh fornito separatamente).
//
// NON MODIFICA il codice esistente (main.cpp, queries.cpp, ecc.) — aggiunge
// solo un punto di ingresso alternativo, così il comportamento a riga di
// comando esistente ("./Main ... query_file output_file") resta invariato.

#include <cstdint>
#include <vector>
#include <optional>

#include "graph.h"
#include "algorithms.h"

extern "C" {

/**
 * Calcola il negative-weight single-source shortest path con l'algoritmo
 * BCF, direttamente in memoria (nessun file, nessun processo esterno).
 *
 * Parametri:
 *   n_nodes      : numero totale di nodi nel grafo (incluso il super-nodo,
 *                  se presente — stessa convenzione usata finora: il
 *                  chiamante Python aggiunge già gli archi super_idx -> v
 *                  con peso 0 prima di chiamare questa funzione)
 *   edge_heads   : array di lunghezza n_edges, nodo di partenza di ogni arco
 *   edge_tails   : array di lunghezza n_edges, nodo di arrivo di ogni arco
 *   edge_weights : array di lunghezza n_edges, peso (int64) di ogni arco
 *                  (può essere negativo)
 *   n_edges      : numero di archi
 *   source       : nodo sorgente per il single-source shortest path
 *                  (tipicamente il super-nodo artificiale)
 *   out_distances: array PREALLOCATO dal chiamante, lunghezza n_nodes.
 *                  Viene riempito con le distanze finali (già sanate,
 *                  corrette con i potenziali — non richiede nessun
 *                  passaggio aggiuntivo di sanificazione in Python).
 *                  Le distanze non raggiunte restano al valore sentinella
 *                  INT64_MAX (equivalente di "infinito").
 *
 * Restituisce:
 *   1 se il calcolo è andato a buon fine (nessun ciclo negativo)
 *   0 se è stato rilevato un ciclo di peso negativo (nessuna soluzione)
 *
 * NOTA: questa funzione fa GIA' internamente sia il calcolo dei potenziali
 * SIA il Dijkstra finale (vedi bcf::BCF in algorithms.cpp) — il chiamante
 * Python NON deve più chiamare sanifica_grafo() o eseguire un secondo
 * Dijkstra: out_distances contiene già il risultato finale.
 */
int bcf_shortest_path(
    int64_t n_nodes,
    const int64_t* edge_heads,
    const int64_t* edge_tails,
    const int64_t* edge_weights,
    int64_t n_edges,
    int64_t source,
    int64_t* out_distances
) {
    // Costruzione del grafo in memoria, dagli array passati da Python.
    // FullEdge = std::tuple<NodeID, NodeID, Distance> = (head, tail, weight)
    std::vector<FullEdge> full_edges;
    full_edges.reserve(static_cast<size_t>(n_edges));
    for (int64_t i = 0; i < n_edges; ++i) {
        full_edges.emplace_back(
            static_cast<NodeID>(edge_heads[i]),
            static_cast<NodeID>(edge_tails[i]),
            static_cast<Distance>(edge_weights[i])
        );
    }

    Graph graph(static_cast<NodeID>(n_nodes), full_edges);

    std::optional<Distances> result = BCF(graph, static_cast<NodeID>(source));

    if (!result.has_value()) {
        // Ciclo negativo rilevato: nessuna soluzione valida.
        return 0;
    }

    const Distances& distances = result.value();
    for (int64_t i = 0; i < n_nodes; ++i) {
        out_distances[i] = distances[static_cast<size_t>(i)];
    }

    return 1;
}

}  // extern "C"

/*
=== ISTRUZIONI DI BUILD (verificate contro il CMakeLists.txt reale del repo) ===

Il CMakeLists.txt del repository definisce già una libreria "Library" con
tutti i sorgenti necessari (algorithms.cpp, graph.cpp, bcf.cpp, gor.cpp,
config.cpp, ecc.) — non serve elencarli di nuovo, basta linkare contro
quella libreria già esistente.

1. Copia questo file dentro negative_weight_SSSP/src/bcf_binding.cpp

2. Apri CMakeLists.txt e aggiungi, subito dopo il blocco
   "add_library(Library ...)" esistente, queste due righe:

   add_library(bcf_shared SHARED src/bcf_binding.cpp)
   target_link_libraries(bcf_shared PRIVATE Library)
   target_compile_options(bcf_shared PRIVATE -O3 -fPIC)

   (Il resto del file, inclusi add_executable(Main ...) ecc., resta
   invariato — questo non modifica il comportamento esistente a riga
   di comando.)

3. Rigenera la build (stessi comandi di build.sh, ma target aggiuntivo):
   cd negative_weight_SSSP
   mkdir -p build && cd build
   cmake ..
   make bcf_shared

4. L'output sarà build/libbcf_shared.so (nome esatto verificabile con
   `ls build/*.so` dopo la compilazione).

5. Verifica che il simbolo sia esportato correttamente:
   nm -D build/libbcf_shared.so | grep bcf_shortest_path
   (dovrebbe stampare una riga con "T bcf_shortest_path")

Se la build fallisce per errori di linking mancanti, il problema più
probabile è che "Library" in questo repo sia una libreria STATICA (default
di add_library senza SHARED) — in quel caso il binding potrebbe richiedere
anche -fPIC sui sorgenti di Library stessa. Se necessario, aggiungi
"set_property(TARGET Library PROPERTY POSITION_INDEPENDENT_CODE ON)"
subito dopo la definizione di Library nel CMakeLists.txt.

NOTA su create_graph.cpp: questo file (incluso in "Library" nel
CMakeLists.txt originale) contiene un proprio main(). Questo non dovrebbe
causare conflitti di linking per bcf_shared (la libreria condivisa non
viene mai eseguita come programma, solo caricata via dlopen/ctypes), ma se
il linker segnalasse un "multiple definition of main" inatteso, la
soluzione più semplice è rimuovere create_graph.cpp dalla lista dei
sorgenti di Library usati da bcf_shared, ricompilando invece bcf_shared
con i sorgenti elencati esplicitamente (algorithms.cpp, graph.cpp, bcf.cpp,
gor.cpp, config.cpp — tutti tranne quelli con un proprio main).
*/

