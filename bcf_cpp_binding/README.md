# Binding diretto C++ per BCF (senza I/O di file/subprocess)

Questa cartella contiene il codice per eliminare l'overhead di comunicazione
Python↔C++ misurato empiricamente nella pipeline principale (scrittura file
→ subprocess → lettura file), che pesa fino al 66% del tempo BCF totale sui
sottografi piccoli del Sub-Graph Routing.

## Come funziona

Invece di lanciare l'eseguibile `Main` come processo esterno (l'approccio
in `src/bcf.py`), compiliamo BCF come **libreria condivisa** (`.so`) e la
chiamiamo direttamente da Python con `ctypes` — i dati passano come array
in memoria, senza mai toccare il filesystem.

## Passi per attivarlo (da fare su Colab, dove il sorgente C++ è disponibile)

1. **Clona ed esegui la build normale** di `negative_weight_SSSP`, come
   già fate in `src/bcf.compila_bcf()`.

2. **Copia `bcf_binding.cpp`** dentro `negative_weight_SSSP/src/`.

3. **Modifica `CMakeLists.txt`** del repo BCF seguendo le istruzioni
   scritte in fondo a `bcf_binding.cpp` (aggiunta di un target
   `bcf_shared`, poche righe).

4. **Ricompila**, ottenendo `negative_weight_SSSP/build/libbcf_shared.so`.

5. **Usa `src/bcf_ctypes.py`** nel resto della pipeline Python al posto
   di `src/bcf.py`:

   ```python
   from src.bcf_ctypes import carica_libreria_bcf, esegui_bcf_ctypes, archi_to_numpy

   lib = carica_libreria_bcf(f"{BCF_DIR}/build/libbcf_shared.so")

   # archi è la lista di stringhe "u v w" già prodotta da
   # costruisci_archi_ridotti — nessun cambiamento lì
   edge_heads, edge_tails, edge_weights = archi_to_numpy(archi)

   distanze = esegui_bcf_ctypes(
       lib, n_nodi_totali=art_idx + 1,
       edge_heads=edge_heads, edge_tails=edge_tails, edge_weights=edge_weights,
       source=art_idx,
   )
   ```

## Differenza importante rispetto a `src/bcf.py`

`esegui_bcf_ctypes()` restituisce **direttamente le distanze finali**
(equivalenti a `sanifica_grafo()` + Dijkstra già eseguiti), non i soli
potenziali. Il flusso si accorcia:

- **Prima** (con `src/bcf.py`): predizioni → `costruisci_archi_ridotti` →
  `esporta_per_bcf` (file) → `esegui_bcf` (subprocess) → parsing output →
  `sanifica_grafo` → Dijkstra in Python
- **Dopo** (con `bcf_ctypes.py`): predizioni → `costruisci_archi_ridotti` →
  `archi_to_numpy` → `esegui_bcf_ctypes` → risultato finale

Non servono più `esporta_per_bcf`, il parsing dell'output testuale, né
`sanifica_grafo` seguito da un secondo Dijkstra — la libreria fa tutto
internamente in una sola chiamata.

## Stato di questo lavoro

Questo binding è stato scritto e verificato **contro il codice sorgente
reale** di `negative_weight_SSSP` (firme delle funzioni, tipi, struttura
del `CMakeLists.txt` controllati direttamente), ma **non è stato
compilato né eseguito** in questa sessione, perché il sorgente C++ vive
solo su Colab (clonato a runtime). Va quindi testato lì prima di
considerarlo pronto per la pipeline principale — verificare in particolare:

- Che la build proceda senza errori di linking
- Che `esegui_bcf_ctypes` produca le stesse distanze di `esegui_bcf`
  (confronto diretto su alcune coppie di test, come sanity check)
- Il reale speedup misurato, per confermare il guadagno atteso
