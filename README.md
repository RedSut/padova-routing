# Learned Predictions for Shortest Path — Padova Road Network

Progetto finale — Advanced Topics in Algorithms
Nicole Mietto, Davide Sut

Applicazione dell'approccio "algorithms with predictions" (Bernstein,
Nanongkai, Wulff-Nilsen — *Faster Fundamental Graph Algorithms via
Learned Predictions*) al problema del Faster Shortest Path su una rete
stradale reale (Padova), con tre varianti esplorate:

1. **A\* con potenziali predetti**: un modello ML predice i potenziali
   duali, usati per "sanare" i pesi negativi del grafo (via
   [BCF — Cassis, Karrenbauer, Nusser, Rinaldi 2025](https://github.com/PaoloLRinaldi/negative_weight_SSSP)),
   rendendo Dijkstra sul grafo sanato equivalente ad A* sul grafo
   originale.
2. **Sub-Graph Routing**: le predizioni sono limitate a un sottografo
   (rettangolo o ellisse) tra source e target.
3. **Interpolazione spaziale**: le predizioni sono calcolate solo su un
   campione di nodi "ancora", e interpolate (Delaunay) per i restanti.

## Struttura del repository

```
padova-routing/
├── src/                  # Logica di sistema (grafo, BCF, algoritmi, predizioni)
├── modelli/               # Strategie di training (standard, anelli, regionale)
├── valutazione/           # Consistenza dell'euristica, benchmark, confronti
├── dati/                  # File .graphml e modelli .json/.joblib (NON in Git)
├── output/                # Grafici e file generati a runtime (NON in Git)
├── notebook_principale.ipynb
└── requirements.txt
```

## Cosa va dove in `dati/`

- **`dati/`** (livello principale): grafi stradali grezzi, scaricati una
  volta da OpenStreetMap — `padova_drive.graphml`, `veneto_drive.graphml`.
  Sono l'input di partenza, non si rigenerano col training.
- **`dati/modelli_salvati/`**: modelli ML già allenati — `modello_bcf_anelli.json`,
  `modello_bcf_6anelli.json`, `modello_bcf_veneto.json`,
  `learned_potentials_model.joblib`. Sono output del training (vedi `modelli/`),
  quindi rigenerabili rilanciando le funzioni corrispondenti, ma comodi da
  salvare per non doverlo rifare ogni volta.

## Setup

### Su Google Colab (consigliato per la prima esecuzione)

1. Apri `notebook_principale.ipynb` su Colab.
2. La prima cella clona automaticamente questo repository e installa le
   dipendenze.
3. Monta il tuo Google Drive e imposta `DATI_DIR` nella cella di
   configurazione, puntando alla cartella che contiene
   `padova_drive.graphml` (e opzionalmente `veneto_drive.graphml`,
   modelli già allenati, ecc.).

### In locale / repository

```bash
git clone <url-di-questo-repo>
cd padova-routing
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebook_principale.ipynb
```

I file `.graphml` (grafo OSMnx) e i modelli allenati (`.json`, `.joblib`)
non sono incluso nel repository per dimensione — vanno scaricati separatamente
e posizionati in `dati/`, oppure referenziati da Google Drive se si lavora
su Colab.

## Note metodologiche principali

- **Precisione dei pesi**: si usano i decimi di secondo (`travel_time_d`)
  come compromesso tra precisione e velocità del motore BCF — vedi
  `valutazione/benchmark_precisione.py` per la misura sperimentale che
  motiva questa scelta (i secondi interi sono risultati i più LENTI, non
  i più rapidi, contro l'aspettativa teorica naive).
- **Segno dei potenziali**: `src/predizioni.py` inverte il segno della
  predizione del modello per ottenere potenziali consistenti con la
  sanazione Bellman-Ford-Moore. Verificato empiricamente in
  `valutazione/consistenza.py`.
- **MAE basso ≠ buona euristica**: un modello con MAE/R² ottimi può
  comunque produrre un'euristica inconsistente nei rami che A* esplora e
  scarta (anche con consistenza perfetta sul percorso ottimale finale).
  Vedi `valutazione/consistenza.verifica_consistenza_nodi_visitati`.
- **Loss custom**: `modelli/base.crea_loss_consistenza` penalizza
  esplicitamente le violazioni di consistenza durante il training
  XGBoost, migliorando sensibilmente il comportamento di A* rispetto a
  un modello allenato con MSE puro.
