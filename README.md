# Learned Predictions for Shortest Path — Padova Road Network

Progetto finale — Advanced Topics in Algorithms
Nicole Mietto, Davide Sut

Applicazione dell'approccio "algorithms with predictions" (ispirato al lavoro originale di Bernstein, Nanongkai, Wulff-Nilsen — *Faster Fundamental Graph Algorithms via Learned Predictions*) al problema del Faster Shortest Path su una rete stradale reale (Padova). 
Nello specifico, al posto della formulazione originaria BNW, utilizziamo la velocizzazione empirica proposta da Cassis, Karrenbauer, Nusser e Rinaldi (2025) tramite il loro motore **BCF**, che si è dimostrata sensibilmente più veloce.

Nel nostro lavoro abbiamo esplorato le seguenti applicazioni/varianti:

1. **Dijkstra sul grafo sanato con diversi modelli ML**: applicazione dell'algoritmo addestrando varie architetture di predizione (modello standard, ad anelli, ecc.) per prevedere i potenziali duali necessari al motore BCF per sanare il grafo.
2. **Sub-Graph Routing**: limitazione della ricerca a un sottografo geometrico (rettangolo o ellisse) tra source e target, riducendo lo spazio di esplorazione a priori.
3. **Interpolazione spaziale**: calcolo delle predizioni ML solo su un campione di nodi "ancora" e interpolazione (Delaunay) per i restanti, tecnica usata congiuntamente al Sub-Graph Routing sull'ellisse.

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

## Cosa va dove sul Drive (struttura reale del progetto)

```
Advanced Topics in Algorithms/
├── notebook_principale.ipynb
├── grafi/
│   ├── padova_drive.graphml
│   └── veneto_drive.graphml
├── modelli_salvati/
│   ├── learned_potentials_model.joblib   (modello standard)
│   ├── modello_bcf_anelli.json           (il migliore finora)
│   ├── modello_bcf_6anelli.json
│   └── modello_bcf_veneto.json
├── traffico/                              (esperimenti per fascia oraria,
│   ├── padova_traffico_mattina.graphml     non ancora integrati nella
│   ├── padova_traffico_sera.graphml        pipeline principale)
│   ├── padova_traffico_notte.graphml
│   └── modello_traffico_padova.joblib
├── grafici e immagini/
│   └── *.png  (output delle celle di analisi/confronto)
└── archivio/
    └── cose vecchie/  (notebook superati, mantenuti per riferimento)
```

I path verso queste cartelle sono configurati nella cella "0. Setup ambiente"
di `notebook_principale.ipynb` (`GRAFI_DIR`, `MODELLI_DIR`, `TRAFFICO_DIR`,
`GRAFICI_DIR`). Se sposti o rinomini una cartella su Drive, aggiorna quella
cella di conseguenza.

## Cosa va dove in `dati/` (esecuzione locale, non Colab)

- **`dati/`** (livello principale): grafi stradali grezzi, scaricati una
  volta da OpenStreetMap — `padova_drive.graphml`, `veneto_drive.graphml`.
  Sono l'input di partenza, non si rigenerano col training.
- **`dati/modelli_salvati/`**: modelli ML già allenati — `modello_bcf_anelli.json`,
  `modello_bcf_6anelli.json`, `modello_bcf_veneto.json`,
  `learned_potentials_model.joblib`. Sono output del training (vedi `modelli/`),
  quindi rigenerabili rilanciando le funzioni corrispondenti, ma comodi da
  salvare per non doverlo rifare ogni volta.

## Attribuzioni e licenze di terze parti

Questo progetto usa, come dipendenza esterna clonata a runtime (non ridistribuita
nel repository), l'implementazione del seguente lavoro:

> Alejandro Cassis, Andreas Karrenbauer, André Nusser, and Paolo Luigi Rinaldi.
> **Algorithm Engineering of SSSP with Negative Edge Weights**.
> 23rd International Symposium on Experimental Algorithms (SEA 2025).
> Codice: https://github.com/PaoloLRinaldi/negative_weight_SSSP
> Licenza: [Creative Commons Attribution 4.0 International (CC-BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

`src/bcf.py` clona ed esegue questo codice come strumento esterno (analogamente
a una libreria), senza copiarne il codice sorgente in questo repository.

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

- **Precisione dei pesi**: si usano i decimi di secondo (`travel_time_d`). Mantenere i decimi permette di distinguere maggiormente le strade (soprattutto quelle più brevi) e introduce meno arrotondamenti anomali rispetto all'uso dei secondi interi. Dal punto di vista empirico, questo approccio si è rivelato preferibile per l'accuratezza e le performance del routing finale.
- **Segno dei potenziali**: `src/predizioni.py` inverte il segno della predizione del modello per ottenere potenziali iniziali compatibili con la convenzione richiesta dalla fase di sanazione del motore BCF. Anche se i costi ridotti iniziali possono includere pesi negativi, questa accortezza permette all'algoritmo di completare correttamente la sanazione. Il comportamento è verificato empiricamente in `valutazione/consistenza.py`.
- **MAE basso ≠ buona euristica**: un modello con MAE/R² ottimi può
  comunque produrre un'euristica inconsistente nei rami esplorati e
  scartati (anche con consistenza perfetta sul percorso ottimale finale).
  Vedi `valutazione/consistenza.verifica_consistenza_nodi_visitati`.
- **Loss custom**: `modelli/base.crea_loss_consistenza` penalizza
  esplicitamente le violazioni di consistenza durante il training
  XGBoost, migliorando sensibilmente il comportamento di Dijkstra sul grafo sanato rispetto a
  un modello allenato con MSE puro.
