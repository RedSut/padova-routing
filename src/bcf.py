"""
src/bcf.py
==========
Compilazione ed esecuzione dell'implementazione BCF (Bringmann, Cassis,
Fischer 2023 / Cassis, Karrenbauer, Nusser, Rinaldi 2025) per il calcolo
del negative-weight single-source shortest path.

Repo originale: https://github.com/PaoloLRinaldi/negative_weight_SSSP
Licenza del codice originale: Creative Commons Attribution 4.0 International
(CC-BY 4.0) — https://creativecommons.org/licenses/by/4.0/

Questo modulo NON ridistribuisce il codice sorgente originale: lo clona e
lo compila a runtime tramite compila_bcf(), trattandolo come una dipendenza
esterna invocata da riga di comando (analogamente a una libreria di sistema).
"""

import os
import re
import shutil
import subprocess
import time


def compila_bcf(bcf_dir: str = "/content/negative_weight_SSSP") -> str:
    """
    Clona e compila l'eseguibile BCF se non già presente. Pensato per
    essere chiamato una sola volta per sessione (Colab o locale).

    Restituisce il percorso assoluto dell'eseguibile compilato (BCF_BIN).
    """
    bcf_bin = f"{bcf_dir}/build/Main"

    if not os.path.exists(bcf_bin):
        print("Clono il repo BCF...")
        subprocess.run(
            [
                "git", "clone", "--depth=1",
                "https://github.com/PaoloLRinaldi/negative_weight_SSSP.git",
                bcf_dir,
            ],
            check=True,
        )

        # Iniezione del timer C++ nel codice clonato prima di compilarlo
        queries_path = os.path.join(bcf_dir, "src", "queries.cpp")
        if os.path.exists(queries_path):
            with open(queries_path, "r") as f:
                content = f.read()
            if "high_resolution_clock" not in content:
                patch = (
                    "auto t_start = std::chrono::high_resolution_clock::now();\n"
                    "        results.push_back(std::visit(run_query, query_data));\n"
                    "        auto t_end = std::chrono::high_resolution_clock::now();\n"
                    "        double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();\n"
                    "        std::cout << \"time to compute potential mean = \" << elapsed_ms << \" ms\" << std::endl;"
                )
                content = content.replace("results.push_back(std::visit(run_query, query_data));", patch)
                # Aggiungiamo <chrono> in cima se manca
                if "<chrono>" not in content:
                    content = "#include <chrono>\n" + content
                with open(queries_path, "w") as f:
                    f.write(content)

        print("Compilo (build.sh usa cmake + make)...")
        subprocess.run(["bash", "build.sh"], cwd=bcf_dir, check=True)
        print(f"✅ Compilato: {bcf_bin}")
    else:
        print(f"✅ Già compilato: {bcf_bin}")

    return bcf_bin


def esporta_per_bcf(archi: list, super_idx: int, filename: str) -> None:
    """
    Scrive il file di input per l'eseguibile BCF.

    Formato (da graph.cpp del repo originale):
      - Prima riga: numero di nodi totali (= super_idx + 1, perché i nodi
        vanno da 0 a super_idx incluso)
      - Righe successive: "source target weight" per ogni arco
    """
    n_nodi_totali = super_idx + 1
    with open(filename, "w") as f:
        f.write(f"{n_nodi_totali}\n")
        f.writelines(archi)


def esegui_bcf(
    bcf_bin: str,
    input_filename: str,
    super_idx: int,
    n_nodi: int,
    algoritmo: str = "BCF",
    diam_apprx: int = 0,
    verbose: bool = False,
    quiet: bool = True,
) -> tuple[dict, float]:
    """
    Esegue l'implementazione BCF/GOR C++ e restituisce i potenziali.

    Parametri:
        bcf_bin         : percorso dell'eseguibile compilato (vedi compila_bcf)
        input_filename  : file generato da esporta_per_bcf
        super_idx       : indice del super-nodo (sorgente del SSSP)
        n_nodi          : numero di nodi REALI (senza il super-nodo) attesi
                          in output; usato per validare che il parsing abbia
                          funzionato
        algoritmo       : "BCF" (default) o "GOR" per il confronto con
                          l'algoritmo di Goldberg-Radzik
        verbose         : se True, stampa l'output grezzo del C++ (solo per
                          debug, può essere molto lungo su grafi grandi)

    Raises:
        AssertionError se il parsing produce meno di n_nodi potenziali —
        questo evita di propagare silenziosamente un grafo non sanato (i
        nodi mancanti finirebbero con phi=0 di default in sanifica_grafo,
        rendendoli potenzialmente irraggiungibili).

    NOTA sui path con spazi: il parser di queries.cpp legge il filename con
    ss >> filename, che si ferma al primo spazio. Per questo copiamo sempre
    il file di input in un percorso sicuro sotto /tmp prima di passarlo a BCF.
    """
    safe_input = "/tmp/bcf_graph.txt"
    query_file = "/tmp/bcf_query.txt"
    output_file = "/tmp/bcf_out.txt"

    shutil.copy(input_filename, safe_input)

    with open(query_file, "w") as f:
        f.write(f"SSSP check {algoritmo} {safe_input} {super_idx}\n")

    t0_python = time.time()
    try:
        result = subprocess.run(
            [bcf_bin, f"diam_apprx={diam_apprx}", "k_factor=1", "cutedges=1",
             query_file, output_file],
            capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Errore C++ ({algoritmo}):\n{e.stderr[:500]}")
        raise

    tempo_totale_python = time.time() - t0_python
    output_cpp = result.stdout
    if verbose:
        print("OUTPUT C++ GREZZO:\n", output_cpp)

    match = re.search(
        r"(?i)time to compute potential.*?mean =\s*([0-9]*\.[0-9]+)\s*ms", output_cpp
    )
    if match:
        t_puro = float(match.group(1)) / 1000.0  # ms -> s
    else:
        match_fallback = re.search(
            r"(?i)(?:time|elapsed|took|duration|inner_loop_all).*?([0-9]*\.[0-9]+)",
            output_cpp,
        )
        t_puro = float(match_fallback.group(1)) if match_fallback else tempo_totale_python

    phi = {}
    with open(output_file) as f:
        for idx, val in enumerate(f.read().strip().split()):
            if val.lower() not in ["inf", "-inf", "infinity", "-infinity"]:
                try:
                    phi[idx] = int(float(val))
                except ValueError:
                    continue

    assert len(phi) >= n_nodi, (
        f"Parsing {algoritmo} fallito: attesi >= {n_nodi} potenziali, "
        f"trovati {len(phi)}.\nEsegui con verbose=True per vedere l'output "
        f"grezzo del C++."
    )

    if not quiet:
        print(
            f"  {algoritmo} completato: {len(phi)} potenziali | "
            f"C++ Puro = {t_puro:.4f}s (I/O Python = {tempo_totale_python:.4f}s)"
        )
    return phi, t_puro
