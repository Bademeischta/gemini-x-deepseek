# Projekt "Ressourcen-Effiziente Dominanz" (RCN) v2

Willkommen beim RCN-Projekt! Dieses Projekt zielt darauf ab, eine hochmoderne Schach-KI zu entwickeln, die auf einer einzigartigen Kombination aus Graphen-neuronalen Netzen (GNN) und Monte Carlo Tree Search (MCTS) basiert. Unser Ziel ist es, eine Engine zu schaffen, die nicht nur stark spielt, sondern auch die zugrundeliegenden strategischen und taktischen Muster einer Schachstellung tiefgreifend versteht.

## Inhaltsverzeichnis

- [Philosophie und Kernkonzepte](#philosophie-und-kernkonzepte)
- [Wie die KI funktioniert: Ein tiefer Einblick](#wie-die-ki-funktioniert-ein-tiefer-einblick)
  - [1. Die Daten-Pipeline](#1-die-daten-pipeline)
  - [2. Das Herzstück: Das Relational Chess Net (RCN v2)](#2-das-herzstück-das-relational-chess-net-rcn-v2)
  - [3. Die Suche: Batch-MCTS](#3-die-suche-batch-mcts)
- [Projektstruktur](#projektstruktur)
- [Einrichtung und Nutzung](#einrichtung-und-nutzung)
  - [Lokale Nutzung](#lokale-nutzung)
  - [Google Colab Nutzung](#google-colab-nutzung)
- [Wichtige Skripte und ihre Verwendung](#wichtige-skripte-und-ihre-verwendung)
- [Troubleshooting](#troubleshooting)
- [Projekt-Status](#projekt-status)

---

## Philosophie und Kernkonzepte

Die zentrale Hypothese dieses Projekts ist, dass Schach weniger ein Brettspiel als vielmehr ein dynamisches Netzwerk von Beziehungen zwischen Figuren ist. Anstatt das Brett als ein 8x8-Gitter zu betrachten, modellieren wir es als einen Graphen, in dem Figuren die Knoten und ihre Interaktionen (Angriff, Verteidigung) die Kanten sind.

Das System basiert auf drei Säulen:
1.  **Daten (KKK):** Ein kompilierter, dichter Korpus aus existierenden, hochwertigen Analysedaten. Wir nutzen Puzzle-Datenbanken und strategische Meisterpartien.
2.  **Architektur (RCN v2):** Ein Graph Attention Network (`GATv2`), das diese Beziehungen modelliert. Version 2 nutzt optimierte Features und eine vereinfachte Kantenstruktur für höhere Effizienz.
3.  **Inferenz (MCTS):** Eine GPU-beschleunigte Monte Carlo Tree Search (`BatchMCTS`). Anstatt klassischer Alpha-Beta-Suche nutzt die Engine die Wahrscheinlichkeiten des neuronalen Netzes, um den Suchbaum selektiv zu expandieren. Blattknoten werden gesammelt und in Batches auf der GPU evaluiert.

---

## Wie die KI funktioniert: Ein tiefer Einblick

### 1. Die Daten-Pipeline
Wir verwenden `.jsonl`-Dateien, die Schachstellungen (im FEN-Format) zusammen mit Metadaten enthalten:
-   `value`: Eine quantitative Bewertung der Stellung.
-   `policy_target`: Der beste Zug in der Stellung (im UCI-Format).
-   `tactic_flag` / `strategic_flag`: Indikatoren für taktische oder strategische Motive.

Das Skript `scripts/dataset.py` lädt diese Daten effizient und wandelt sie on-the-fly in Graphen um. Dabei werden `legal_moves_mask` (Maske für legale Züge) für das Training vorberechnet, um die Policy-Vorhersagen valide zu halten.

### 2. Das Herzstück: Das Relational Chess Net (RCN v2)
Das RCN v2 ist ein Graphen-neuronales Netz, definiert in `scripts/model.py`. Es wurde gegenüber v1 stark optimiert:

-   **Graphen-Erstellung (`fen_to_graph_data_v2.py`):**
    -   **Knoten (15 Features):** Jede Figur ist ein Knoten. Features sind: 12x One-Hot für Figur/Farbe, 1x File (normiert), 1x Rank (normiert), 1x Mobilität. Keine Embeddings mehr nötig!
    -   **Kanten (2 Features):** Gerichtete Kanten repräsentieren Angriffe und Verteidigungen. `Edge Attr`: [1, 0] für Angriff (Attack), [0, 1] für Verteidigung (Defend).
-   **Modell-Köpfe:**
    -   **Value Head:** Bewertung der Stellung (Scalar).
    -   **Joint Policy Head:** Ein Output-Vektor der Größe 4096, der alle möglichen Züge (From-Square * 64 + To-Square) abdeckt.
    -   **Auxiliary Heads:** Tactic/Strategic Flags zur Unterstützung des Lernprozesses.

### 3. Die Suche: Batch-MCTS
Die `engine.py` nutzt nun einen **Batch Monte Carlo Tree Search** (`scripts/mcts.py`).
-   **Selektion:** Der Baum wird basierend auf der PUCT-Formel (Predictor + Upper Confidence Bound applied to Trees) traversiert, die die Policy-Vorhersagen des RCN ("Priors") mit den Besuchszahlen ("Exploration") abwägt.
-   **Batch-Evaluierung:** Um die GPU effizient zu nutzen, werden Blattknoten gesammelt, bis ein Batch voll ist (z.B. 16 oder 32 Stellungen). Dieser Batch wird parallel durch das RCN geschleust.
-   **Backpropagation:** Die Ergebnisse (Value und Policy) werden im Baum zurückpropagiert.

---

## Projektstruktur

```
├── data/                      # Trainingsdaten
├── models/                    # Trainierte Modelle (rcn_model.pth)
├── scripts/                   # Kern-Logik und Hilfsskripte
│   ├── dataset.py             # Robustes Dataset mit Pre-Computing
│   ├── fen_to_graph_data_v2.py # V2 Graph Builder (SSOT für Konstanten)
│   ├── mcts.py                # Batch-MCTS Implementierung
│   ├── model.py               # RCN v2 Modellarchitektur
│   ├── uci_index.py           # Move-Index Konvertierung (4096)
│   ├── graph_utils.py         # Deprecated (Legacy Shim)
│   └── ...
├── tests/                     # Unit-Tests und Benchmarks
├── config.py                  # Zentrale Konfiguration
├── engine.py                  # UCI-Engine (nutzt MCTS)
├── train.py                   # Trainings-Loop (Mixed Precision)
└── requirements.txt           # Abhängigkeiten
```

---

## Einrichtung und Nutzung

Dieses Projekt kann lokal oder in Google Colab ausgeführt werden.

### Lokale Nutzung

**1. Installation:**
```bash
git clone <repo-url>
cd gemini-x-deepseek
pip install -r requirements.txt
```

**2. Hardware-Test:**
```bash
python benchmark_hardware.py
```

**3. Training:**
```bash
python train.py
```
Das Training nutzt automatisch Mixed Precision (`AMP`) und speichert Checkpoints.

**4. Engine:**
Die Engine ist UCI-kompatibel. Starten Sie sie in Ihrer GUI:
```bash
python engine.py
```
*Hinweis: Die Engine benötigt ein trainiertes Modell in `models/rcn_model.pth` oder `config.MODEL_SAVE_PATH`. Wenn keines gefunden wird, wird eine Warnung ausgegeben und ein zufälliges Modell verwendet (nur zum Testen).*

---

## Wichtige Skripte und ihre Verwendung

### `scripts/fen_to_graph_data_v2.py`
Der Kern-Algorithmus zur Umwandlung von `chess.Board` Objekten in PyTorch Geometric Graphen. Definiert auch die globalen Konstanten `NODE_FEATURES` und `EDGE_FEATURES`.

### `scripts/mcts.py`
Die Such-Engine. Implementiert `BatchMCTS`. Kann konfiguriert werden, um die Anzahl der Simulationen pro Zug zu steuern.

### `train.py`
Der Trainings-Loop.
-   **Features:** Mixed Precision Training, Gradient Clipping, Progress Bars (tqdm), Validation Split.
-   **Logik:** Nutzt pre-computed Masks aus dem Dataset, um illegale Züge im Policy-Loss zu ignorieren.

### `engine.py`
Der Einstiegspunkt für GUIs. Initialisiert das Modell (mit Fallback-Mechanismus) und den `BatchMCTS` Searcher.

---

## Troubleshooting

-   **Problem: `AssertionError` in `model.py`**
    -   **Ursache:** Veraltete `graph_utils.py` oder falsche Feature-Dimensionen.
    -   **Lösung:** Stellen Sie sicher, dass Sie `fen_to_graph_data_v2` verwenden und das Modell korrekt importiert wird.
-   **Problem: Engine stürzt ab "No model found"**
    -   **Lösung:** Die Engine läuft jetzt weiter (mit Warnung), spielt aber zufällige Züge. Trainieren Sie ein Modell mit `train.py`, um echte Spielstärke zu erhalten.
-   **Problem: Benchmark Crash**
    -   **Lösung:** Stellen Sie sicher, dass Sie `tests/benchmark_graph_builder.py` ausführen, welches jetzt korrekt `chess.Board` Objekte verwendet.

---

## Projekt-Status

**Aktuell:** RCN v2 Implementierung (Konsistente Architektur, MCTS Integration, Robustes Dataset).
**Nächste Schritte:** Erweiterung der Trainingsdaten, Hyperparameter-Tuning des MCTS.
