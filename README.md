# Projekt "Ressourcen-Effiziente Dominanz" (RCN)

Willkommen beim RCN-Projekt! Dieses Projekt zielt darauf ab, eine hochmoderne Schach-KI zu entwickeln, die auf einer einzigartigen Kombination aus Graphen-neuronalen Netzen und traditionellen Suchalgorithmen basiert. Unser Ziel ist es, eine Engine zu schaffen, die nicht nur stark spielt, sondern auch die zugrundeliegenden strategischen und taktischen Muster einer Schachstellung tiefgreifend versteht.

## Inhaltsverzeichnis

- [Philosophie und Kernkonzepte](#philosophie-und-kernkonzepte)
- [Wie die KI funktioniert: Ein tiefer Einblick](#wie-die-ki-funktioniert-ein-tiefer-einblick)
  - [1. Die Daten-Pipeline](#1-die-daten-pipeline)
  - [2. Das Herzstück: Das Relational Chess Net (RCN)](#2-das-herzstück-das-relational-chess-net-rcn)
  - [3. Die Suche: Information-Rich Alpha-Beta (IR-AB)](#3-die-suche-information-rich-alpha-beta-ir-ab)
- [Projektstruktur](#projektstruktur)
- [Einrichtung und Nutzung](#einrichtung-und-nutzung)
  - [Lokale Nutzung](#lokale-nutzung)
  - [Google Colab Nutzung](#google-colab-nutzung)
- [Wichtige Skripte und ihre Verwendung](#wichtige-skripte-und-ihre-verwendung)
  - [`benchmark_hardware.py`](#benchmark_hardwarepy)
  - [`train.py`](#trainpy)
  - [`engine.py`](#enginepy)
  - [Datenverarbeitungs-Skripte](#datenverarbeitungs-skripte)
- [Troubleshooting](#troubleshooting)
- [Projekt-Status](#projekt-status)

---

## Philosophie und Kernkonzepte

Die zentrale Hypothese dieses Projekts ist, dass Schach weniger ein Brettspiel als vielmehr ein dynamisches Netzwerk von Beziehungen zwischen Figuren ist. Anstatt das Brett als ein 8x8-Gitter zu betrachten, modellieren wir es als einen Graphen, in dem Figuren die Knoten und ihre Interaktionen (Angriff, Verteidigung, Fesselung) die Kanten sind.

Das System basiert auf drei Säulen:
1.  **Daten (KKK):** Ein kompilierter, dichter Korpus aus existierenden, hochwertigen Analysedaten (`Kompilierter Kritischer Korpus`). Wir nutzen Puzzle-Datenbanken und strategische Meisterpartien, um dem Modell sowohl taktische Schärfe als auch positionelles Verständnis beizubringen.
2.  **Architektur (RCN):** Ein Graph Attention Network (`Relational Chess Net`), das diese Beziehungen modelliert. Es lernt, die Bedeutung von Figuren in Abhängigkeit von ihrem Kontext zu bewerten – genau wie ein menschlicher Großmeister.
3.  **Inferenz (IR-AB):** Eine CPU-basierte Alpha-Beta-Suche, die durch die GPU-gestützte neuronale Intelligenz geführt wird (`Information-Rich Alpha-Beta`). Die KI sagt nicht nur den "besten" Zug voraus, sondern liefert eine reichhaltige Heuristik (Wertung, Zug-Wahrscheinlichkeiten, taktische Flags), die eine intelligentere und effizientere Suche ermöglicht.

---

## Wie die KI funktioniert: Ein tiefer Einblick

### 1. Die Daten-Pipeline
Alles beginnt mit den Daten. Wir verwenden `.jsonl`-Dateien, die Schachstellungen (im FEN-Format) zusammen mit Metadaten enthalten:
-   `value`: Eine quantitative Bewertung der Stellung (z.B. von Stockfish).
-   `policy_target`: Der beste Zug in der Stellung (im UCI-Format).
-   `tactic_flag`: Ein boolescher Wert, der anzeigt, ob die Stellung eine taktische Sequenz enthält.
-   `strategic_flag`: Ein boolescher Wert, der strategische Motive hervorhebt.

Diese Daten werden mit Skripten wie `process_puzzles.py` und `process_elite_games.py` aus Rohdaten (z.B. PGN-Dateien) generiert.

### 2. Das Herzstück: Das Relational Chess Net (RCN)
Das RCN ist ein Graphen-neuronales Netz (`GNN`) mit `GATv2Conv`-Schichten (Graph Attention Network).

-   **Graphen-Erstellung (`fen_to_graph_data`):** Für jede FEN-Stellung wird ein Graph erstellt:
    -   **Knoten:** Jede Figur auf dem Brett ist ein Knoten. Die Knoten-Features enthalten Informationen wie Figurentyp, Farbe, Position und globale Zustandsinformationen (Rochaderechte, En-Passant-Feld, 50-Züge-Zähler).
    -   **Kanten:** Kanten repräsentieren die Beziehungen zwischen den Figuren. Wir verwenden verschiedene Kantentypen: `ATTACKS`, `DEFENDS`, `PIN` (Fesselung) und `XRAY`.
-   **Lernprozess:** Das Modell lernt, die Aufmerksamkeit auf die wichtigsten Beziehungen in einer Stellung zu lenken. Es hat mehrere "Köpfe" (Output-Layer), die Folgendes vorhersagen:
    -   **Value Head:** Die allgemeine Bewertung der Stellung.
    -   **Policy Heads (From, To, Promotion):** Die Wahrscheinlichkeit für jeden möglichen Zug, aufgeteilt in Startfeld, Zielfeld und Umwandlungsfigur.
    -   **Tactic/Strategic Flag Heads:** Ob es sich um eine taktische oder strategische Stellung handelt.

### 3. Die Suche: Information-Rich Alpha-Beta (IR-AB)
Die `engine.py` implementiert eine klassische Negamax-Suche mit Alpha-Beta-Pruning. Der Clou ist, wie die Vorhersagen des RCN-Modells die Suche steuern:
1.  **Zug-Sortierung (Move Ordering):** Anstatt Züge blind zu durchsuchen, werden sie intelligent sortiert. Eine gute Sortierung ist entscheidend für effizientes Pruning. Die Priorität ist:
    1.  PV-Zug (aus der Transpositionstabelle)
    2.  Gute Schlagzüge (bewertet nach MVV-LVA)
    3.  Killer-Züge
    4.  Vom Policy-Netzwerk vorhergesagte Züge.
2.  **Quiescence Search:** Um den "Horizon-Effekt" zu vermeiden, wird am Ende der Hauptsuche eine spezielle, flachere Suche nach Schlagzügen durchgeführt, um die Stellung zu stabilisieren.
3.  **Transposition Table:** Bereits analysierte Stellungen werden in einer Hashtabelle (Transposition Table) gespeichert, um redundante Berechnungen zu vermeiden.

---

## Projektstruktur

```
├── data/                  # Trainingsdaten (z.B. puzzles.jsonl)
├── models/                # Trainierte Modelle (rcn_model.pth) und Checkpoints
├── scripts/               # Hilfsskripte
│   ├── dataset.py         # PyTorch Dataset-Klassen
│   ├── model.py           # RCN-Modellarchitektur
│   ├── graph_utils.py     # Logik zur Umwandlung von FEN in Graphen
│   └── ...                # Weitere Skripte zur Datenverarbeitung
├── tests/                 # Unit- und Integrationstests
├── benchmark_hardware.py  # Skript zur Hardware-Diagnose
├── config.py              # Zentrale Konfigurationsdatei
├── engine.py              # UCI-kompatible Schach-Engine
├── train.py               # Skript zum Trainieren des Modells
└── requirements.txt       # Projektabhängigkeiten
```

---

## Einrichtung und Nutzung

Dieses Projekt kann sowohl lokal als auch in Google Colab ausgeführt werden.

### Lokale Nutzung

**1. Voraussetzungen:**
- Python 3.8+
- Git
- Eine NVIDIA-GPU mit CUDA-Unterstützung wird für ernsthaftes Training dringend empfohlen.

**2. Installation:**
```bash
git clone https://github.com/Bademeischta/gemini-x-deepseek
cd gemini-x-deepseek
pip install -r requirements.txt
```

**3. Hardware-Benchmark (Empfohlen):**
Führen Sie vor dem ersten Training den Hardware-Benchmark aus. Dieses Skript testet Ihre Systemleistung und gibt wichtige Empfehlungen zur Optimierung Ihrer Konfiguration.
```bash
python benchmark_hardware.py
```
Analysieren Sie die Ausgabe und passen Sie ggf. die `config.py` (z.B. `BATCH_SIZE`) an.

**4. Training starten:**
Stellen Sie sicher, dass Ihre `.jsonl`-Datendateien im `data/`-Verzeichnis liegen. Starten Sie dann das Training:
```bash
python train.py
```
Das Training kann fortgesetzt werden, falls ein Checkpoint in `models/` gefunden wird. Fortschrittsbalken informieren Sie über den Status.

**5. Engine verwenden:**
Die Engine kommuniziert über das Standard-UCI-Protokoll. Sie können sie in jeder UCI-kompatiblen Schach-GUI (z.B. Arena, Cute Chess, Scid vs. PC) einbinden. Geben Sie als Engine-Pfad den folgenden Befehl an:
```bash
python /pfad/zu/ihrem/projekt/engine.py
```

### Google Colab Nutzung

Colab ist eine hervorragende Option für das Training, da es kostenlosen GPU-Zugang bietet. Das Projekt ist für die persistente Speicherung in Google Drive ausgelegt.

**1. Notebook einrichten:**
- Öffnen Sie ein neues Colab-Notebook.
- Stellen Sie sicher, dass Sie eine GPU-Laufzeit verwenden: `Laufzeit` → `Laufzeittyp ändern` → `T4 GPU`.

**2. Google Drive verbinden:**
Führen Sie diesen Befehl in einer Zelle aus, um Ihr Google Drive zu mounten.
```python
from google.colab import drive
drive.mount('/content/drive')
```

**3. Projekt klonen und installieren:**
```python
%cd /content/drive/MyDrive/
!git clone https://github.com/Bademeischta/gemini-x-deepseek
%cd gemini-x-deepseek
!pip install -r requirements.txt
```

**4. Daten-Upload:**
Laden Sie Ihre `.jsonl`-Datensätze in den Ordner `/content/drive/MyDrive/gemini-x-deepseek/data/` hoch.

**5. Hardware-Benchmark & Training:**
Führen Sie die Skripte wie bei der lokalen Nutzung aus, aber mit einem `!` vor dem Befehl:
```python
!python benchmark_hardware.py
!python train.py
```
Trainierte Modelle und Checkpoints werden persistent in Ihrem Google Drive im `models/`-Ordner gespeichert.

---

## Wichtige Skripte und ihre Verwendung

### `benchmark_hardware.py`
Ein Diagnose-Tool, das **vor dem Training** ausgeführt werden sollte. Es prüft:
-   Ob eine GPU erkannt wird und korrekt funktioniert.
-   Die Geschwindigkeit des Datentransfers zur GPU.
-   Die Forward-Pass-Geschwindigkeit des Modells bei verschiedenen Batch-Größen.
-   Gibt konkrete Empfehlungen zur Optimierung von `BATCH_SIZE`, `num_workers`, etc.

### `train.py`
Das Hauptskript für das Training des RCN-Modells.
-   **Fortschrittsanzeigen:** Verwendet `tqdm`, um den Fortschritt auf Epochen- und Batch-Ebene anzuzeigen.
-   **GPU-Optimierung:** Nutzt automatisch Mixed Precision (`torch.amp`) und optimierte `DataLoader`-Einstellungen, wenn eine GPU verfügbar ist, um das Training erheblich zu beschleunigen.
-   **Checkpointing:** Speichert nach jeder Epoche einen Checkpoint und das beste Modell. Das Training kann jederzeit unterbrochen und fortgesetzt werden.
-   **Deadlock-sicher:** Die Daten-Pipeline ist so konzipiert, dass sie auch bei paralleler Datenverarbeitung nicht blockiert.

### `engine.py`
Die UCI-Engine. Dieses Skript wird von Schach-GUIs aufgerufen. Es lädt das trainierte Modell aus `models/rcn_model.pth` und startet die Suchlogik.

### Datenverarbeitungs-Skripte
-   `scripts/process_puzzles.py`: Verarbeitet eine `.csv`-Datei mit Schachpuzzles, um ein `.jsonl`-Trainingsset zu erstellen.
-   `scripts/process_elite_games.py`: Analysiert PGN-Dateien von hochrangigen Partien, um strategische Trainingsdaten zu generieren.

---

## Troubleshooting

-   **Problem: "CUDA out of memory"**
    -   **Lösung:** Reduzieren Sie die `BATCH_SIZE` in `config.py`. Führen Sie `benchmark_hardware.py` erneut aus, um eine empfohlene Größe zu finden.
-   **Problem: GPU-Auslastung in `nvidia-smi` ist < 30%**
    -   **Ursache:** Die CPU kann die Daten nicht schnell genug vorbereiten (CPU-Bottleneck).
    -   **Lösung:** Erhöhen Sie `num_workers` in `train.py` (nur wenn auf GPU trainiert wird), stellen Sie sicher, dass die Daten auf einer schnellen Festplatte (SSD) liegen.
-   **Problem: Training startet nicht, `FileNotFoundError`**
    -   **Lösung:** Stellen Sie sicher, dass das `data/`-Verzeichnis existiert. Das Trainingsskript kann Dummy-Dateien erstellen, aber nicht das Verzeichnis selbst. Erstellen Sie es manuell: `mkdir data`.

---

## Projekt-Status

Dieses Projekt befindet sich in aktiver Entwicklung. Eine detaillierte Liste der abgeschlossenen und geplanten Aufgaben finden Sie im oberen Teil dieser `README.md` und in der `CHANGELOG.md`.
