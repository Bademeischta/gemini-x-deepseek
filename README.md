# Projekt "Ressourcen-Effiziente Dominanz" (RCN)

Dieses Projekt zielt darauf ab, eine neuartige Schach-KI zu entwickeln, die das Potenzial hat, State-of-the-Art-Engines zu übertreffen, indem sie auf extrem ressourcen-effizienten Methoden basiert (Training auf einer einzigen GPU).

## Kernkonzept

Das System basiert auf drei Säulen:
1.  **Daten (KKK):** Ein kompilierter, dichter Korpus aus existierenden, hochwertigen Analysedaten (`Kompilierter Kritischer Korpus`).
2.  **Architektur (RCN):** Ein Graph Attention Network (`Relational Chess Net`), das Schach als Beziehungsgeflecht modelliert.
3.  **Inferenz (IR-AB):** Eine CPU-basierte Alpha-Beta-Suche, die durch die GPU-gestützte neuronale Intelligenz geführt wird (`Information-Rich Alpha-Beta`).

---

## Roadmap & Status

-   [x] **Phase 0: Konzeption**
    -   [x] Entwicklung des Gesamtkonzepts (KKK, RCN, IR-AB)
-   [ ] **Phase 1: Prototyping**
    -   [x] **Arbeitspaket 1.1: Daten-Pipeline (Puzzles)**
        -   [x] Skript `process_puzzles.py` zur Erstellung des taktischen Datensatz-Subsets aus der Lichess Puzzle DB
    -   [x] **Arbeitspaket 1.2: Daten-Pipeline (Strategie)**
        -   [x] Skript `process_elite_games.py` zur Erstellung des strategischen Datensatz-Subsets aus der Lichess Elite DB
    -   [x] **Arbeitspaket 1.3: RCN-Modell Implementierung**
        -   [x] Definition der Graphen-Struktur in PyTorch Geometric
        -   [x] Implementierung des Multi-Task-Modells (Value, Policy, Tactic, Strategic Heads)
    -   [x] **Arbeitspaket 1.4: Training & Validierung**
        -   [x] Trainings-Loop für das RCN-Modell
        -   [ ] Validierung der Konvergenz auf dem KKK-Subset
-   [ ] **Phase 2: Inferenz-Implementierung**
    -   [x] **Arbeitspaket 2.1: UCI-Grundgerüst**
    -   [ ] **Arbeitspaket 2.2: Vollständige IR-AB-Implementierung**
-   [ ] **Phase 3: Integration & Test**
    -   [ ] Integration von RCN und IR-AB zu einer funktionalen Engine
    -   [ ] Testpartien gegen Benchmark-Engines

---

## Aktueller Stand (23. Oktober 2025)

Wir befinden uns nun im finalen Schritt von Phase 2: der Implementierung des IR-AB Suchalgorithmus in engine.py.

## Nutzung

### 1. Erstellung der Datensätze

#### Taktischer Datensatz

1.  Stellen Sie sicher, dass Python 3.8+ und die erforderlichen Bibliotheken (`requests`, `python-chess`) installiert sind.
2.  Führen Sie das Skript vom Hauptverzeichnis aus:
    ```bash
    python scripts/process_puzzles.py
    ```
3.  **Ergebnis:** Eine Datei `data/kkk_subset_puzzles.jsonl` wird erstellt.

#### Strategischer Datensatz

1.  Stellen Sie sicher, dass Python 3.8+ und die erforderlichen Bibliotheken (`requests`, `python-chess`, `zstandard`) installiert sind.
2.  Führen Sie das Skript vom Hauptverzeichnis aus:
    ```bash
    python scripts/process_elite_games.py
    ```
3.  **Ergebnis:** Eine Datei `data/kkk_subset_strategic.jsonl` wird erstellt.

### 2. Training des Modells

1.  Stellen Sie sicher, dass die Datensätze aus Schritt 1 generiert wurden und sich in `data/` befinden.
2.  Installieren Sie die zusätzlichen ML-Bibliotheken: `torch`, `torch-geometric`.
3.  Starten Sie das Training:
    ```bash
    python train.py
    ```
4.  Das Skript lädt die Datensätze, teilt sie in Trainings- und Validierungs-Sets auf und beginnt den Trainingsprozess. Der Fortschritt wird auf der Konsole ausgegeben.
5.  **Ergebnis:** Das beste Modell wird kontinuierlich in `models/rcn_model.pth` gespeichert, basierend auf dem niedrigsten Validierungsverlust.
