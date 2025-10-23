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
    -   [ ] **Arbeitspaket 1.1: Daten-Pipeline (Puzzles)**
        -   [x] Skript `process_puzzles.py` zur Erstellung des taktischen Datensatz-Subsets aus der Lichess Puzzle DB
    -   [ ] **Arbeitspaket 1.2: Daten-Pipeline (Strategie)**
        -   [ ] Skript `process_elite_games.py` zur Erstellung des strategischen Datensatz-Subsets aus der Lichess Elite DB
    -   [ ] **Arbeitspaket 1.3: RCN-Modell Implementierung**
        -   [ ] Definition der Graphen-Struktur in PyTorch Geometric
        -   [ ] Implementierung des Multi-Task-Modells (Value, Policy, Tactic, Strategic Heads)
    -   [ ] **Arbeitspaket 1.4: Training & Validierung**
        -   [ ] Trainings-Loop für das RCN-Modell
        -   [ ] Validierung der Konvergenz auf dem KKK-Subset
-   [ ] **Phase 2: Inferenz-Implementierung**
    -   [ ] Implementierung des IR-AB Suchalgorithmus
-   [ ] **Phase 3: Integration & Test**
    -   [ ] Integration von RCN und IR-AB zu einer funktionalen Engine
    -   [ ] Testpartien gegen Benchmark-Engines

---

## Aktueller Stand (23. Oktober 2025)

Die Konzeption ist abgeschlossen. Wir befinden uns in **Phase 1 (Prototyping)**.

Das erste Arbeitspaket (`1.1`) wurde fertiggestellt. Das Skript `scripts/process_puzzles.py` kann nun den ersten Teil unseres KKK-Datensatzes, den taktischen Teil, erstellen.

## Nutzung

### Erstellung des taktischen Datensatzes

1.  Stellen Sie sicher, dass Python 3.8+ und die erforderlichen Bibliotheken (`requests`, `python-chess`) installiert sind.
2.  Führen Sie das Skript vom Hauptverzeichnis aus:
    ```bash
    python scripts/process_puzzles.py
    ```
3.  Das Skript wird die Lichess Puzzle-Datenbank (ca. 3 GB) automatisch herunterladen und in das `data/`-Verzeichnis entpacken. Dieser Schritt kann einige Zeit dauern.
4.  Anschließend werden die Daten verarbeitet. Der Fortschritt wird auf der Konsole ausgegeben.
5.  **Ergebnis:** Eine Datei `data/kkk_subset_puzzles.jsonl` wird erstellt, die bis zu 1 Million validierte, taktische Schachpositionen im JSON-Lines-Format enthält. Eine `error_log.txt` wird für alle fehlerhaften Zeilen erstellt.
