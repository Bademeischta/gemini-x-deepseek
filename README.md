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
-   [x] **Phase 1: Prototyping**
    -   [x] Arbeitspaket 1.1: Daten-Pipeline (Puzzles)
    -   [x] Arbeitspaket 1.2: Daten-Pipeline (Strategie)
    -   [x] Arbeitspaket 1.3: RCN-Modell Implementierung
    -   [x] Arbeitspaket 1.4: Training & Validierung
-   [x] **Phase 2: Inferenz-Implementierung**
    -   [x] Arbeitspaket 2.1: UCI-Grundgerüst
    -   [x] Arbeitspaket 2.2: Vollständige IR-AB-Implementierung
-   [ ] **Phase 3: Integration & Test**
    -   [x] **Arbeitspaket 3.1: Finale Integration & Packaging**
    -   [ ] Testpartien gegen Benchmark-Engines

---

## Installation

1.  Klonen Sie das Repository auf Ihren lokalen Rechner.
2.  Installieren Sie alle erforderlichen Abhängigkeiten in einer Python-Umgebung (Version 3.8+ empfohlen) mit dem folgenden Befehl:
    ```bash
    pip install -r requirements.txt
    ```

## Nutzung

Das gesamte Projekt wird über den zentralen Einstiegspunkt `main.py` gesteuert. Alle Befehle müssen vom Hauptverzeichnis des Projekts aus ausgeführt werden.

### 1. Datenverarbeitung

Um die taktischen und strategischen Datensätze (`kkk_subset_puzzles.jsonl` und `kkk_subset_strategic.jsonl`) zu erstellen, die für das Training benötigt werden, führen Sie aus:
```bash
python main.py process-data
```

### 2. Modell-Training

Um das RCN-Modell auf den zuvor erstellten Datensätzen zu trainieren, verwenden Sie:
```bash
python main.py train
```
Das Skript speichert das Modell mit der besten Validierungsleistung in `models/rcn_model.pth`.

### 3. Engine ausführen

Um die Schach-Engine über die UCI-Schnittstelle zu starten (z.B. zur Verwendung mit einer GUI wie Cute Chess), führen Sie aus:
```bash
python main.py run-engine
```

### (Optional) Dummy-Modell erstellen

Für Entwicklungs- und Testzwecke, falls kein trainiertes Modell verfügbar ist, können Sie ein Dummy-Modell mit zufälligen Gewichten erstellen:
```bash
python main.py create-dummy-model
```
