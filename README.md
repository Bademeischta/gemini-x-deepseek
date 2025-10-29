# Projekt "Ressourcen-Effiziente Dominanz" (RCN) - Abgeschlossen

Dieses Projekt zielte darauf ab, eine neuartige Schach-KI zu entwickeln, die das Potenzial hat, State-of-the-Art-Engines zu übertreffen, indem sie auf extrem ressourcen-effizienten Methoden basiert. Das Projekt ist nun von der Konzeption bis zur Validierung vollständig abgeschlossen.

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
-   [x] **Phase 3: Integration & Test**
    -   [x] Arbeitspaket 3.1: Finale Integration & Packaging
    -   [x] Arbeitspaket 3.2: Implementierung des Test-Frameworks

---

## Projektstruktur

Der gesamte Arbeitsablauf des Projekts ist in vier Hauptschritte unterteilt, die nacheinander ausgeführt werden sollten.

### Schritt 1: Installation

1.  Klonen Sie das Repository auf Ihren lokalen Rechner.
2.  Installieren Sie alle erforderlichen Python-Abhängigkeiten in einer virtuellen Umgebung (Python 3.8+ empfohlen):
    ```bash
    pip install -r requirements.txt
    ```

### Schritt 2: Datenverarbeitung

Erstellen Sie die taktischen und strategischen Datensätze (`kkk_subset_puzzles.jsonl` und `kkk_subset_strategic.jsonl`), die für das Training benötigt werden:
```bash
python main.py process-data
```

### Schritt 3: Modell-Training

Trainieren Sie das RCN-Modell auf den zuvor erstellten Datensätzen:
```bash
python main.py train
```
Das Skript speichert das Modell mit der besten Validierungsleistung in `models/rcn_model.pth`.

*(Optional) Für Entwicklungszwecke können Sie ein Dummy-Modell mit zufälligen Gewichten erstellen, falls kein trainiertes Modell verfügbar ist:*
```bash
python main.py create-dummy-model
```

### Schritt 4: Validierung & Test

Um die Spielstärke der trainierten Engine zu validieren, wird ein automatisches Turnier gegen eine Benchmark-Engine (Stockfish) durchgeführt.

#### a) Testumgebung einrichten

1.  Erstellen Sie ein `tools/`-Verzeichnis im Projekstamm.
2.  Laden Sie die `cutechess-cli`-Executable für Ihr Betriebssystem herunter und platzieren Sie sie als `cutechess-cli` in diesem Verzeichnis.
    -   **Quelle:** [Cute Chess GitHub Releases](https://github.com/cutechess/cutechess/releases) (Suchen Sie nach der `.AppImage`-Datei für Linux).
3.  Laden Sie eine Stockfish-Engine-Executable herunter und platzieren Sie sie als `stockfish` in diesem Verzeichnis.
    -   **Quelle:** [Stockfish Downloads](https://stockfishchess.org/download/) (Die AVX2-Version wird für moderne CPUs empfohlen).

*Hinweis: Das `tools/`-Verzeichnis wird von Git ignoriert. Sie müssen diese Dateien manuell bereitstellen.*

#### b) Testlauf durchführen

Führen Sie das Test-Skript aus, um ein Turnier mit 20 Partien zu starten:
```bash
python run_tests.py
```
Das Skript führt das Turnier durch, speichert die Ergebnisse in `results.txt` und gibt am Ende eine klare Erfolgs- oder Fehlermeldung aus, je nachdem, ob die RCN-Engine mehr als 50% der Punkte erzielt hat.

### (Manuelle Nutzung) Engine ausführen

Um die Schach-Engine manuell über die UCI-Schnittstelle zu starten (z.B. zur Verwendung mit einer GUI wie Cute Chess), führen Sie aus:
```bash
python main.py run-engine
```
