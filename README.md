# Projekt "Ressourcen-Effiziente Dominanz" (RCN)

Dieses Projekt zielt darauf ab, eine neuartige Schach-KI zu entwickeln, die das Potenzial hat, State-of-the-Art-Engines zu übertreffen, indem sie auf extrem ressourcen-effizienten Methoden basiert.

## Kernkonzept

Das System basiert auf drei Säulen:
1.  **Daten (KKK):** Ein kompilierter, dichter Korpus aus existierenden, hochwertigen Analysedaten (`Kompilierter Kritischer Korpus`).
2.  **Architektur (RCN):** Ein Graph Attention Network (`Relational Chess Net`), das Schach als Beziehungsgeflecht modelliert.
3.  **Inferenz (IR-AB):** Eine CPU-basierte Alpha-Beta-Suche, die durch die GPU-gestützte neuronale Intelligenz geführt wird (`Information-Rich Alpha-Beta`).

---

## Projekt-Status: Überarbeitung (Stand: 2025-11-05)

Nach einer kritischen Analyse wurden fundamentale Fehler in der Implementierung identifiziert. Das Projekt befindet sich derzeit in einer umfassenden Überarbeitungsphase, um diese Probleme zu beheben.

### ✅ Abgeschlossen

**Phase 0 & 1: Kritische Algorithmus-Korrekturen**
-   [x] **Fix 1: `uci_to_index` Crash:** `uci_to_index` gibt nun `0` für ungültige Züge zurück und verhindert Abstürze.
-   [x] **Fix 2: Echte Negamax-Implementierung:** Die fehlerhafte Minimax/Negamax-Hybrid-Suche wurde durch eine korrekte Negamax-Implementierung ersetzt.
-   [x] **Fix 3: PV-Rekonstruktion:** Die Suchfunktion gibt nun den besten Zug zurück, was eine korrekte Rekonstruktion der Principal Variation ermöglicht.
-   [x] **Fix 4: Quiescence-Search-Logik:** Die Tiefenprüfung erfolgt nun vor der Evaluierung, um Endlosrekursionen zu verhindern.
-   [x] **Fix 5: Dataset Memory Leak:** `ChessGraphDataset` ist nun ein robuster Kontextmanager, um das Schließen von Dateihandles zu garantieren.
-   [x] **Fix 6: UCI Race Condition:** Der `isready`-Handler wartet nun auf die vollständige Initialisierung der Engine.
-   [x] **Fix 7: Zeitmanagement-Präzision:** Die Zeitberechnung erfolgt nun mit Ganzzahlen (Millisekunden), um Rundungsfehler zu vermeiden.
-   [x] **Fix 9: Gradient Clipping:** Dem Trainingsprozess wurde Gradient Clipping hinzugefügt, um die Stabilität zu erhöhen.
-   [x] **Fix 19 & Zusätzliche Anforderung 3 (teilweise): Move-Ordering-Skalierung:** Policy-Logits werden mit Softmax normalisiert und MVV-LVA-Scores skaliert.
-   [x] **Fix 18 & Zusätzliche Anforderung 1: Duplikate in Trainingsdaten:** Die Datenverarbeitung verhindert nun doppelte Stellungen.
-   [x] **Fix 15 & 22: Logging-Rotation:** Die Engine verwendet nun einen `RotatingFileHandler`, um unbegrenztes Wachstum der Log-Datei zu verhindern.
-   [x] **Fix 16 & 23: Magic Numbers entfernt:** Hartcodierte Konstanten wurden in eine zentrale `config.py`-Datei ausgelagert.
-   [x] **Fix 12: Value-Head Normalisierung:** Die `tanh`-Aktivierung des Value-Heads wurde bestätigt.
-   [x] **Fix 13 & Zusätzliche Anforderung 4: Erweiterte Edge-Features (Pins & X-Rays):** Die Graphen-Erstellung erkennt nun Fesselungen (Pins) und "X-Ray"-Angriffe als eigene Kantentypen.
-   [x] **Fix 8: Vollständige Graph-Features:** Die Graphen-Daten enthalten nun alle globalen Zustandsinformationen (Rochaderechte, En-Passant, 50-Züge-Regel).
-   [x] **Fix 15: Batch Normalization:** Dem Modell wurden `BatchNorm`-Schichten hinzugefügt, um das Training zu stabilisieren.
-   [x] **Fix 17 & Zusätzliche Anforderung 2: Test-Framework:** Ein Test-Framework zum Spielen von Matches gegen Stockfish wurde implementiert.

**Architektur-Korrekturen**
-   [x] **Global Features Batching:** Das kritische Batching-Problem für globale Features wurde gelöst.

### 🔴 Noch fehlend

-   [ ] **Fix 17: Move Generation Caching:** Caching für die Zug-Sortierung ist nicht implementiert.
-   [ ] **Fix 19 (Engine): Tree Reuse:** Die Wiederverwendung von Teilen des Suchbaums zwischen den Zügen fehlt.
-   [ ] **Fix 18 (Performance): Graph-Erstellung optimieren:** Die `fen_to_graph_data`-Funktion ist noch nicht auf Performance optimiert.
-   [ ] **Fix 21: Einheitliches Error-Handling:** Das Error-Handling im Projekt ist noch inkonsistent.
-   [ ] **Fix 24 & 25: Type Hints & Docstrings:** Viele Funktionen haben noch keine vollständigen Type Hints oder Docstrings.
-   [ ] **Integrationstests:** Es fehlen dedizierte Integrationstests für die UCI-Kommunikation und das Zusammenspiel der Suchkomponenten.
-   [ ] **Profiling:** Es wurde noch kein formelles Performance- und Speicher-Profiling durchgeführt.
-   [ ] **Dokumentation:** Eine `CHANGELOG.md` und ein Benchmark-Report fehlen noch.

---

## Nutzung

Dieses Projekt kann sowohl lokal auf Ihrem Rechner als auch in Google Colab ausgeführt werden.

### Lokale Nutzung

**1. Voraussetzungen:**
- Python 3.8 oder höher
- Git

**2. Installation:**
Klonen Sie das Repository und installieren Sie die Abhängigkeiten:
```bash
git clone https://github.com/Bademeischta/gemini-x-deepseek
cd gemini-x-deepseek
pip install -r requirements.txt
```

**3. Daten und Modelle vorbereiten:**
- Platzieren Sie Ihre Trainingsdaten (im `.jsonl`-Format) im `data/`-Verzeichnis.
- Trainierte Modelle werden standardmäßig im `models/`-Verzeichnis gespeichert und von dort geladen.

**4. Training starten:**
```bash
python train.py
```

**5. Engine verwenden:**
Die Engine kommuniziert über das UCI-Protokoll. Sie können sie in jeder UCI-kompatiblen Schach-GUI verwenden, indem Sie den folgenden Befehl als Engine-Pfad angeben:
```bash
python engine.py
```

### Google Colab Nutzung

**1. Notebook einrichten:**
Öffnen Sie ein neues Colab-Notebook und stellen Sie sicher, dass Sie eine GPU-Laufzeit verwenden (`Laufzeit` -> `Laufzeittyp ändern` -> `T4 GPU`).

**2. Projekt klonen und installieren:**
Führen Sie die folgenden Befehle in einer Code-Zelle aus, um das Projekt zu klonen und die Abhängigkeiten zu installieren:
```python
!git clone https://github.com/Bademeischta/gemini-x-deepseek
%cd gemini-x-deepseek
!pip install -r requirements.txt
```

**3. Daten-Upload:**
Laden Sie Ihre `.jsonl`-Datensätze in den `data/`-Ordner im Colab-Dateisystem hoch. Sie können dies manuell über die Seitenleiste tun oder `gdown` bzw. `wget` verwenden, wenn Ihre Daten online verfügbar sind.

**4. Training in Colab:**
Starten Sie das Training, indem Sie das Skript aus einer Zelle heraus aufrufen:
```python
!python train.py
```
Das trainierte Modell (`rcn_model.pth`) finden Sie danach im `models/`-Verzeichnis.

**5. Engine in Colab (für Analyse):**
Obwohl eine direkte GUI-Anbindung in Colab nicht möglich ist, können Sie die Engine programmatisch für Analysen verwenden. Hier ist ein Beispiel, wie Sie die Engine mit `python-chess` steuern können:

```python
import chess
import chess.engine

# Pfad zur Engine in Colab
engine_path = "/content/gemini-x-deepseek/engine.py"

# Engine starten
engine = chess.engine.SimpleEngine.popen_uci(["python", engine_path])

board = chess.Board()
info = engine.analyse(board, chess.engine.Limit(time=1.0))

print("Bewertung:", info["score"])

# Engine beenden
engine.quit()
```
