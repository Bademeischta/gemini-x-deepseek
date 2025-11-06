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

## Projektstruktur & Nutzung

(Die restlichen Abschnitte zur Installation und Nutzung bleiben wie zuvor.)
