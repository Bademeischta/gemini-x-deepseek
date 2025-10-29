
import subprocess
import re
import sys
import os

# --- Configuration ---
# Pfade zu den ausführbaren Dateien
CUTECHESS_CLI_PATH = "./tools/cutechess-cli"
STOCKFISH_PATH = "./tools/stockfish"

# Engine-Konfiguration
RCN_ENGINE_NAME = "RCN-Engine"
# Befehl zum Starten unserer Engine. sys.executable stellt sicher, dass der richtige Python-Interpreter verwendet wird.
RCN_ENGINE_CMD = f"{sys.executable} main.py run-engine"
STOCKFISH_NAME = "Stockfish"

# Turnier-Parameter
NUM_GAMES = 20
TIME_CONTROL_SECONDS = 10
RESULTS_FILE = "results.txt"

def run_tournament():
    """
    Führt ein Schachturnier mit cutechess-cli als externem Prozess durch.
    Die Ausgabe wird zur späteren Analyse in eine Datei umgeleitet.
    """
    print("--- Starte RCN Engine Validierungsturnier ---")

    # Überprüfen, ob die erforderlichen Tools vorhanden sind
    for path in [CUTECHESS_CLI_PATH, STOCKFISH_PATH]:
        if not os.path.exists(path):
            print(f"Fehler: Erforderliches Tool nicht gefunden unter '{path}'", file=sys.stderr)
            print("Bitte folgen Sie den Anweisungen in der README.md, um die Testumgebung einzurichten.", file=sys.stderr)
            sys.exit(1)

    # Befehl für cutechess-cli zusammenbauen
    command = [
        CUTECHESS_CLI_PATH,
        "-engine", f"name={RCN_ENGINE_NAME}", f"cmd={RCN_ENGINE_CMD}",
        "-engine", f"name={STOCKFISH_NAME}", f"cmd={STOCKFISH_PATH}",
        "-each", f"tc={TIME_CONTROL_SECONDS}",
        "-games", str(NUM_GAMES),
        # PGN-Daten der Partien werden ebenfalls in die Ergebnisdatei geschrieben
        "-pgnout", RESULTS_FILE
    ]

    print(f"Führe Befehl aus: {' '.join(command)}")

    try:
        # Starte den Prozess und leite stdout in die Ergebnisdatei um
        with open(RESULTS_FILE, "w") as f:
            # Wir leiten stderr nach stdout um, damit alle Ausgaben in einer Datei landen
            process = subprocess.run(
                command,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                check=True
            )
        print(f"Turnier erfolgreich beendet. Ergebnisse in '{RESULTS_FILE}' gespeichert.")
    except FileNotFoundError:
        print(f"Fehler: Der Befehl '{CUTECHESS_CLI_PATH}' wurde nicht gefunden.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Fehler während der Ausführung von cutechess-cli (Exit-Code: {e.returncode}).", file=sys.stderr)
        print(f"Überprüfen Sie die '{RESULTS_FILE}' für detaillierte Fehlermeldungen.", file=sys.stderr)
        sys.exit(1)

def parse_and_validate_results():
    """
    Parst die Ergebnisdatei, extrahiert die Punktzahl und validiert das Ergebnis.
    """
    print("\n--- Parse und validiere Ergebnisse ---")

    try:
        with open(RESULTS_FILE, "r") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Fehler: Ergebnisdatei '{RESULTS_FILE}' nicht gefunden.", file=sys.stderr)
        sys.exit(1)

    # Regulärer Ausdruck, um die finale Punktetabelle zu finden.
    # Sucht nach "Score of RCN-Engine vs Stockfish: 11.5 / 20"
    score_pattern = re.compile(
        r"Score of " + re.escape(RCN_ENGINE_NAME) + r" vs " + re.escape(STOCKFISH_NAME) +
        r":\s*([\d\.]+)\s*/\s*" + str(NUM_GAMES)
    )

    match = score_pattern.search(content)

    if not match:
        print(f"Fehler: Konnte die Punktzahl für '{RCN_ENGINE_NAME}' nicht in '{RESULTS_FILE}' finden.", file=sys.stderr)
        print("Das Turnier wurde möglicherweise nicht korrekt abgeschlossen.", file=sys.stderr)
        sys.exit(1)

    rcn_score = float(match.group(1))
    target_score = NUM_GAMES / 2.0

    print(f"RCN-Engine Punktzahl: {rcn_score}/{NUM_GAMES}")
    print(f"Benötigte Punktzahl für Erfolg: > {target_score}/{NUM_GAMES}")

    # Validierung des Ergebnisses
    if rcn_score > target_score:
        print("\n--- VALIDIERUNG ERFOLGREICH ---")
        print("Die RCN-Engine hat die Benchmark von über 50% der Punkte erreicht.")
        return True
    else:
        print("\n--- VALIDIERUNG FEHLGESCHLAGEN ---")
        print("Die RCN-Engine hat die Benchmark von über 50% nicht erreicht.")
        return False

if __name__ == "__main__":
    run_tournament()
    if parse_and_validate_results():
        sys.exit(0)  # Erfolgreicher Exit-Code
    else:
        sys.exit(1)  # Fehler-Exit-Code
