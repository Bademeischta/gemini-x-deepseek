import torch
import os
from scripts.model import RCNModel

# Pfad, an dem die Engine das Modell erwartet
MODEL_PATH = "models/rcn_model.pth"

def create_dummy_model():
    """
    Initialisiert ein RCNModel mit zufälligen Gewichten und speichert es,
    um die Entwicklung der Engine zu ermöglichen, bevor das Training abgeschlossen ist.
    """
    print("Erstelle ein untrainiertes Dummy-Modell für Entwicklungszwecke...")

    # 1. Sicherstellen, dass das Verzeichnis existiert
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # 2. Modell instanziieren
    model = RCNModel()

    # 3. Modellgewichte speichern
    torch.save(model.state_dict(), MODEL_PATH)

    print(f"Dummy-Modell erfolgreich gespeichert unter: {MODEL_PATH}")

if __name__ == "__main__":
    create_dummy_model()
