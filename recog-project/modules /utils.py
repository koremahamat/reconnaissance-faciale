import numpy as np
import logging

def charger_embeddings_lfw(chemin_fichier):
    """Charge les embeddings depuis un fichier."""
    try:
        return np.load(chemin_fichier)
    except FileNotFoundError:
        logging.error(f"Erreur : Le fichier {chemin_fichier} est introuvable.")
        exit()