import logging
from .config import DEVICE, DATASET_DIR

# Configuration de la journalisation
logging.basicConfig(level=logging.INFO)
logging.info(f"Package initialisé. Utilisation de l'appareil : {DEVICE}")

# Création du répertoire pour les images capturées (si nécessaire)
import os
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)
    logging.info(f"Répertoire créé pour les images capturées : {DATASET_DIR}")