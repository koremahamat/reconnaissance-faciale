import torch
import os

# Configuration du périphérique
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Répertoire pour les images capturées
DATASET_DIR = "Dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

# Chemin vers le dataset LFW
DATASET_PATH = '/home/kore/reconnaissace-faciale/recog-project/lfw-deepfunneled'

# Chemins pour les fichiers sauvegardés
EMBEDDINGS_FILE = 'lfw_embeddings.npy'
LABELS_FILE = 'lfw_labels.npy'

# Seuil de reconnaissance
RECOGNITION_THRESHOLD = 0.6