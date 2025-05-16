from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.distance import cosine
import logging
import os

# Configuration de la journalisation
logging.basicConfig(level=logging.INFO)

# Initialisation du détecteur de visages et du modèle d'embeddings
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
model = InceptionResnetV1(pretrained='vggface2', classify=False).eval().to(device)

# Chargement des embeddings LFW et des labels
def charger_embeddings_lfw(chemin_fichier):
    """Charge les embeddings depuis un fichier."""
    try:
        return np.load(chemin_fichier)
    except FileNotFoundError:
        logging.error("Erreur : Le fichier des embeddings LFW est introuvable.")
        exit()

def comparer_embeddings(embedding_extrait, embeddings_lfw):
    """Compare l'embedding extrait avec ceux du dataset LFW."""
    distances = [cosine(embedding_extrait.flatten(), lfw.flatten()) for lfw in embeddings_lfw]
    return distances

def afficher_visage(image_capturee, nom_reconnu):
    """Affiche l'image capturée avec le nom reconnu."""
    draw = ImageDraw.Draw(image_capturee)
    font = ImageFont.load_default()  # Charger une police par défaut
    draw.text((10, 10), f"Reconnu : {nom_reconnu}", fill="white", font=font)

    # Afficher l'image avec le nom
    image_capturee.show(title=f"Reconnu : {nom_reconnu}")

def reconnaitre_visage_image(image_path, embeddings_lfw, labels_lfw):
    """Reconnaître un visage à partir d'une image fournie."""
    img = Image.open(image_path)
    img_tensor = mtcnn(img)  # Détecter les visages

    if img_tensor is not None and img_tensor.size(0) > 0:
        embedding = model(img_tensor).detach().cpu().numpy()
        logging.info(f"Dimension de l'embedding : {embedding.shape}")

        if embedding.shape[1] != 512:
            logging.error(f"Dimension d'embedding inattendue : {embedding.shape[1]}")
            return

        distances = comparer_embeddings(embedding, embeddings_lfw)

        # Identifier le meilleur match
        best_match_index = np.argmin(distances)
        if distances[best_match_index] < 0.6:  # Seuil de reconnaissance
            nom_reconnu = labels_lfw[best_match_index]
            logging.info(f"Visage reconnu : {nom_reconnu}")
            
            # Afficher l'image avec le nom reconnu
            afficher_visage(img, nom_reconnu)
        else:
            logging.info("Visage non reconnu.")
    else:
        logging.info("Aucun visage détecté dans l'image.")

# Charger les embeddings LFW et les labels
embeddings_lfw = charger_embeddings_lfw('lfw_embeddings.npy')
labels_lfw = np.load('lfw_labels.npy')
logging.info("Embeddings LFW chargés : %d", len(embeddings_lfw))

# Chemin de l'image à reconnaître
image_path = '/home/kore/reconnaissace-faciale/images/sener4.jpeg'  # Remplacez par le chemin de votre image
reconnaitre_visage_image(image_path, embeddings_lfw, labels_lfw)