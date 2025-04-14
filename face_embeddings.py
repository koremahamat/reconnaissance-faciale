import os
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import numpy as np

# Initialiser les modèles
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)  # Détecteur de visages
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)  # Extrait les embeddings

def get_face_embedding(face_image):
    """Convertit une image de visage en vecteur (embedding)."""
    face = mtcnn(face_image)  # Détection et alignement
    if face is not None:
        face_tensor = face.to(device)  # Déplace le tensor vers le bon device
        embedding = resnet(face_tensor)  # Vecteur de 512 dimensions
        return embedding.detach().cpu().numpy()  # Retourne un array NumPy
    return None

def process_images_from_directory(directory):
    """Traitement des images dans un répertoire donné pour extraire les embeddings."""
    embeddings = {}

    # Parcourir tous les fichiers dans le répertoire
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(directory, filename)
            face_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

            # Extraire l'embedding
            embedding = get_face_embedding(face_image)
            if embedding is not None:
                embeddings[filename] = embedding
                print(f"Image: {filename}, Embedding shape: {embedding.shape}")
            else:
                print(f"Aucun visage détecté dans : {filename}")

    return embeddings

if __name__ == "__main__":
    dataset_dir = "Dataset/nom_de_la_personne"  # Remplacer par le chemin de votre dossier
    embeddings = process_images_from_directory(dataset_dir)
    
    # Optionnel : enregistrer les embeddings dans un fichier
    np.savez("embeddings.npz", **embeddings)

    #https://kean-chan.medium.com/real-time-facial-recognition-with-pytorch-facenet-ca3f6a510816