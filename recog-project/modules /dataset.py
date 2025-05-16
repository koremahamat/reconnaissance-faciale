import os
import numpy as np
from PIL import Image
from embeddings import load_model
from config import DEVICE, DATASET_PATH, EMBEDDINGS_FILE, LABELS_FILE

def extract_embeddings_from_dataset():
    """Extrait les embeddings des images dans le dataset."""
    embeddings = []
    labels = []

    # Charger le modèle et le détecteur de visages
    mtcnn, model = load_model()

    # Parcourir les sous-dossiers (chaque dossier correspond à une personne)
    for label in os.listdir(DATASET_PATH):
        person_folder = os.path.join(DATASET_PATH, label)
        if os.path.isdir(person_folder):
            for img_name in os.listdir(person_folder):
                img_path = os.path.join(person_folder, img_name)
                if os.path.isfile(img_path):
                    img = Image.open(img_path).convert('RGB')

                    # Détection des visages
                    faces = mtcnn(img)
                    if faces is not None:
                        for face in faces:
                            # Extraire l'embedding
                            face_embedding = model(face.unsqueeze(0)).detach().cpu().numpy()
                            embeddings.append(face_embedding)
                            labels.append(label)
                    else:
                        print(f"Aucun visage détecté dans l'image : {img_path}")
                else:
                    print(f"Image introuvable : {img_path}")

    # Sauvegarder les embeddings et les labels
    if embeddings:
        embeddings = np.vstack(embeddings)
        np.save(EMBEDDINGS_FILE, embeddings)
        np.save(LABELS_FILE, np.array(labels))
        print("Embeddings extraits et sauvegardés.")
    else:
        print("Aucun embedding trouvé.")

    return EMBEDDINGS_FILE, LABELS_FILE