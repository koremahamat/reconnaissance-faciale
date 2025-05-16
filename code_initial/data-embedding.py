import torchvision.transforms as transforms
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import numpy as np
import os
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# Chemin vers le dataset LFW
dataset_path = '/home/kore/reconnaissace-faciale/recog-project/lfw-deepfunneled'  
embeddings = []
labels = []

# Charger le modèle préentraîné Inception-ResNet
def load_model():
    mtcnn = MTCNN(keep_all=True, device=device)
    model = InceptionResnetV1(pretrained='vggface2', classify=False).eval().to(device)
    return mtcnn, model

mtcnn, model = load_model()

# Parcourir les sous-dossiers (chaque dossier correspond à une personne)
for label in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, label)
    if os.path.isdir(person_folder):
        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            if os.path.isfile(img_path):
                img = Image.open(img_path).convert('RGB')

                # Détection des visages
                faces = mtcnn(img)
                if faces is not None:
                    for face in faces:
                        face_embedding = model(face.unsqueeze(0)).detach().cpu().numpy()
                        print(f"Taille de l'embedding : {face_embedding.shape}")  # Ajout
                        embeddings.append(face_embedding)
                        labels.append(label)
                else:
                    print(f"Aucun visage détecté dans l'image : {img_path}")
            else:
                print(f"Image introuvable : {img_path}")

# Concaténer les embeddings
if embeddings:
    embeddings = np.vstack(embeddings)
    np.save('lfw_embeddings.npy', embeddings)
    np.save('lfw_labels.npy', np.array(labels))
    print("Embeddings extraits et sauvegardés.")
else:
    print("Aucun embedding trouvé.")

# Pour charger et vérifier les fichiers sauvegardés
loaded_embeddings = np.load('lfw_embeddings.npy')
loaded_labels = np.load('lfw_labels.npy')

print("Chargement des données :")
print("Embeddings shape:", loaded_embeddings.shape)
print("Labels shape:", loaded_labels.shape)