from facenet_pytorch import MTCNN
import cv2
import os
from PIL import Image
import numpy as np
from rich.progress import Progress

# Créer le détecteur de visages (utiliser le CPU)
mtcnn = MTCNN(margin=20, keep_all=True, post_process=False, device='cpu')

# Créer la base de données
dataset_dir = "Dataset"
os.makedirs(dataset_dir, exist_ok=True)

def create_dataset(name):
    person_dir = os.path.join(dataset_dir, name)
    os.makedirs(person_dir, exist_ok=True)
    return person_dir  # Retourner le chemin du dossier créé

# Remplacer 'name' par le nom de la personne
name = "noms"  # À définir avant de créer le dataset
person_dir = create_dataset(name)

cap = cv2.VideoCapture(0)
count = 0

with Progress() as progress:
    task = progress.add_task("Capturing images...", total=50)

    while not progress.finished:
        ret, frame = cap.read()
        if not ret:
            print("Pas de capture")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        # Détecter les visages
        boxes, _ = mtcnn.detect(frame_pil)

        # Si des visages sont détectés
        if boxes is not None:
            for i, box in enumerate(boxes):
                if count < 50:
                    # Obtenez les coordonnées du cadre
                    x1, y1, x2, y2 = box.astype(int)

                    # Sauvegarder le visage détecté
                    face_image = frame[y1:y2, x1:x2]
                    #face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                    face_path = os.path.join(person_dir, f"face_{count}.jpg")
                    Image.fromarray(face_image).save(face_path)
                    count += 1
                    progress.update(task, advance=1)

                    # Dessiner un rectangle autour du visage
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Afficher la vidéo avec les cadres
        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:
            break

cap.release()
cv2.destroyAllWindows()