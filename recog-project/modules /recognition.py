import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw, ImageFont
from detection import detect_faces
from embeddings import extract_embedding, comparer_embeddings
from config import RECOGNITION_THRESHOLD
import logging
import numpy as np
import torch


def afficher_visage(image_capturee, nom_reconnu):
    """Affiche l'image capturée avec le nom reconnu."""
    draw = ImageDraw.Draw(image_capturee)
    font = ImageFont.load_default()  # Charger une police par défaut
    draw.text((10, 10), f"Reconnu : {nom_reconnu}", fill="green", font=font)
    image_capturee.show(title=f"Reconnu : {nom_reconnu}")


def reconnaitre_visage_une_fois(embeddings_lfw, labels_lfw):
    """Reconnaître des visages à partir de la webcam une seule fois."""
    # Initialisation du modèle et du détecteur
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=True, device=device)
    model = InceptionResnetV1(pretrained='vggface2', classify=False).eval().to(device)

    # Accès à la webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Erreur : Impossible d'ouvrir la webcam.")
        return

    ret, frame = cap.read()
    if not ret:
        logging.error("Erreur : Impossible de capturer l'image.")
        cap.release()
        return

    # Conversion de l'image en format PIL
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Détection des visages
    boxes, _ = mtcnn.detect(img)
    if boxes is not None:
        for box in boxes:
            # Recadrage et redimensionnement du visage
            face = img.crop((box[0], box[1], box[2], box[3])).resize((160, 160))

            # Convertir le visage en tenseur
            face_tensor = mtcnn(face)
            if face_tensor is not None:
                # Vérifier que le tenseur a la dimension correcte
                if len(face_tensor.shape) == 3:  # Un seul visage
                    face_tensor = face_tensor.unsqueeze(0)  # Ajouter une dimension batch

                try:
                    # Passage du tenseur au modèle pour générer l'embedding
                    embedding = model(face_tensor).detach().cpu().numpy()
                    logging.info(f"Dimension de l'embedding : {embedding.shape}")

                    # Comparaison avec les embeddings LFW
                    distances = comparer_embeddings(embedding, embeddings_lfw)

                    # Identifier le meilleur match
                    best_match_index = np.argmin(distances)
                    if distances[best_match_index] < RECOGNITION_THRESHOLD:  # Seuil de reconnaissance
                        nom_reconnu = labels_lfw[best_match_index]
                        logging.info(f"Visage reconnu : {nom_reconnu}")

                        # Afficher l'image capturée avec le nom reconnu
                        afficher_visage(img, nom_reconnu)
                    else:
                        logging.info("Visage non reconnu.")
                except Exception as e:
                    logging.error(f"Erreur lors du traitement du tenseur : {e}")
            else:
                logging.error("Le tenseur de visage est vide ou invalide.")
    else:
        logging.error("Aucun visage détecté.")

    # Libérer la webcam
    cap.release()
    cv2.destroyAllWindows()