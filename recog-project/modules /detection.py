from facenet_pytorch import MTCNN
from config import DEVICE

# Initialisation du détecteur de visages
mtcnn = MTCNN(keep_all=True, device=DEVICE)

def detect_faces(image):
    """Détecte les visages dans une image."""
    boxes, _ = mtcnn.detect(image)
    return boxes