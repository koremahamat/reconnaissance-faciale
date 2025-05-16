import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import cosine
from config import DEVICE

# Charger les modèles à un niveau global
mtcnn = MTCNN(keep_all=True, device=DEVICE)
model = InceptionResnetV1(pretrained='vggface2', classify=False).eval().to(DEVICE)

def load_model():
    """Renvoie les modèles MTCNN et InceptionResnetV1."""
    return mtcnn, model

def extract_embedding(face_tensor):
    """Extrait l'embedding à partir d'un tenseur de visage."""
    with torch.no_grad():
        embedding = model(face_tensor).detach().cpu().numpy()
    return embedding

def comparer_embeddings(embedding_extrait, embeddings_lfw):
    """Compare l'embedding extrait avec ceux du dataset LFW."""
    distances = [cosine(embedding_extrait.flatten(), lfw.flatten()) for lfw in embeddings_lfw]
    return distances