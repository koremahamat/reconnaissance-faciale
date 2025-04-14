import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity

def compare_embeddings(embedding1, embedding2):
    # Calculer la distance euclidienne
    euclidean_distance = euclidean(embedding1, embedding2)
    
    # Calculer la similarité cosinus
    cosine_sim = cosine_similarity(embedding1, embedding2)[0][0]
    
    return euclidean_distance, cosine_sim

# Exemple d'utilisation
if __name__ == "__main__":
    # Supposons que vous ayez deux embeddings
    embedding1 = np.random.rand(512)  # Remplacer par votre premier embedding
    embedding2 = np.random.rand(512)  # Remplacer par votre deuxième embedding

    distance, similarity = compare_embeddings(embedding1, embedding2)
    print(f"Distance Euclidienne: {distance}")
    print(f"Similarité Cosinus: {similarity}")

    # Définir un seuil pour décider si les visages sont similaires
    threshold = 0.7  # Exemple de seuil pour la similarité cosinus
    if similarity > threshold:
        print("Les visages sont similaires.")
    else:
        print("Les visages ne sont pas similaires.")