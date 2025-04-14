import cv2
import getpass
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes

# Générer une clé privée pour l'utilisateur
private_key = ec.generate_private_key(ec.SECP256R1(), default_backend())
public_key = private_key.public_key()

# Fonction pour chiffrer les données
def encrypt_data(data, public_key):
    ciphertext = public_key.encrypt(
        data.encode(),
        padding.ECIES(
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return ciphertext

# Fonction pour déchiffrer les données
def decrypt_data(ciphertext, private_key):
    plaintext = private_key.decrypt(
        ciphertext,
        padding.ECIES(
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return plaintext.decode()

# Fonction d'authentification
def authenticate():
    # Demander le mot de passe
    password = getpass.getpass("Entrez le mot de passe pour accéder à la caméra : ")

    # Vous pouvez ici vérifier le mot de passe (ex: avec un hash)
    return password == "koremahamat"  # Remplacez par votre logique

def main():
    if authenticate():
        print("Authentification réussie.")
        
        # Accéder à la caméra
        
    else:
        print("Accès refusé.")

if __name__ == "__main__":
    main()