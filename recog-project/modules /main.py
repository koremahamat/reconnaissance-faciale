import logging
import os
from getpass import getpass  # Pour masquer la saisie du mot de passe
from dataset import extract_embeddings_from_dataset
from recognition import reconnaitre_visage_une_fois
from utils import charger_embeddings_lfw

# Configuration de la journalisation
LOG_FILE = "app.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),  # Sauvegarde des logs dans un fichier
        logging.StreamHandler()        # Affichage des logs dans la console
    ]
)

# Mot de passe de l'application (à remplacer par votre propre mot de passe sécurisé)
APP_PASSWORD = "koremahamat"

def authentification():
    """
    Authentifie l'utilisateur avant de permettre l'accès à l'application.
    """
    logging.info("Veuillez entrer le mot de passe pour accéder à l'application.")
    for _ in range(3):  # Limite à 3 tentatives
        password = getpass("Mot de passe : ")
        if password == APP_PASSWORD:
            logging.info("Authentification réussie.")
            return True
        else:
            logging.warning("Mot de passe incorrect. Veuillez réessayer.")
    logging.error("Trop de tentatives échouées. Accès refusé.")
    return False

def verifier_fichiers(embeddings_file, labels_file):
    """Vérifie l'existence et l'accessibilité des fichiers nécessaires."""
    if not os.path.exists(embeddings_file):
        logging.error(f"Le fichier d'embeddings est introuvable : {embeddings_file}")
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
    if not os.path.exists(labels_file):
        logging.error(f"Le fichier de labels est introuvable : {labels_file}")
        raise FileNotFoundError(f"Labels file not found: {labels_file}")
    logging.info("Tous les fichiers nécessaires sont disponibles.")

def main():
    """Point d'entrée principal du programme."""
    try:
        # Authentification
        if not authentification():
            return  # Quitter l'application si l'authentification échoue

        # Étape 1 : Extraction des embeddings depuis le dataset
        logging.info("Début de l'extraction des embeddings depuis le dataset...")
        embeddings_file, labels_file = extract_embeddings_from_dataset()

        # Étape 2 : Vérifier les fichiers extraits
        verifier_fichiers(embeddings_file, labels_file)

        # Étape 3 : Charger les embeddings extraits
        logging.info("Chargement des embeddings extraits...")
        embeddings_lfw = charger_embeddings_lfw(embeddings_file)
        labels_lfw = charger_embeddings_lfw(labels_file)
        logging.info("Embeddings LFW chargés : %d", len(embeddings_lfw))

        # Étape 4 : Démarrer la reconnaissance via webcam
        logging.info("Démarrage de la reconnaissance faciale via webcam...")
        reconnaitre_visage_une_fois(embeddings_lfw, labels_lfw)

    except FileNotFoundError as fnf_error:
        logging.error(f"Fichier manquant : {fnf_error}")
    except ValueError as value_error:
        logging.error(f"Erreur de valeur : {value_error}")
    except Exception as e:
        logging.error(f"Une erreur inattendue est survenue : {e}")
    finally:
        logging.info("Programme terminé.")

if __name__ == "__main__":
    main()