import os
import sys
import logging

from flask import Flask, render_template, request, redirect, url_for

# Ajoute le dossier courant au PYTHONPATH pour les imports locaux
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import des modules (adapte les chemins si besoin)
from dataset import extract_embeddings_from_dataset
from recognition import reconnaitre_visage_une_fois
from utils import charger_embeddings_lfw

# Configuration de la journalisation
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Variables globales pour les embeddings (chargés une fois)
embeddings_lfw, labels_lfw = None, None

def charger_embeddings_si_necessaire():
    """Charge les embeddings et labels si ce n'est pas déjà fait."""
    global embeddings_lfw, labels_lfw
    if embeddings_lfw is None or labels_lfw is None:
        logging.info("Chargement des embeddings et labels...")
        embeddings_file, labels_file = extract_embeddings_from_dataset()
        embeddings_lfw = charger_embeddings_lfw(embeddings_file)
        labels_lfw = charger_embeddings_lfw(labels_file)

@app.route('/')
def index():
    """Affiche la page d'accueil."""
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    """Traite la reconnaissance faciale et affiche le résultat."""
    try:
        # Ici, tu peux récupérer l'image envoyée par l'utilisateur :
        image_file = request.files.get('image')
        charger_embeddings_si_necessaire()
        logging.info("Démarrage de la reconnaissance faciale...")
        result = reconnaitre_visage_une_fois(embeddings_lfw, labels_lfw)
        message = result if result else "Reconnaissance faciale terminée avec succès."
        return render_template('result.html', message=message)
    except Exception as e:
        logging.error(f"Erreur lors du traitement : {e}")
        return render_template('result.html', message=f"Erreur : {e}")

if __name__ == '__main__':
    app.run(debug=True)