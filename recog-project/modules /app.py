from flask import Flask, render_template, request, jsonify
import os
import sys
import logging

# Ajoute le dossier courant au PYTHONPATH pour les imports locaux
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Adapte les imports à ton projet
from dataset import extract_embeddings_from_dataset
from recognition import reconnaitre_visage_une_fois
from utils import charger_embeddings_lfw

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)

embeddings_lfw, labels_lfw = None, None

def charger_embeddings_si_necessaire():
    global embeddings_lfw, labels_lfw
    if embeddings_lfw is None or labels_lfw is None:
        logging.info("Chargement des embeddings et labels...")
        embeddings_file, labels_file = extract_embeddings_from_dataset()
        embeddings_lfw = charger_embeddings_lfw(embeddings_file)
        labels_lfw = charger_embeddings_lfw(labels_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        charger_embeddings_si_necessaire()
        image_file = request.files.get('image')
        # Ici, tu passes image_file à ta fonction de reconnaissance
        result = reconnaitre_visage_une_fois(embeddings_lfw, labels_lfw, image_file)
        message = result if result else "Reconnaissance faciale terminée avec succès."
        return jsonify(status="ok", message=message)
    except Exception as e:
        logging.error(f"Erreur lors du traitement : {e}")
        return jsonify(status="error", message=f"Erreur : {e}")

if __name__ == '__main__':
    app.run(debug=True)