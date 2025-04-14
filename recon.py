import cv2
import os
import numpy as np

# CREATE DATABASE
dir = "Dataset"
os.makedirs(dir, exist_ok=True)

def create_dataset(name):
    person = os.path.join(dir, name)
    os.makedirs(person, exist_ok=True)
    return person  # Retourner le chemin du dossier créé

# Remplacer 'name' par le nom de la personne
name = "nom_de_la_personne"  # À définir avant de créer le dataset
person = create_dataset(name)

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Pas de capture")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('/home/kore/reconnaissace-faciale/venv/lib/python3.13/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        count += 1

        face_path = os.path.join(person, f"{name}_{count}.jpg")
        cv2.imwrite(face_path, face_img)
        cv2.rectangle(gray_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Frame", gray_frame)  # Afficher le cadre avec le visage détecté

    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:
        break

cap.release()  # Corriger ici de cv2.release() à cap.release()
cv2.destroyAllWindows()