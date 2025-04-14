import cv2

# Chargement de l'image et de la vidéo
img = cv2.imread('images/femme.jpeg')
video = cv2.VideoCapture('images/video.mp4')
cv2.imshow('image', img)
"""
# Récupération des dimensions de l'image
height, width, channels = img.shape
print(f"Taille de l'image : {width} x {height}, Canaux : {channels}")
"""
"""# Vérification de l'ouverture de la vidéo
if not video.isOpened():
    print("Erreur : impossible d'ouvrir la vidéo.")
    exit()

# Boucle de lecture de la vidéo
while True:
    ret, frame = video.read()
    if not ret:
        print("Fin de la vidéo.")
        break
"""
    # Conversion de l'image en niveaux de gris
gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # adaptive_frame = cv2.adaptiveThreshold(gray_frame, 255,
    #                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                        cv2.THRESH_BINARY, 11, 2)
    # blurred_frame = cv2.GaussianBlur(adaptive_frame, (5, 5), 0)
# Détection des visages
face_cascade = cv2.CascadeClassifier('/home/kore/reconnaissace-faciale/venv/lib/python3.13/site-packages/cv2/data/haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    # Dessin des rectangles autour des visages détectés
for (x, y, w, h) in faces:
        cv2.rectangle(gray_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # Affichage des résultats
cv2.imshow('image', img)
cv2.imshow('image en Niveaux de Gris', gray_frame)
    # cv2.imshow('Seuillage Adaptatif', adaptive_frame)
    # cv2.imshow('Flou Gaussien', blurred_frame)
"""
    # Sortir de la boucle si 'q' est pressé
if cv2.waitKey(100) & 0xFF == ord('q'): 
break 
"""
cv2.waitKey(0)
video.release()
cv2.destroyAllWindows()