import cv2

# Ouvrir la webcam
cap = cv2.VideoCapture(0)

# Vérifier si la webcam s'est ouverte correctement
if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la webcam.")
    exit()

# Lire une image
ret, frame = cap.read()

# Vérifier si l'image a été capturée
if ret:
    # Enregistrer l'image sous le nom "KORE.jpg"
    cv2.imwrite('KORE.jpg', frame)
    print("Photo enregistrée sous 'KORE.jpg'")
else:
    print("Erreur : Impossible de capturer l'image.")

# Libérer la webcam
cap.release()