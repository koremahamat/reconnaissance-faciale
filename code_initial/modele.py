from selenium import webdriver
from selenium.webdriver.common.by import By
from time import sleep
import requests
import os

# Créer un dossier pour enregistrer les images
if not os.path.exists('images'):
    os.makedirs('images')

# Initialiser le navigateur
driver = webdriver.Chrome()  # Assurez-vous que le chemin vers ChromeDriver est correct
driver.get('https://www.google.com/imghp?hl=fr')  # Accéder à Google Images

# Rechercher des images d'oiseaux
search_box = driver.find_element(By.NAME, 'q')
search_box.send_keys('oiseaux')
search_box.submit()

# Attendre que les résultats se chargent
sleep(2)

# Faire défiler la page pour charger plus d'images
for _ in range(3):  # Ajustez le nombre de fois que vous souhaitez faire défiler
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    sleep(2)

# Trouver toutes les images
images = driver.find_elements(By.XPATH, '//img[@class="t0fcAb"]')

for img in images:
    try:
        img_url = img.get_attribute('src')
        if img_url:
            img_data = requests.get(img_url).content
            img_name = os.path.join('images', f"bird_{images.index(img)}.jpg")  # Nommer les images
            with open(img_name, 'wb') as handler:
                handler.write(img_data)
            print(f'Téléchargé : {img_name}')
    except Exception as e:
        print(f'Erreur lors du téléchargement : {e}')

# Fermer le navigateur
driver.quit()