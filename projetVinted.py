# ==============================================
# PARTIE 0 : IMPORTATION DES BIBLIOTHEQUES
# ==============================================
import cv2
import numpy as np
import os
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time
import sys
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split

# Configuration de l'encodage pour eviter les problèmes de caractères
sys.stdout.reconfigure(encoding='utf-8')


# ==============================================
# PARTIE 1 : SCRAPING DES DONNEES
# ==============================================
def scrapping_images():
    # Recupère les images et metadonnees de vêtements sur Vinted
    
    # Configuration de Chrome
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    # Dictionnaire des URLs et leurs labels
    url_to_label = {
        "https://www.vinted.fr/catalog/1206-outerwear?time=1744054265": 2,            # 2 = Haut
        "https://www.vinted.fr/catalog/76-tops-and-t-shirts?time=1744054265": 2,      # 2 = Haut
        "https://www.vinted.fr/catalog/79-jumpers-and-sweaters?time=1744054265": 2,   # 2 = Haut
        "https://www.vinted.fr/catalog/34-trousers?time=1744054265": 1,               # 1 = Pantalon & Short
        "https://www.vinted.fr/catalog/80-shorts?time=1744054265": 1,                 # 1 = Pantalon & Short
        "https://www.vinted.fr/catalog/1231-shoes?time=1744306415": 0,                # 0 = Chaussures
        "https://www.vinted.fr/catalog/1231-shoes?time=1744306415&page=2": 0          # 0 = Chaussures
    }

    # Initialisation du driver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    clothes = {}

    # Acceder aux differentes pages pour la recuperation des liens des images
    try:
        for url, label in url_to_label.items():
            print(f"Scraping de la page : {url}")
            driver.get(url)
            time.sleep(random.uniform(2, 4))

            # Gestion des cookies pour le pop up si present
            try:
                cookie_button = driver.find_element(By.ID, "onetrust-reject-all-handler")
                cookie_button.click()
                time.sleep(1)
            except:
                print("Pas de popup cookies trouve")
                pass

            # Scraping des articles
            time.sleep(3)
            items = driver.find_elements(By.CSS_SELECTOR, "div.feed-grid__item:not(.feed-grid__item--full-row)")
            
            # Recuperation des images et differentes informations
            for item in items:
                try:
                    # Extraction des donnees : lien image et informations relatives a l'article
                    image_url = item.find_element(By.TAG_NAME, 'img').get_attribute('src')
                    infos = item.text.split("\n")
                    
                    # Gestion de l'index selon la presence d'un numero
                    j = 1 if infos[0].isdigit() else 0
                    
                    # Ajout au dictionnaire
                    clothes[image_url] = {
                        'brand': infos[j],
                        'size': infos[j+1].split(" · ")[0],
                        'price': infos[j+2],
                        'label': label
                    }
                except Exception as e:
                    print(f"Erreur sur un item : {e}")
                    continue

            # Attente aleatoire avant la prochaine page
            time.sleep(random.uniform(1, 3))

    finally:
        driver.quit()
    
    return clothes


# ==============================================
# PARTIE 2 : TELECHARGEMENT DES IMAGES
# ==============================================
def download_images(clothes, save_path):
    # Telecharge les images et conserve leurs metadonnees
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Liste pour stocker les infos complètes
    images_data = []  

    for idx, (img_url, data) in enumerate(clothes.items()):
        img_path = os.path.join(save_path, f"image_{idx}_label{data['label']}.jpg")

        response = requests.get(img_url)
        if response.status_code == 200:
            with open(img_path, "wb") as file:
                file.write(response.content)
            print(f"Image {idx} telechargee : {img_url}")

            # Ajouter l'image avec ses informations à la liste
            images_data.append({
                "image_path": img_path,
                "image_url": img_url,
                "brand": data.get("brand", "Unknown"),
                "size": data.get("size", "Unknown"),
                "price": data.get("price", "Unknown"),
                "label": data.get("label", "Unknown")
            })
        else:
            print(f"echec du telechargement pour {img_url}")

    return images_data

# ==============================================
# PARTIE 3 : PReTRAITEMENT DES IMAGES
# ==============================================

def preprocess_images(images_data, save_path, size=(128, 128)):
    # Pretraite les images (niveaux de gris, redimensionnement, normalisation)
    processed_path = os.path.join(save_path, 'preprocessed')
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)

    # Liste pour stocker les donnees pretraitees
    processed_images = []  

    for idx, img_data in enumerate(images_data):
        img_path = img_data["image_path"]

        try:
            # Charger l'image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Impossible de lire {img_path}")
                continue

            # Conversion en niveaux de gris
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Redimensionner l'image
            resized_img = cv2.resize(gray_img, size)

            # Sauvegarder l'image pretraitee
            processed_img_path = os.path.join(processed_path, os.path.basename(img_path))
            cv2.imwrite(processed_img_path, resized_img)
            print(f"Image pretraitee {idx} sauvegardee : {processed_img_path}")

            # Ajouter les infos de l'image pretraitee à la liste
            processed_images.append({
                "image_path": processed_img_path,
                "image_url": img_data["image_url"],
                "brand": img_data["brand"],
                "size": img_data["size"],
                "price": img_data["price"],
                "label": img_data["label"]
            })
        except Exception as e:
            print(f"Erreur lors du pretraitement de {img_path}: {str(e)}")

    return processed_images


# ==============================================
# PARTIE 4 : ANALYSE EXPLORATOIRE (EDA)
# ==============================================
def perform_eda(processed_images, save_path):
    # Realise une analyse exploratoire des donnees
    eda_dir = os.path.join(save_path, "EDA_Results")
    os.makedirs(eda_dir, exist_ok=True)

    # Conversion en DataFrame
    df = pd.DataFrame(processed_images)
    
    # Nettoyage des prix
    def clean_price(price_str):
        try:
            # Extraction du premier nombre trouve dans la chaîne
            price = ''.join(c for c in price_str if c.isdigit() or c in [',', '.'])
            price = price.replace(',', '.')
            return float(price) if price else None
        except:
            return None
    
    df['price_numeric'] = df['price'].apply(clean_price)
    
    # Suppression des lignes où le prix n'a pas pu être converti
    df = df.dropna(subset=['price_numeric'])
    
    # Mapping des labels
    label_map = {0: "Chaussures", 1: "Pantalon & Short", 2: "Haut"}
    df['category'] = df['label'].map(label_map)

    # Sauvegarde des metadonnees
    df.to_csv(os.path.join(eda_dir, "metadata.csv"), index=False)

    # 1. Distribution des categories
    plt.figure(figsize=(10, 5))
    ax = sns.countplot(data=df, x='category', order=df['category'].value_counts().index)
    plt.title("Distribution des Categories")
    plt.xticks(rotation=45)
    
    # Ajout des pourcentages
    total = len(df)
    for p in ax.patches:
        percentage = f'{100 * p.get_height()/total:.1f}%'
        ax.annotate(percentage, (p.get_x() + p.get_width()/2., p.get_height()),
                   ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    
    plt.savefig(os.path.join(eda_dir, "1_label_distribution.png"))
    plt.close()

    # 2. Distribution des prix
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='category', y='price_numeric', showfliers=False)
    plt.title("Distribution des Prix par Categorie (sans outliers)")
    plt.ylabel("Prix (€)")
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(eda_dir, "2_price_distribution.png"))
    plt.close()

    # 3. Statistiques des images
    stats = []
    for img_path in df['image_path']:
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            stats.append({
                'width': img.shape[1],
                'height': img.shape[0],
                'mean_pixel': img.mean(),
                'std_pixel': img.std()
            })
        except:
            continue
    
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(os.path.join(eda_dir, "image_stats.csv"), index=False)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(stats_df['mean_pixel'], bins=30)
    plt.title("Distribution de la luminosite moyenne")
    
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=stats_df, x='width', y='height')
    plt.title("Dimensions des images")
    plt.savefig(os.path.join(eda_dir, "3_image_stats.png"))
    plt.close()

    print(f"\nEDA termine. Resultats sauvegardes dans: {eda_dir}")

    return df


# ==============================================
# PARTIE 5 : PRePARATION POUR LE MACHINE LEARNING
# ==============================================
def prepare_ml_data(processed_images):
    # Preparation des donnees pour l'entraînement des modèles
    
    # Chargement des images et labels
    X = []
    y = []
    
    for img_data in processed_images:
        try:
            img = cv2.imread(img_data['image_path'], cv2.IMREAD_GRAYSCALE)
            if img is not None:
                X.append(img)
                y.append(img_data['label'])
        except Exception as e:
            print(f"Erreur sur {img_data['image_path']}: {str(e)}")
    
    # Conversion en numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Normalisation
    X = X.astype('float32') / 255.0
    
    # Reshape pour CNN
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    
    # Split des donnees
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42,
        stratify=y
    )
    
    print(f"\nDonnees preparees pour le ML:")
    print(f"- Train set: {X_train.shape[0]} images")
    print(f"- Test set: {X_test.shape[0]} images")
    print(f"- Dimensions des images: {X_train.shape[1:]}")
    
    return X_train, X_test, y_train, y_test

def save_ml_data(X_train, X_test, y_train, y_test, save_path):
    # Sauvegarde les donnees preparees pour le ML
    prep_dir = os.path.join(save_path, "Prepared_Data")
    os.makedirs(prep_dir, exist_ok=True)
    
    # Sauvegarde des donnees numpy
    np.save(os.path.join(prep_dir, "X_train.npy"), X_train)
    np.save(os.path.join(prep_dir, "X_test.npy"), X_test)
    np.save(os.path.join(prep_dir, "y_train.npy"), y_train)
    np.save(os.path.join(prep_dir, "y_test.npy"), y_test)
    
    # Sauvegarde du mapping des labels
    label_map = {0: "Chaussures", 1: "Pantalon & Short", 2: "Haut"}
    with open(os.path.join(prep_dir, "label_mapping.txt"), 'w') as f:
        for key, value in label_map.items():
            f.write(f"{key}:{value}\n")
    
    print(f"\nDonnees ML sauvegardees dans: {prep_dir}")
    print("Contenu du dossier:")
    print(f"├── X_train.npy (shape: {X_train.shape})")
    print(f"├── X_test.npy (shape: {X_test.shape})")
    print(f"├── y_train.npy (shape: {y_train.shape})")
    print(f"├── y_test.npy (shape: {y_test.shape})")
    print(f"└── label_mapping.txt")
    
    return prep_dir


# ==============================================
# FONCTION PRINCIPALE
# ==============================================
def main():
    # Scraping des donnees
    print("\n🕸️ Debut du scraping Vinted...")
    clothes = scrapping_images()
    print(f"{len(clothes)} articles scrapes avec succès")
    
    # Telechargement des images
    save_path = input("\n📂 Dossier de sauvegarde (ex: ./vinted_data) : ").strip()
    os.makedirs(save_path, exist_ok=True)
    
    print("\n⬇️ Telechargement des images...")
    images_data = download_images(clothes, save_path)
    
    # Pretraitement des images
    print("\n🛠️ Pretraitement des images...")
    processed_images = preprocess_images(images_data, save_path)
    
    # Analyse exploratoire (EDA)
    print("\n🔍 Analyse exploratoire des donnees (EDA)...")
    df = perform_eda(processed_images, save_path)
    
    # Preparation pour le Machine Learning
    print("\n📊 Preparation des donnees pour le machine learning...")
    X_train, X_test, y_train, y_test = prepare_ml_data(processed_images)
    
    # Sauvegarde des donnees preparees
    save_ml_data(X_train, X_test, y_train, y_test, save_path)
    
    print("\n🎉 Toutes les etapes ont ete executees avec succès!")
    print("Vous pouvez maintenant passer à l'entraînement des modèles.")



main()
