import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os
import sys
import gdown

# Configuration de l'encodage
sys.stdout.reconfigure(encoding='utf-8')

# Dictionnaire des labels
LABELS = {
    0: "Chaussures", 
    1: "Pantalon & Short", 
    2: "Haut"
}

# Fonction pour télécharger le modèle depuis Google Drive
def download_model(model_path, model_url = 'https://drive.google.com/uc?id=17-s9lmrPNuVAdcNJEAJrH4QEq16K_PiC'):
    # Vérifie si le modèle existe déjà
    if not os.path.exists(model_path):
        st.write("Téléchargement du modèle depuis Google Drive...")
        gdown.download(model_url, model_path, quiet=False)
        st.success(f"Modèle téléchargé avec succès dans '{model_path}'")
    else:
        st.success(f"Le modèle est déjà présent dans '{model_path}'")

# Fonction pour charger le modèle
def load_custom_model(model_path):
    try:
        if not os.path.exists(model_path):
            st.error(f"❌ Fichier introuvable : {os.path.abspath(model_path)}")
            return None
            
        model = load_model(
            model_path,
            compile=False,
            custom_objects={'InputLayer': InputLayer}
        )
        
        st.success(f"""
        ✅ Modèle chargé avec succès !
        - Input shape : {model.input_shape}
        - Nombre de couches : {len(model.layers)}
        """)
        return model
        
    except Exception as e:
        st.error(f"""
        ❌ Erreur de chargement :
        {str(e)}
        
        Solutions :
        1. Vérifiez que le modèle est généré avec TensorFlow 2.12+
        2. Exécutez `pip install tensorflow==2.12.0`
        3. Vérifiez les couches personnalisées
        """)
        return None

# Interface Streamlit
st.title("🔍 Classificateur Vinted - CY Tech")

# URL de Google Drive où ton modèle est hébergé
model_url = 'https://drive.google.com/uc?id=17-s9lmrPNuVAdcNJEAJrH4QEq16K_PiC'
model_path = "model/vinted_cnn_model.h5"

# Vérifier si le dossier "model" existe, sinon le créer
if not os.path.exists("model"):
    os.makedirs("model")

# Télécharger le modèle s'il n'est pas déjà présent
download_model(model_path, model_url)

# Chargement du modèle
model = load_custom_model(model_path)

if model is not None:
    uploaded_file = st.file_uploader(
        "Déposez une image de vêtement...", 
        type=["jpg", "png", "jpeg"],
        help="Image de chaussures, pantalon ou haut"
    )
    
    if uploaded_file is not None:
        try:
            # Prétraitement en niveaux de gris
            img = Image.open(uploaded_file).convert("L")  # Convertir en niveaux de gris
            img = np.array(img)  # Convertir en tableau numpy
            img = Image.fromarray(img).resize((128, 128))  # Redimensionner avec Pillow
            img = np.array(img)  # Reconvertir en array après resize

            img = img.reshape(1, 128, 128, 1)  # Ajouter les dimensions batch et canal
            img = img.astype('float32') / 255.0  # Normalisation

            # Prédiction
            pred = model.predict(img)
            class_id = np.argmax(pred)
            confidence = pred[0][class_id]
            
            # Affichage
            col1, col2 = st.columns(2)
            with col1:
                st.image(img[0].squeeze(), caption="Image traitée", width=200, clamp=True)
            with col2:
                st.metric(
                    label="Prédiction", 
                    value=LABELS[class_id],
                    delta=f"{confidence:.2%} confiance"
                )
            
        except Exception as e:
            st.error(f"""
            Erreur de traitement :
            {str(e)}
            
            Vérifiez que :
            1. L'image est valide
            2. Le modèle attend bien du 128x128 en niveaux de gris
            """)

# Footer
st.markdown("---")
st.caption("Projet Data Mining - CY Tech 2025 | Streamlit + TensorFlow 2.12")
