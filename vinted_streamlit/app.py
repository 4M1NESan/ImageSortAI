import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import cv2
import os
import sys

# Configuration de l'encodage
sys.stdout.reconfigure(encoding='utf-8')

# Dictionnaire des labels
LABELS = {
    0: "Chaussures", 
    1: "Pantalon & Short", 
    2: "Haut"
}

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

# Chargement du modèle
model_path = "model/vinted_cnn_model.h5"
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
            img = cv2.resize(img, (128, 128))  # Redimensionner (adaptez à la taille attendue par votre modèle)
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
