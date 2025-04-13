# ==============================================
# PARTIE 0 : IMPORTATION DES BIBLIOTHEQUES
# ==============================================
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import save_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


# ==============================================
# PARIE 1 : CHARGEMENT DES DONNEES PREPAREES
# ==============================================
def load_data(data_dir):
    # Charge les données préparées
    print(f"\n📂 Chargement des données depuis {data_dir}...")
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    X_test = np.load(os.path.join(data_dir, "X_test.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    y_test = np.load(os.path.join(data_dir, "y_test.npy"))
    
    # Charger le mapping des labels
    label_map = {}
    with open(os.path.join(data_dir, "label_mapping.txt"), 'r') as f:
        for line in f:
            key, val = line.strip().split(':')
            label_map[int(key)] = val
    
    print("Données chargées avec succès")
    print(f"- Train set: {X_train.shape[0]} images")
    print(f"- Test set: {X_test.shape[0]} images")
    print(f"- Dimensions des images: {X_train.shape[1:]}")
    print(f"- Classes: {label_map}")
    
    return X_train, X_test, y_train, y_test, label_map


# ==============================================
# PARTIE 2 : CONTRUCTION DU CNN
# ==============================================
def build_cnn(input_shape=(128, 128, 1), num_classes=3):
    # Construction du modèle CNN
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.3),  # Nouveau dropout après la 1ère couche
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.4),  # Nouveau dropout après la 2ème couche
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.5),  # Nouveau dropout après la 3ème couche
        
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.6),  # Dropout existant augmenté
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()
    return model


# ==============================================
# PARTIE 3 : ENTRAINEMENT DU MODELE
# ==============================================
def train_model(model, X_train, y_train, X_val, y_val, epochs=30, batch_size=32):
    # Entraîne le modèle CNN
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint('best_model.h5', save_best_only=True)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

# ==============================================
# PARTIE 4 : EVALUATION
# ==============================================
def evaluate_model(model, X_test, y_test, label_map):
    # Évalue le modèle sur le test set
    
    # Prédictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Rapport de classification
    print("\nRapport de classification:")
    print(classification_report(y_test, y_pred_classes, target_names=label_map.values()))
    
    # Matrice de confusion
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_map.values(), 
                yticklabels=label_map.values())
    plt.title('Matrice de Confusion')
    plt.ylabel('Vraies étiquettes')
    plt.xlabel('Prédictions')
    plt.show()
    
    # Accuracy finale
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n📈 Accuracy finale sur le test set: {test_acc:.2%}")

def plot_history(history):
    """Affiche les courbes d'apprentissage"""
    plt.figure(figsize=(12, 5))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# ==============================================
# PARTIE 5 : FONCTION PRINCIPALE
# ==============================================
def main():
    # 1. Charger les données
    data_dir = input("Entrez le chemin vers le dossier Prepared_Data: ").strip()
    X_train, X_test, y_train, y_test, label_map = load_data(data_dir)
    
    # 2. Split validation set
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, 
        test_size=0.2, 
        random_state=42
    )
    
    # 3. Construction du modèle
    model = build_cnn(input_shape=X_train.shape[1:], num_classes=len(label_map))
    
    # 4. Entraînement
    history = train_model(model, X_train, y_train, X_val, y_val, epochs=30)
    
    # 5. Visualisation des résultats
    plot_history(history)
    
    # 6. Évaluation
    evaluate_model(model, X_test, y_test, label_map)
    
    # 7. Sauvegarde du modèle final
    save_model(model, "vinted_cnn_model.h5", save_format="h5")
    print("\nModèle final sauvegardé dans 'vinted_cnn_model.h5'")


main()
