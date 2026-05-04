"""
PROJET TRAITEMENT D'IMAGES - Détection des Maladies des Plantes
Fichier 02 : Modèle CNN + Entraînement (CR3 & CR4)
Groupe     : [Hsan Ellouze - Mohamed Kmiha - Amir Mallek - Moncef Koubaa]
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
)
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.model_selection import train_test_split
import cv2

# CONFIGURATION DU PROJET
CONFIG = {
    "img_size": (224, 224),
    "batch_size": 32,
    "epochs_frozen": 10,
    "epochs_finetune": 10,
    "learning_rate": 1e-3,
    "lr_finetune": 1e-5,
    "num_classes": 38, 
    "seed": 42,
    "results_dir": "./results",
    "models_dir": "./models",
    "data_dir": "./data",
    "validation_split": 0.2,
    "test_split": 0.1,
}

os.makedirs(CONFIG["results_dir"], exist_ok=True)
os.makedirs(CONFIG["models_dir"], exist_ok=True)
tf.random.set_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

# Noms des classes PlantVillage
CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry___Powdery_mildew", "Cherry___healthy",
    "Corn___Cercospora_leaf_spot", "Corn___Common_rust", "Corn___Northern_Leaf_Blight", "Corn___healthy",
    "Grape___Black_rot", "Grape___Esca", "Grape___Leaf_blight", "Grape___healthy",
    "Orange___Haunglongbing",
    "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper___Bacterial_spot", "Pepper___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


# ============================================================
# 1. ARCHITECTURE DU MODÈLE
# ============================================================
def build_model():
    """
    Construit le modèle ResNet50 avec transfer learning.
    Phase 1: backbone gelé, tête entraînable
    Phase 2: fine-tuning complet
    """
    # Charger ResNet50 pré-entraîné sur ImageNet
    base_model = ResNet50(
        weights='imagenet',
        input_shape=(*CONFIG["img_size"], 3),
        include_top=False
    )
    
    # Geler le backbone pour la Phase 1
    base_model.trainable = False
    # Construire le modèle complet
    model = keras.Sequential([
        layers.Input(shape=(*CONFIG["img_size"], 3)),
        
        # Normalisation des images ImageNet
        

        layers.Lambda(preprocess_input),
        
        # Backbone ResNet50 gelé
        base_model,
        
        # Tête de classification entraînable
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(CONFIG["num_classes"], activation='softmax')
    ])
    
    return model, base_model


def unfreeze_backbone(model):
    """Dégèle le backbone ResNet50 pour fine-tuning."""
    base_model = model.get_layer('resnet50')  # ResNet50 layer
    base_model.trainable = True
    
    # Geler les premières couches (entraîner les derniers blocs)
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    return model


# ============================================================
# 2. CHARGEMENT DES DONNÉES
# ============================================================
def load_image(img_path):
    """Charge et redimensionne une image."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, CONFIG["img_size"])
    return img / 255.0


def load_dataset_from_folder(data_dir, test_split=0.1, val_split=0.2):
    """
    Charge le dataset depuis une structure de dossiers.
    Structure attendue: data/class_name/*.jpg
    """
    X, y, filepaths = [], [], []
    
    if not os.path.exists(data_dir):
        print(f"[INFO] Dataset non trouvé dans {data_dir}")
        print("[INFO] Génération d'un dataset synthétique pour démonstration...")
        return generate_synthetic_dataset()
    
    class_to_idx = {cls: i for i, cls in enumerate(CLASS_NAMES)}
    
    for class_name in CLASS_NAMES:
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        
        print(f"[Loading] {class_name}...")
        for img_file in os.listdir(class_path):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            img_path = os.path.join(class_path, img_file)
            img = load_image(img_path)
            
            if img is not None:
                X.append(img)
                y.append(class_to_idx[class_name])
                filepaths.append(img_path)
    
    if len(X) == 0:
        print("[WARNING] Aucune image trouvée, génération synthétique...")
        return generate_synthetic_dataset()
    
    X = np.array(X)
    y = np.array(y)
    
    # Split train/val/test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_split, random_state=CONFIG["seed"], stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_split/(1-test_split), 
        random_state=CONFIG["seed"], stratify=y_temp
    )
    
    print(f"[Dataset] Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }


def generate_synthetic_dataset(num_per_class=50):
    """Génère un dataset synthétique pour démonstration."""
    print("[INFO] Génération d'un dataset synthétique...")
    
    X, y = [], []
    
    for class_idx in range(CONFIG["num_classes"]):
        for i in range(num_per_class):
            # Image aléatoire avec différents motifs
            img = np.random.rand(224, 224, 3)
            
            # Ajouter des variations selon la classe
            if "healthy" in CLASS_NAMES[class_idx]:
                img = img * 0.7 + 0.3  # Plus clair (feuille saine)
                img[::20, ::20] = np.random.rand(12, 12, 3) * 0.3  # Texture
            else:
                img = img * 0.5 + 0.2  # Plus foncé (malade)
                # Ajouter des taches
                y_pos, x_pos = np.random.randint(0, 200, 2)
                size = np.random.randint(10, 40)
                img[y_pos:y_pos+size, x_pos:x_pos+size] *= 0.5
            
            X.append(img.astype(np.float32))
            y.append(class_idx)
    
    X = np.array(X)
    y = np.array(y)
    
    # Split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=CONFIG["test_split"], random_state=CONFIG["seed"], stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=CONFIG["validation_split"]/(1-CONFIG["test_split"]),
        random_state=CONFIG["seed"], stratify=y_temp
    )
    
    print(f"[Synthetic Dataset] Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }


# ============================================================
# 3. ENTRAÎNEMENT RÉEL
# ============================================================
def train_model(model, dataset, phase=1):
    """
    Entraîne le modèle avec données réelles.
    Phase 1: backbone gelé
    Phase 2: fine-tuning
    """
    X_train = dataset["X_train"]
    y_train = dataset["y_train"]
    X_val = dataset["X_val"]
    y_val = dataset["y_val"]
    
    # Convertir en one-hot
    y_train_oh = keras.utils.to_categorical(y_train, CONFIG["num_classes"])
    y_val_oh = keras.utils.to_categorical(y_val, CONFIG["num_classes"])
    
    # Data augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator()
    
    # Compiler le modèle
    if phase == 1:
        lr = CONFIG["learning_rate"]
        epochs = CONFIG["epochs_frozen"]
    else:
        lr = CONFIG["lr_finetune"]
        epochs = CONFIG["epochs_finetune"]
    
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join(CONFIG["models_dir"], f"best_model_phase{phase}.h5"),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
    ]
    
    # Entraîner
    print(f"\n[Training] Phase {phase} – {'Backbone gelé' if phase == 1 else 'Fine-tuning'}")
    print(f"Learning Rate: {lr}")
    print(f"Epochs: {epochs}\n")
    
    history = model.fit(
        train_datagen.flow(X_train, y_train_oh, batch_size=CONFIG["batch_size"]),
        epochs=epochs,
        validation_data=val_datagen.flow(X_val, y_val_oh, batch_size=CONFIG["batch_size"]),
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def plot_training_curves(history, phase):
    """Trace les courbes d'entraînement."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'Phase {phase} – Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'Phase {phase} – Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(CONFIG["results_dir"], f"fig5_training_curves_phase{phase}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[Saved] {save_path}")
    plt.close()


def plot_architecture_diagram(save_path=None):
    """Diagramme visuel de l'architecture du modèle."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_title("Architecture du Modèle – ResNet50 Transfer Learning",
                 fontsize=15, fontweight='bold', pad=20)
    
    layers_list = [
        (0.3, 4.0, 1.2, 0.8, '#4472C4', 'white', "Input\n224×224×3"),
        (1.8, 4.0, 1.2, 0.8, '#ED7D31', 'white', "ResNet50\nBackbone\n(gelé Phase 1)"),
        (3.8, 4.0, 1.2, 0.6, '#A9D18E', 'black', "GAP\n(2048,)"),
        (5.3, 4.0, 1.2, 0.8, '#4472C4', 'white', "Dense(512)\n+BN+ReLU"),
        (7.0, 4.0, 0.8, 0.5, '#FF6B6B', 'white', "Dropout\n(0.4)"),
        (8.3, 4.0, 1.2, 0.8, '#4472C4', 'white', "Dense(256)\n+BN+ReLU"),
        (9.9, 4.0, 0.8, 0.5, '#FF6B6B', 'white', "Dropout\n(0.3)"),
        (11.2, 4.0, 1.2, 0.8, '#70AD47', 'white', "Dense(38)\nSoftmax"),
    ]
    
    for x, y, w, h, color, tc, label in layers_list:
        rect = plt.Rectangle((x, y - h/2), w, h, 
                               facecolor=color, edgecolor='black', linewidth=1.5, zorder=3)
        ax.add_patch(rect)
        ax.text(x + w/2, y, label, ha='center', va='center',
                fontsize=8, fontweight='bold', color=tc, zorder=4)
    
    # Flèches de connexion
    connections = [(1.5, 1.8), (3.0, 3.8), (4.5, 5.3), (6.5, 7.0),
                   (7.8, 8.3), (9.5, 9.9), (10.7, 11.2)]
    for x1, x2 in connections:
        ax.annotate("", xy=(x2, 4.0), xytext=(x1, 4.0),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    legend_items = [
        mpatches.Patch(facecolor='#4472C4', label='Couche Dense'),
        mpatches.Patch(facecolor='#ED7D31', label='Backbone (ResNet50)'),
        mpatches.Patch(facecolor='#FF6B6B', label='Dropout'),
        mpatches.Patch(facecolor='#70AD47', label='Couche de sortie'),
        mpatches.Patch(facecolor='#A9D18E', label='Global Avg Pooling'),
    ]
    ax.legend(handles=legend_items, loc='lower center', ncol=5, fontsize=9,
              bbox_to_anchor=(0.5, -0.05))
    
    ax.text(2.4, 2.0, "Phase 1: couches gelées\nPhase 2: fine-tuning complet",
            ha='center', va='center', fontsize=8, color='darkorange',
            bbox=dict(boxstyle='round', facecolor='#FFF2CC', alpha=0.8))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {save_path}")
    plt.close()


# MAIN
if __name__ == "__main__":
    print("\n" + "="*70)
    print("  CONCEPTION ET ENTRAÎNEMENT RÉEL DU MODÈLE")
    print("="*70 + "\n")
    
    # Architecture diagram
    print("1. Génération du diagramme d'architecture...")
    plot_architecture_diagram(os.path.join(CONFIG["results_dir"], "fig4_architecture.png"))
    
    # Charger les données
    print("\n2. Chargement du dataset...")
    dataset = load_dataset_from_folder(CONFIG["data_dir"])
    
    # Construire le modèle
    print("\n3. Construction du modèle ResNet50...")
    model, base_model = build_model()
    model.summary()
    
    # Phase 1: entraînement avec backbone gelé
    print("\n4. Phase 1 – Entraînement avec backbone gelé...")
    history_phase1 = train_model(model, dataset, phase=1)
    plot_training_curves(history_phase1, phase=1)
    
    # Phase 2: fine-tuning
    print("\n5. Phase 2 – Fine-tuning du modèle complet...")
    model = unfreeze_backbone(model)
    history_phase2 = train_model(model, dataset, phase=2)
    plot_training_curves(history_phase2, phase=2)
    
    # Sauvegarder le modèle final
    final_model_path = os.path.join(CONFIG["models_dir"], "final_model.h5")
    model.save(final_model_path)
    print(f"\n[Saved] Modèle final: {final_model_path}")
    
    # Évaluation rapide
    print("\n6. Évaluation sur l'ensemble test...")
    X_test = dataset["X_test"]
    y_test = dataset["y_test"]
    y_test_oh = keras.utils.to_categorical(y_test, CONFIG["num_classes"])
    
    test_loss, test_acc, test_prec, test_rec = model.evaluate(X_test, y_test_oh, verbose=0)
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Sauvegarder les résultats
    results_summary = {
        'phases': 2,
        'epochs_phase1': CONFIG['epochs_frozen'],
        'epochs_phase2': CONFIG['epochs_finetune'],
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'num_classes': CONFIG['num_classes'],
        'dataset_size': {
            'train': len(dataset['X_train']),
            'val': len(dataset['X_val']),
            'test': len(dataset['X_test']),
        }
    }
    
    with open(os.path.join(CONFIG["models_dir"], "training_results.json"), 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("\n" + "="*70)
    print("  ENTRAÎNEMENT TERMINÉ")
    print("="*70)
