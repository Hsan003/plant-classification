"""
PROJET TRAITEMENT D'IMAGES - Détection des Maladies des Plantes
Fichier 02 : Modèle CNN + Entraînement (Version Stable OOM/GPU)
Corrections : pipeline tf.data (pas de chargement RAM complet), gestion GPU stricte,
              Functional API (pas de double Input), sample_weights intégrés.
Groupe     : [Hsan Ellouze - Mohamed Kmiha - Amir Mallek - Moncef Koubaa]
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# Réduire le bruit TensorFlow AVANT l'import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import (
    ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
)
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import cv2

# ============================================================
# IMPORT CONFIG CENTRALISÉE
# ============================================================
from config_shared import CONFIG, CLASS_NAMES, CLASS_TO_IDX, PATHS, save_class_metadata

# ============================================================
# GESTION STRICTE DE LA MÉMOIRE GPU (CRITIQUE pour éviter le freeze VS Code)
# ============================================================
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[GPU] {len(gpus)} GPU(s) détecté(s), memory growth activé.")
        print(f"[GPU] TensorFlow n'allouera la VRAM que de manière incrémentale.")
    except RuntimeError as e:
        print(f"[GPU] Erreur configuration: {e}")
else:
    print("[GPU] Aucun GPU détecté, entraînement sur CPU.")

os.makedirs(PATHS["models"], exist_ok=True)
os.makedirs(PATHS["results"], exist_ok=True)
tf.random.set_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

NUM_CLASSES = CONFIG["num_classes"]

# ============================================================
# 1. ARCHITECTURE DU MODÈLE
# ============================================================
def build_model(num_classes):
    """
    Construit le modèle ResNet50 avec transfer learning.
    Utilise la Functional API pour éviter le double Input et les graphes invalides.
    L'augmentation est intégrée dans le modèle (activée uniquement en training).
    """
    base_model = ResNet50(
        weights='imagenet',
        input_shape=(*CONFIG["img_size"], 3),
        include_top=False,
        name='resnet50'
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(*CONFIG["img_size"], 3), name='input')

    # Augmentation de données intégrée (layers Keras 3.x / TF 2.x)
    x = layers.RandomFlip("horizontal", name='rand_flip')(inputs)
    x = layers.RandomRotation(0.2, name='rand_rot')(x)
    x = layers.RandomZoom(0.2, name='rand_zoom')(x)
    x = layers.RandomTranslation(0.2, 0.2, name='rand_trans')(x)

    # Backbone (training=False en phase 1 pour que BN reste figé)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dense(512, activation='relu', name='dense_512')(x)
    x = layers.BatchNormalization(name='bn_512')(x)
    x = layers.Dropout(0.4, name='drop_512')(x)
    x = layers.Dense(256, activation='relu', name='dense_256')(x)
    x = layers.BatchNormalization(name='bn_256')(x)
    x = layers.Dropout(0.3, name='drop_256')(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32', name='predictions')(x)

    model = keras.Model(inputs, outputs, name='plantdoc_resnet50')
    return model, base_model


def unfreeze_backbone(model):
    """Dégèle le backbone ResNet50 pour fine-tuning progressif."""
    base_model = model.get_layer('resnet50')
    base_model.trainable = True
    # Geler les premières couches (fine-tuning progressif)
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    return model


# ============================================================
# 2. PIPELINE DE DONNÉES TF.DATA (PAS DE CHARGEMENT RAM COMPLET)
# ============================================================
def list_dataset_paths(data_dir):
    """
    Liste les chemins d'images et leurs labels ENTIERS sans charger les pixels en RAM.
    Retourne deux listes Python : paths, labels.
    """
    if not os.path.exists(data_dir):
        print(f"[INFO] Dataset non trouvé dans {data_dir}")
        return None, None

    detected_classes = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])

    valid_classes = [c for c in detected_classes if c in CLASS_TO_IDX]
    if len(valid_classes) == 0:
        print("[WARNING] Aucune classe connue trouvée dans data/")
        print(f"  Détectées: {detected_classes}")
        print(f"  Attendues: {CLASS_NAMES}")
        return None, None

    paths, labels = [], []
    for class_name in valid_classes:
        class_path = os.path.join(data_dir, class_name)
        print(f"[Scanning] {class_name}...")
        for img_file in os.listdir(class_path):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            paths.append(os.path.join(class_path, img_file))
            labels.append(CLASS_TO_IDX[class_name])

    if len(paths) == 0:
        print("[WARNING] Aucune image trouvée.")
        return None, None

    return paths, labels


def create_tf_dataset(paths, labels, shuffle=False):
    """
    Crée un tf.data.Dataset à la volée depuis les chemins (images sur disque).
    Aucune image n'est chargée entièrement en RAM : seul le batch courant l'est.
    """
    paths = np.array(paths)
    labels = np.array(labels, dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(paths), 2000), reshuffle_each_iteration=True)

    def load_and_preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, CONFIG["img_size"])
        img = tf.cast(img, tf.float32)
        img = preprocess_input(img)
        return img, label

    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(CONFIG["batch_size"]).prefetch(tf.data.AUTOTUNE)
    return ds


def create_weighted_tf_dataset(paths, labels, class_weight_dict, shuffle=False):
    """
    Dataset d'entraînement incluant les sample_weights directement dans le pipeline.
    C'est la méthode la plus robuste pour gérer les déséquilibres avec tf.data.
    Le dataset yield (image, label, sample_weight).
    """
    paths = np.array(paths)
    labels = np.array(labels, dtype=np.int32)
    sample_weights = np.array([class_weight_dict[l] for l in labels], dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels, sample_weights))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(paths), 2000), reshuffle_each_iteration=True)

    def load_and_preprocess(path, label, weight):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, CONFIG["img_size"])
        img = tf.cast(img, tf.float32)
        img = preprocess_input(img)
        return img, label, weight

    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(CONFIG["batch_size"]).prefetch(tf.data.AUTOTUNE)
    return ds



def load_dataset_from_folder(data_dir, test_split=0.1, val_split=0.2):
    """
    Charge le dataset depuis les dossiers SANS saturer la RAM.
    Retourne des tf.data.Dataset et les métadonnées nécessaires.
    """
    paths, labels = list_dataset_paths(data_dir)

    if paths is None or len(paths) == 0:
        raise FileNotFoundError(
            f"Dataset non trouvé ou vide dans {data_dir}. Veuillez ajouter les images PlantVillage localement."
        )

    # Split stratifié sur les CHEMINS (pas sur les images en mémoire)
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        paths, labels, test_size=test_split, random_state=CONFIG["seed"], stratify=labels
    )
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels,
        test_size=val_split / (1 - test_split),
        random_state=CONFIG["seed"], stratify=train_labels
    )

    # Calcul des poids de classe pour l'entraînement
    class_weight = compute_class_weights(np.array(train_labels))

    # Construction des pipelines tf.data (lecture à la volée depuis le disque)
    train_ds = create_weighted_tf_dataset(train_paths, train_labels, class_weight, shuffle=True)
    val_ds = create_tf_dataset(val_paths, val_labels, shuffle=False)
    test_ds = create_tf_dataset(test_paths, test_labels, shuffle=False)

    print(f"[Dataset] Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
    print(f"[Classes] {len(set(labels))} classes actives sur {NUM_CLASSES} configurées")
    print(f"[Memory]  Chargement à la volée (pas de stockage RAM complet)")

    return {
        "train_ds": train_ds, "val_ds": val_ds, "test_ds": test_ds,
        "y_train": np.array(train_labels),
        "num_train": len(train_paths), "num_val": len(val_paths), "num_test": len(test_paths),
    }


def compute_class_weights(y_train):
    """Calcule les poids de classe pour compenser le déséquilibre."""
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )
    return {int(c): float(w) for c, w in zip(classes, weights)}


# ============================================================
# 3. ENTRAÎNEMENT
# ============================================================
def train_model(model, dataset, phase=1):
    """
    Entraîne le modèle.
    Phase 1: backbone gelé
    Phase 2: fine-tuning
    """
    train_ds = dataset["train_ds"]
    val_ds = dataset["val_ds"]

    lr = CONFIG["learning_rate"] if phase == 1 else CONFIG["lr_finetune"]
    epochs = CONFIG["epochs_frozen"] if phase == 1 else CONFIG["epochs_finetune"]

    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        ModelCheckpoint(
            PATHS[f"model_phase{phase}"],
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

    print(f"\n[Training] Phase {phase} – {'Backbone gelé' if phase == 1 else 'Fine-tuning'}")
    print(f"Learning Rate: {lr}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {CONFIG['batch_size']}")
    print(f"Device: {'GPU' if gpus else 'CPU'}")
    print(f"Loss: sparse_categorical_crossentropy")

    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )
    return history


def plot_training_curves(history, phase):
    """Trace les courbes d'entraînement."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'Phase {phase} – Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'Phase {phase} – Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(PATHS["results"], f"fig5_training_curves_phase{phase}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[Saved] {save_path}")
    plt.close()


def plot_architecture_diagram(save_path=None):
    """Diagramme visuel de l'architecture."""
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
        (11.2, 4.0, 1.2, 0.8, '#70AD47', 'white', f"Dense({NUM_CLASSES})\nSoftmax"),
    ]

    for x, y, w, h, color, tc, label in layers_list:
        rect = plt.Rectangle((x, y - h/2), w, h,
                             facecolor=color, edgecolor='black', linewidth=1.5, zorder=3)
        ax.add_patch(rect)
        ax.text(x + w/2, y, label, ha='center', va='center',
                fontsize=8, fontweight='bold', color=tc, zorder=4)

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
    ax.text(2.4, 2.0, "Phase 1: couches gelées\nPhase 2: fine-tuning progressif",
            ha='center', va='center', fontsize=8, color='darkorange',
            bbox=dict(boxstyle='round', facecolor='#FFF2CC', alpha=0.8))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {save_path}")
    plt.close()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("  CONCEPTION ET ENTRAÎNEMENT DU MODÈLE")
    print("  (Version Stable – OOM/GPU corrigés)")
    print("="*70 + "\n")

    # Architecture diagram
    print("1. Génération du diagramme d'architecture...")
    plot_architecture_diagram(os.path.join(PATHS["results"], "fig4_architecture.png"))

    # Sauvegarder les mappings de classes AVANT entraînement
    save_class_metadata()

    # Charger les données (à la volée, pas en RAM)
    print("\n2. Chargement du dataset (lecture disque)...")
    dataset = load_dataset_from_folder(PATHS["data"])

    # Vérification rapide d'un batch pour s'assurer que le pipeline est sain
    print("\n[Check] Vérification d'un batch d'entraînement...")
    for batch in dataset["train_ds"].take(1):
        if len(batch) == 3:
            imgs, lbls, w = batch
            print(f"  Batch OK : images {imgs.shape}, labels {lbls.shape}, weights {w.shape}")
        else:
            imgs, lbls = batch
            print(f"  Batch OK : images {imgs.shape}, labels {lbls.shape}")

    # Construire le modèle
    print("\n3. Construction du modèle ResNet50 (Functional API)...")
    model, base_model = build_model(NUM_CLASSES)
    model.summary()

    # Phase 1
    print("\n4. Phase 1 – Entraînement avec backbone gelé...")
    history_phase1 = train_model(model, dataset, phase=1)
    plot_training_curves(history_phase1, phase=1)

    # Phase 2
    print("\n5. Phase 2 – Fine-tuning du modèle complet...")
    model = unfreeze_backbone(model)
    history_phase2 = train_model(model, dataset, phase=2)
    plot_training_curves(history_phase2, phase=2)

    # Sauvegarder le modèle final
    model.save(PATHS["model_final"])
    print(f"\n[Saved] Modèle final: {PATHS['model_final']}")

    # Évaluation rapide sur le test set (via tf.data.Dataset)
    print("\n6. Évaluation sur l'ensemble test...")
    test_loss, test_acc = model.evaluate(dataset["test_ds"], verbose=0)
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    # Sauvegarder les résultats
    results_summary = {
        'phases': 2,
        'epochs_phase1': CONFIG['epochs_frozen'],
        'epochs_phase2': CONFIG['epochs_finetune'],
        'test_accuracy': float(test_acc),
        'test_loss': float(test_loss),
        'num_classes': NUM_CLASSES,
        'class_names': CLASS_NAMES,
        'dataset_size': {
            'train': dataset['num_train'],
            'val': dataset['num_val'],
            'test': dataset['num_test'],
        }
    }
    with open(PATHS["training_results"], 'w') as f:
        json.dump(results_summary, f, indent=2)

    print("\n" + "="*70)
    print("  ENTRAÎNEMENT TERMINÉ")
    print("="*70)
