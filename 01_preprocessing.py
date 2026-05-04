"""
PROJET TRAITEMENT D'IMAGES - Détection des Maladies des Plantes
Fichier 01 : Prétraitement des images
Groupe     : [Hsan Ellouze - Mohamed Kmiha - Amir Mallek - Moncef Koubaa]
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

from config_shared import CONFIG, CLASS_NAMES, PATHS

# CONFIGURATION
IMG_SIZE = CONFIG["img_size"]
DATA_DIR = PATHS["data"]
OUTPUT_DIR = os.path.join(PATHS["results"], "preprocessing_demo")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# 1. CHARGEMENT ET INSPECTION DES DONNÉES
def inspect_dataset(data_dir):
    """
    Inspecte la structure du dataset PlantVillage.
    Retourne les statistiques par classe.
    """
    stats = {"classes": [], "counts": {}, "total": 0}

    if not os.path.exists(data_dir):
        print(f"[INFO] Dataset non trouvé dans {data_dir}.")
        return None

    for class_name in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            images = [f for f in os.listdir(class_path)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            stats["classes"].append(class_name)
            stats["counts"][class_name] = len(images)
            stats["total"] += len(images)

    print(f"[Dataset] Classes trouvées : {len(stats['classes'])}")
    print(f"[Dataset] Images totales   : {stats['total']}")
    for c, n in stats["counts"].items():
        print(f"  {n:5d}  {c}")
    return stats


def load_sample_images(data_dir, max_per_class=2):
    """Charge quelques images réelles du dataset pour la démo."""
    samples = {}
    if not os.path.exists(data_dir):
        return samples
    for class_name in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        imgs = [f for f in os.listdir(class_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if imgs:
            selected = imgs[:max_per_class]
            samples[class_name] = [
                cv2.imread(os.path.join(class_path, f))
                for f in selected
                if cv2.imread(os.path.join(class_path, f)) is not None
            ]
    return samples



# 2. PRÉTRAITEMENT DES IMAGES (données réelles uniquement)


# 3. PIPELINE DE PRÉTRAITEMENT (pour visualisation / démo)
def preprocess_image(img, target_size=IMG_SIZE, augment=False):
    """
    Pipeline complet de prétraitement d'une image.
    Ici pour démo/visualisation: normalise [0,1].
    L'entraînement utilise preprocess_input de ResNet50 directement.
    """
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    if augment:
        img_norm = apply_augmentation(img_norm)
    return img_norm


def apply_augmentation(img):
    """Applique des transformations aléatoires."""
    h, w = img.shape[:2]
    if np.random.rand() > 0.5:
        img = img[:, ::-1, :]
    if np.random.rand() > 0.7:
        img = img[::-1, :, :]
    angle = np.random.uniform(-30, 30)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)
    brightness = np.random.uniform(-0.2, 0.2)
    img = np.clip(img + brightness, 0, 1)
    contrast = np.random.uniform(0.8, 1.2)
    mean = img.mean()
    img = np.clip((img - mean) * contrast + mean, 0, 1)
    return img


# 4. FILTRES D'AMÉLIORATION
def apply_filters_demo(img):
    """Démontre plusieurs filtres classiques."""
    results = {}
    img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)

    results["Flou Gaussien"] = cv2.GaussianBlur(img_uint8, (5, 5), 0)
    results["Filtre Médian"] = cv2.medianBlur(img_uint8, 5)
    edges = cv2.Canny(gray, 50, 150)
    results["Détection Contours (Canny)"] = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    results["CLAHE"] = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    results["Segmentation Végétation"] = cv2.bitwise_and(img_uint8, img_uint8, mask=mask)
    kernel_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    results["Accentuation"] = cv2.filter2D(img_uint8, -1, kernel_sharp)
    return results


# 5. VISUALISATIONS
def plot_preprocessing_demo_real(samples, save_path=None):
    """
    Génère la figure de démonstration avec des images RÉELLES du dataset.
    """
    if not samples:
        print("[ERROR] Pas d'images réelles disponibles pour la démonstration de prétraitement.")
        return

    selected_classes = list(samples.keys())[:4]
    fig, axes = plt.subplots(len(selected_classes), 4, figsize=(16, 4 * len(selected_classes)))
    if len(selected_classes) == 1:
        axes = np.expand_dims(axes, axis=0)
    fig.suptitle("Pipeline de Prétraitement – Images Réelles", fontsize=18, fontweight='bold', y=0.98)

    col_titles = ["Original", "Redimensionné (224×224)", "Normalisé [0,1]", "Augmenté"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

    for row, class_name in enumerate(selected_classes):
        imgs = samples[class_name]
        if not imgs:
            continue
        raw = imgs[0]
        label = class_name.replace("___", "\n").replace("__", "\n")

        axes[row, 0].imshow(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
        axes[row, 0].set_ylabel(label, fontsize=9, fontweight='bold')

        resized = cv2.resize(raw, IMG_SIZE)
        axes[row, 1].imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

        normed = preprocess_image(raw, augment=False)
        axes[row, 2].imshow(normed)

        augmented = preprocess_image(raw, augment=True)
        axes[row, 3].imshow(np.clip(augmented, 0, 1))

        for ax in axes[row]:
            ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {save_path}")
    plt.close()


def plot_filters_demo(save_path=None):
    """Démontre les filtres d'amélioration sur une image réelle du dataset."""
    samples = load_sample_images(DATA_DIR, max_per_class=1)
    raw = None
    for imgs in samples.values():
        if imgs:
            raw = imgs[0]
            break
    if raw is None:
        print("[ERROR] Pas d'image réelle disponible pour démontrer les filtres.")
        return
    rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    filters = apply_filters_demo(rgb)

    n = len(filters) + 1
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle("Filtres d'Amélioration", fontsize=16, fontweight='bold')
    axes = axes.flatten()

    axes[0].imshow(rgb)
    axes[0].set_title("Image Originale", fontweight='bold')
    axes[0].axis('off')

    for i, (name, filtered) in enumerate(filters.items(), 1):
        axes[i].imshow(filtered)
        axes[i].set_title(name, fontsize=10)
        axes[i].axis('off')

    for j in range(n, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {save_path}")
    plt.close()


def plot_histogram_analysis_real(samples, save_path=None):
    """Analyse les histogrammes de couleur sur des images réelles."""
    healthy_imgs = []
    diseased_imgs = []
    for cls, imgs in samples.items():
        if not imgs:
            continue
        img_rgb = cv2.cvtColor(imgs[0], cv2.COLOR_BGR2RGB)
        if "healthy" in cls.lower():
            healthy_imgs.append(img_rgb)
        else:
            diseased_imgs.append(img_rgb)

    if not healthy_imgs or not diseased_imgs:
        print("[ERROR] Pas assez d'images réelles pour l'analyse des histogrammes.")
        return

    healthy = healthy_imgs[0]
    diseased = diseased_imgs[0]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle("Analyse des Histogrammes de Couleur (Données Réelles)", fontsize=14, fontweight='bold')

    labels = ["Feuille Saine", "Feuille Malade"]
    images = [healthy, diseased]
    colors_ch = ['red', 'green', 'blue']
    ch_names = ['Rouge', 'Vert', 'Bleu']

    for row, (label, img) in enumerate(zip(labels, images)):
        axes[row, 0].imshow(img)
        axes[row, 0].set_title(label, fontweight='bold')
        axes[row, 0].axis('off')

        for col, (ch, color, name) in enumerate(zip(range(3), colors_ch, ch_names)):
            hist = cv2.calcHist([img], [ch], None, [256], [0, 256])
            axes[row, col + 1].plot(hist, color=color)
            axes[row, col + 1].fill_between(range(256), hist.flatten(), alpha=0.3, color=color)
            axes[row, col + 1].set_title(f"Canal {name}")
            axes[row, col + 1].set_xlim([0, 256])
            axes[row, col + 1].set_xlabel("Intensité")
            axes[row, col + 1].set_ylabel("Fréquence")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {save_path}")
    plt.close()


    # L'analyse des histogrammes est réalisée uniquement sur des images réelles.


# MAIN
if __name__ == "__main__":
    print("\n")
    print("  PRÉTRAITEMENT DES IMAGES – PlantVillage")
    print("\n")

    stats = inspect_dataset(DATA_DIR)

    print("\nGénération figure prétraitement avant/après: ")
    samples = load_sample_images(DATA_DIR, max_per_class=1)
    plot_preprocessing_demo_real(
        samples,
        os.path.join(OUTPUT_DIR, "fig1_preprocessing.png")
    )

    print("Génération figure filtres d'amélioration: ")
    plot_filters_demo(os.path.join(OUTPUT_DIR, "fig2_filters.png"))

    print("Génération analyse histogrammes: ")
    plot_histogram_analysis_real(
        samples,
        os.path.join(OUTPUT_DIR, "fig3_histograms.png")
    )

    print("\nPrétraitement terminé. Figures sauvegardées dans:", OUTPUT_DIR)
