"""
PROJET TRAITEMENT D'IMAGES - Détection des Maladies des Plantes
Fichier 01 : Prétraitement des images (CR2)
Groupe     : [Hsan Ellouze - Mohamed Kmiha - Amir Mallek - Moncef Koubaa]
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image, ImageEnhance
import warnings
warnings.filterwarnings('ignore')

# CONFIGURATION
IMG_SIZE = (224, 224)
DATA_DIR = "./data"
OUTPUT_DIR = "./results/preprocessing_demo"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# 1. CHARGEMENT ET INSPECTION DES DONNÉES
def inspect_dataset(data_dir):
    """
    Inspecte la structure du dataset PlantVillage
    Retourne les statistiques
    """
    stats = {"classes": [], "counts": {}, "total": 0}
    
    if not os.path.exists(data_dir):
        print(f"[INFO] Dataset non trouvé dans {data_dir}.")
        print("[INFO] Génération d'un dataset synthétique pour démonstration...")
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
    return stats


# 2. GÉNÉRATION D'IMAGES SYNTHÉTIQUES (mode démo)
def generate_synthetic_leaf(diseased=False, disease_type="spot", seed=None):
    """
    Génère une image synthétique de feuille pour démonstration.
    - diseased=False : feuille saine (verte uniforme)
    - diseased=True  : feuille avec taches/jaunissement selon disease_type
    """
    if seed is not None:
        np.random.seed(seed)
    
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    
    # Fond clair
    img[:, :] = [240, 235, 220]
    
    # Forme de la feuille (ellipse verte)
    center = (128, 128)
    axes = (90, 110)
    color = (34, 139, 34) if not diseased else (45, 160, 45)
    cv2.ellipse(img, center, axes, 0, 0, 360, color, -1)
    
    # Nervures
    cv2.line(img, (128, 20), (128, 240), (20, 100, 20), 2)
    for i in range(5):
        y = 60 + i * 35
        cv2.line(img, (128, y), (65, y - 20), (20, 100, 20), 1)
        cv2.line(img, (128, y), (191, y - 20), (20, 100, 20), 1)
    
    if diseased:
        if disease_type == "spot":
            # Taches brunes (Early Blight)
            for _ in range(np.random.randint(5, 15)):
                cx = np.random.randint(50, 200)
                cy = np.random.randint(50, 200)
                # Vérifier que le point est dans la feuille
                if ((cx - 128)**2 / 90**2 + (cy - 128)**2 / 110**2) < 1:
                    r = np.random.randint(5, 20)
                    cv2.circle(img, (cx, cy), r, (80, 50, 20), -1)
                    cv2.circle(img, (cx, cy), r + 2, (160, 120, 40), 1)
        
        elif disease_type == "yellow":
            # Jaunissement (Chlorose / Late Blight)
            mask = np.zeros((256, 256), dtype=np.uint8)
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
            yellow_overlay = np.zeros_like(img)
            yellow_overlay[:, :] = [0, 200, 200]  # BGR: jaune
            yellow_mask = (np.random.rand(256, 256) > 0.5).astype(np.uint8)
            yellow_mask = cv2.GaussianBlur(yellow_mask.astype(np.float32), (21, 21), 0)
            for c in range(3):
                img[:, :, c] = np.where(
                    (mask > 0) & (yellow_mask > 0.3),
                    np.clip(img[:, :, c].astype(int) + yellow_overlay[:, :, c].astype(int) * yellow_mask * 0.4, 0, 255).astype(np.uint8),
                    img[:, :, c]
                )
        
        elif disease_type == "rust":
            # Rouille (Rust)
            for _ in range(np.random.randint(20, 40)):
                cx = np.random.randint(50, 200)
                cy = np.random.randint(50, 200)
                if ((cx - 128)**2 / 90**2 + (cy - 128)**2 / 110**2) < 1:
                    r = np.random.randint(2, 8)
                    cv2.circle(img, (cx, cy), r, (20, 80, 180), -1)
    
    # Légère texture (bruit)
    noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return img


# 3. PIPELINE DE PRÉTRAITEMENT
def preprocess_image(img, target_size=IMG_SIZE, augment=False):
    """
    Pipeline complet de prétraitement d'une image:
    1. Redimensionnement
    2. Normalisation
    3. (Optionnel) Augmentation de données
    
    Paramètres:
        img        : np.array BGR (lecture OpenCV)
        target_size: tuple (H, W)
        augment    : bool - appliquer augmentation
    
    Retourne:
        img_processed : np.array float32 normalisé [0, 1]
    """
    # Redimensionnement
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Conversion BGR → RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Normalisation [0, 255] → [0, 1]
    img_norm = img_rgb.astype(np.float32) / 255.0
    
    # Étape 4 : Augmentation
    if augment:
        img_norm = apply_augmentation(img_norm)
    
    return img_norm


def apply_augmentation(img):
    """
    Applique des transformations aléatoires pour augmenter la diversité:
    - Flip horizontal/vertical
    - Rotation ±30°
    - Variation de luminosité/contraste
    - Zoom léger
    """
    h, w = img.shape[:2]
    
    # Flip horizontal (prob 50%)
    if np.random.rand() > 0.5:
        img = img[:, ::-1, :]
    
    # Flip vertical (prob 30%)
    if np.random.rand() > 0.7:
        img = img[::-1, :, :]
    
    # Rotation aléatoire ±30°
    angle = np.random.uniform(-30, 30)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)
    
    # Variation de luminosité [-0.2, +0.2]
    brightness = np.random.uniform(-0.2, 0.2)
    img = np.clip(img + brightness, 0, 1)
    
    # Variation de contraste [0.8, 1.2]
    contrast = np.random.uniform(0.8, 1.2)
    mean = img.mean()
    img = np.clip((img - mean) * contrast + mean, 0, 1)
    
    return img

# 4. FILTRES D'AMÉLIORATION
def apply_filters_demo(img):
    """
    Démontre plusieurs filtres classiques de traitement d'image.
    Retourne un dictionnaire {nom: image_filtrée}.
    """
    results = {}
    img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    
    # --- Flou gaussien (réduction bruit) ---
    results["Flou Gaussien"] = cv2.GaussianBlur(img_uint8, (5, 5), 0)
    
    # --- Filtre médian (préservation contours) ---
    results["Filtre Médian"] = cv2.medianBlur(img_uint8, 5)
    
    # --- Détection de contours (Canny) ---
    edges = cv2.Canny(gray, 50, 150)
    results["Détection Contours (Canny)"] = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    # --- Égalisation d'histogramme (CLAHE) ---
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    results["CLAHE (Amélioration contraste)"] = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # --- Segmentation par couleur (masque vert) ---
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    results["Segmentation Végétation"] = cv2.bitwise_and(img_uint8, img_uint8, mask=mask)
    
    # --- Sharpening (accentuation) ---
    kernel_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    results["Accentuation (Sharpen)"] = cv2.filter2D(img_uint8, -1, kernel_sharp)
    
    return results


# 5. VISUALISATION AVANT / APRÈS
def plot_preprocessing_demo(save_path=None):
    """
    Génère la figure de démonstration avant/après prétraitement.
    """
    # Générer images de démo
    imgs_raw = {
        "Saine": generate_synthetic_leaf(False, seed=42),
        "Early Blight\n(Taches)": generate_synthetic_leaf(True, "spot", seed=7),
        "Late Blight\n(Jaunissement)": generate_synthetic_leaf(True, "yellow", seed=13),
        "Rust\n(Rouille)": generate_synthetic_leaf(True, "rust", seed=21),
    }
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle("Pipeline de Prétraitement – Avant / Après", fontsize=18, fontweight='bold', y=0.98)
    
    col_titles = ["Original", "Redimensionné (224×224)", "Normalisé [0,1]", "Augmenté"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    
    for row, (label, raw_img) in enumerate(imgs_raw.items()):
        # Col 0: Original
        axes[row, 0].imshow(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB))
        axes[row, 0].set_ylabel(label, fontsize=10, fontweight='bold')
        
        # Col 1: Redimensionné
        resized = cv2.resize(raw_img, IMG_SIZE)
        axes[row, 1].imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        
        # Col 2: Normalisé
        normed = preprocess_image(raw_img, augment=False)
        axes[row, 2].imshow(normed)
        
        # Col 3: Augmenté
        augmented = preprocess_image(raw_img, augment=True)
        axes[row, 3].imshow(np.clip(augmented, 0, 1))
        
        for ax in axes[row]:
            ax.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {save_path}")
    plt.close()


def plot_filters_demo(save_path=None):
    """
    Démontre les filtres d'amélioration sur une feuille malade.
    """
    raw = generate_synthetic_leaf(True, "spot", seed=42)
    rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    filters = apply_filters_demo(rgb)
    
    n = len(filters) + 1
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle("Filtres d'Amélioration – Traitement d'Image", fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    axes[0].imshow(rgb)
    axes[0].set_title("Image Originale", fontweight='bold')
    axes[0].axis('off')
    
    for i, (name, filtered) in enumerate(filters.items(), 1):
        axes[i].imshow(filtered)
        axes[i].set_title(name, fontsize=10)
        axes[i].axis('off')
    
    # Masquer l'axe en surplus
    for j in range(n, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {save_path}")
    plt.close()


def plot_histogram_analysis(save_path=None):
    """
    Analyse les histogrammes de couleur (sain vs malade).
    """
    healthy = generate_synthetic_leaf(False, seed=42)
    diseased = generate_synthetic_leaf(True, "spot", seed=7)
    
    healthy_rgb = cv2.cvtColor(healthy, cv2.COLOR_BGR2RGB)
    diseased_rgb = cv2.cvtColor(diseased, cv2.COLOR_BGR2RGB)
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle("Analyse des Histogrammes de Couleur", fontsize=14, fontweight='bold')
    
    labels = ["Feuille Saine", "Feuille Malade (Early Blight)"]
    images = [healthy_rgb, diseased_rgb]
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

# MAIN
if __name__ == "__main__":
    print("\n")
    print("  PRÉTRAITEMENT DES IMAGES – PlantVillage")
    print("\n")
    
    # Inspection dataset (si disponible)
    stats = inspect_dataset(DATA_DIR)
    
    print("\nGénération figure prétraitement avant/après: ")
    plot_preprocessing_demo(os.path.join(OUTPUT_DIR, "fig1_preprocessing.png"))
    
    print("Génération figure filtres d'amélioration: ")
    plot_filters_demo(os.path.join(OUTPUT_DIR, "fig2_filters.png"))
    
    print("Génération analyse histogrammes: ")
    plot_histogram_analysis(os.path.join(OUTPUT_DIR, "fig3_histograms.png"))
    
    print("\nPrétraitement terminé. Figures sauvegardées dans:", OUTPUT_DIR)
