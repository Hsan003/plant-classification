"""
PROJET TRAITEMENT D'IMAGES - Détection des Maladies des Plantes
Fichier 03 : Évaluation du modèle (Version Stable tf.data)
Corrections : évaluation via tf.data.Dataset (pas de rechargement RAM complet)
Groupe     : [Hsan Ellouze - Mohamed Kmiha - Amir Mallek - Moncef Koubaa]
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, classification_report,
                              precision_recall_fscore_support, accuracy_score)
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras

# ============================================================
# IMPORT CONFIG CENTRALISÉE
# ============================================================
from config_shared import CONFIG, CLASS_NAMES, CLASS_TO_IDX, PATHS, load_class_metadata

NUM_CLASSES = len(CLASS_NAMES)
RESULTS_DIR = PATHS["results"]
MODELS_DIR = PATHS["models"]
DATA_DIR = PATHS["data"]
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 1. CHARGEMENT DES DONNÉES ET PRÉDICTIONS
# ============================================================
def load_test_data():
    """Charge les données de test via le module d'entraînement (pipeline tf.data)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("training", "02_model_training.py")
    training_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training_module)
    dataset = training_module.load_dataset_from_folder(DATA_DIR)
    return dataset


def generate_predictions(model, test_ds):
    """Génère les prédictions du modèle sur le test set (tf.data.Dataset)."""
    y_pred_proba = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    return y_pred, y_pred_proba


def get_true_labels(test_ds):
    """Extrait les vraies labels d'un tf.data.Dataset batché."""
    y_true = []
    for batch in test_ds:
        # batch peut être (images, labels) ou (images, labels, weights)
        if isinstance(batch, tuple):
            labels = batch[1]
        else:
            labels = batch
        y_true.append(labels.numpy())
    return np.concatenate(y_true)


def evaluate_model_real(model_path, dataset):
    """Évalue un modèle entraîné sur l'ensemble test."""
    print(f"[Loading] Modèle: {model_path}")
    model = keras.models.load_model(model_path)

    test_ds = dataset["test_ds"]
    print(f"[Evaluating] Test set: {dataset['num_test']} images")

    y_test = get_true_labels(test_ds)
    y_pred, y_pred_proba = generate_predictions(model, test_ds)
    return y_test, y_pred, y_pred_proba


# ============================================================
# 2. MATRICE DE CONFUSION (toutes les classes)
# ============================================================
def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Trace la matrice de confusion normalisée et brute pour TOUTES les classes.
    """
    n_classes = len(class_names)
    labels = list(range(n_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Éviter division par zéro pour normalisation
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm.astype(float), row_sums,
                        out=np.zeros_like(cm, dtype=float),
                        where=row_sums != 0)

    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    fig.suptitle(f"Matrice de Confusion – {n_classes} classes",
                 fontsize=14, fontweight='bold')

    short_names = [c.replace("___", "\n").replace("__", "\n") for c in class_names]

    for ax, data, title, fmt in zip(
        axes, [cm, cm_norm],
        ["Valeurs Absolues", "Normalisée (par classe)"],
        ['d', '.2f']
    ):
        im = ax.imshow(data, interpolation='nearest',
                       cmap='Blues' if fmt == '.2f' else 'YlOrRd')
        ax.set_title(title, fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(short_names, fontsize=7)
        ax.set_xlabel("Prédit", fontsize=11)
        ax.set_ylabel("Réel", fontsize=11)

        thresh = data.max() / 2 if data.max() > 0 else 1
        for i in range(n_classes):
            for j in range(n_classes):
                val = data[i, j]
                text = f"{val:{fmt}}" if fmt == 'd' else f"{val:.2f}"
                ax.text(j, i, text, ha='center', va='center',
                        fontsize=6, color='white' if val > thresh else 'black')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {save_path}")
    plt.close()
    return cm


# ============================================================
# 3. MÉTRIQUES PAR CLASSE
# ============================================================
def compute_and_plot_metrics(y_true, y_pred, class_names, save_path=None):
    """Calcule et visualise Precision, Recall, F1-Score par classe."""
    n_classes = len(class_names)
    labels = list(range(n_classes))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    accuracy = accuracy_score(y_true, y_pred)

    x = np.arange(n_classes)
    width = 0.25

    short_names = [c.replace("___", "\n").replace("__", "\n") for c in class_names]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 14))
    fig.suptitle(f"Métriques par Classe – Accuracy Globale: {accuracy*100:.1f}%",
                 fontsize=14, fontweight='bold')

    bars1 = ax1.bar(x - width, precision * 100, width, label='Précision', color='#4472C4', alpha=0.85)
    bars2 = ax1.bar(x, recall * 100, width, label='Rappel (Recall)', color='#ED7D31', alpha=0.85)
    bars3 = ax1.bar(x + width, f1 * 100, width, label='F1-Score', color='#70AD47', alpha=0.85)

    ax1.set_xlabel("Classes", fontsize=11)
    ax1.set_ylabel("Score (%)", fontsize=11)
    ax1.set_title("Précision, Rappel et F1-Score par Classe", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
    ax1.set_ylim([0, 105])
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=accuracy * 100, color='red', linestyle='--', alpha=0.5)

    for bar in [bars1, bars2, bars3]:
        for b in bar:
            h = b.get_height()
            if h > 0:
                ax1.text(b.get_x() + b.get_width()/2, h + 0.3, f'{h:.0f}',
                         ha='center', va='bottom', fontsize=5)

    # Tableau de synthèse
    ax2.axis('off')

    macro_prec = np.mean(precision) * 100
    macro_rec = np.mean(recall) * 100
    macro_f1 = np.mean(f1) * 100
    weighted_f1 = np.average(f1, weights=support) * 100 if len(support) > 0 else 0

    summary_data = [
        ["Métrique", "Valeur", "Interprétation"],
        ["Accuracy globale", f"{accuracy*100:.2f}%", "Performance globale"],
        ["Précision (macro)", f"{macro_prec:.2f}%", "Moyenne des précisions"],
        ["Rappel (macro)", f"{macro_rec:.2f}%", "Moyenne des rappels"],
        ["F1-Score (macro)", f"{macro_f1:.2f}%", "Équilibre précision/rappel"],
        ["F1-Score (weighted)", f"{weighted_f1:.2f}%", " pondéré par le support"],
    ]

    table = ax2.table(
        cellText=summary_data[1:],
        colLabels=summary_data[0],
        cellLoc='left', loc='center',
        colWidths=[0.25, 0.15, 0.55]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.0)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#EBF3FB')

    ax2.set_title("Tableau de Synthèse", fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[Saved] {save_path}")
    plt.close()

    return {"accuracy": accuracy, "precision": macro_prec, "recall": macro_rec,
            "f1_macro": macro_f1, "f1_weighted": weighted_f1}


# ============================================================
# 4. ANALYSE DES ERREURS
# ============================================================
def plot_error_analysis(y_true, y_pred, class_names, save_path=None):
    """Analyse les erreurs et biais du modèle."""
    n_classes = len(class_names)
    errors = y_true != y_pred
    error_rate = errors.mean() * 100

    error_by_class = []
    for cls in range(n_classes):
        mask_cls = y_true == cls
        if mask_cls.sum() == 0:
            error_by_class.append(0)
        else:
            err = (y_true[mask_cls] != y_pred[mask_cls]).mean() * 100
            error_by_class.append(err)

    short_names = [c.replace("___", "\n").replace("__", "\n") for c in class_names]

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f"Analyse des Erreurs (Taux global: {error_rate:.1f}%)",
                 fontsize=13, fontweight='bold')

    colors = ['#FF4444' if e > 7 else '#FFA500' if e > 4 else '#4CAF50'
              for e in error_by_class]
    bars = axes[0].bar(range(n_classes), error_by_class, color=colors, alpha=0.85, edgecolor='black')
    axes[0].set_xticks(range(n_classes))
    axes[0].set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
    axes[0].set_ylabel("Taux d'erreur (%)")
    axes[0].set_title("Taux d'erreur par Classe")
    axes[0].axhline(y=error_rate, color='gray', linestyle='--', label=f'Moyenne ({error_rate:.1f}%)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].axis('off')
    critique_text = f"""ANALYSE CRITIQUE DES RÉSULTATS

DATASET RÉEL:
• {n_classes} classes détectées localement
• Sous-ensemble PlantVillage (Pepper, Potato, Tomato)
• Déséquilibre marqué entre classes

POINTS FORTS :
✅ Modèle réel entraîné sur données réelles
✅ Transfer Learning + Fine-tuning
✅ Callbacks (EarlyStopping, ReduceLROnPlateau)
✅ Class weights pour déséquilibre
✅ Pipeline tf.data (pas de saturation RAM)

LIMITES IDENTIFIÉES :
1. Sous-ensemble limité (15/38 classes)
2. Images en conditions contrôlées
3. Pas de test sur données terrain/mobile
4. Déséquilibre des classes (152 à 3208 images)

PISTES D'AMÉLIORATION :
• Augmentation de données plus agressive
• Sur-échantillonnage des classes rares
• Tester sur photos mobiles réelles
• Mixed precision pour accélérer l'entraînement
"""
    axes[1].text(0.02, 0.98, critique_text, transform=axes[1].transAxes,
                 va='top', ha='left', fontsize=9,
                 fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='#F8F9FA', alpha=0.9))

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
    print("  ÉVALUATION DU MODÈLE – PlantVillage")
    print("="*70 + "\n")

    try:
        # Charger les données (pipeline tf.data, pas de RAM complète)
        print("[1] Chargement des données (tf.data)...")
        dataset = load_test_data()

        # Charger le modèle
        model_path = PATHS["model_final"]
        if not os.path.exists(model_path):
            print(f"\n[ERROR] Modèle non trouvé: {model_path}")
            print("[INFO] Entraînez d'abord: python 02_model_training.py")
            exit(1)

        print("[2] Chargement du modèle entraîné...")
        y_true, y_pred, y_pred_proba = evaluate_model_real(model_path, dataset)

        # Utiliser les classes du config (ou recharger depuis le JSON si différent)
        eval_class_names = CLASS_NAMES
        loaded_classes, _ = load_class_metadata(MODELS_DIR)
        if loaded_classes and len(loaded_classes) == NUM_CLASSES:
            eval_class_names = loaded_classes

        print("[3] Génération des visualisations...")

        print("[3a] Matrice de confusion...")
        plot_confusion_matrix(y_true, y_pred, eval_class_names,
                              os.path.join(RESULTS_DIR, "fig6_confusion_matrix.png"))

        print("[3b] Métriques et synthèse...")
        metrics = compute_and_plot_metrics(y_true, y_pred, eval_class_names,
                                           os.path.join(RESULTS_DIR, "fig7_metrics.png"))

        print("[3c] Analyse des erreurs...")
        plot_error_analysis(y_true, y_pred, eval_class_names,
                            os.path.join(RESULTS_DIR, "fig8_error_analysis.png"))

        print("[3d] Rapport de classification...")
        unique_labels = np.unique(y_true)
        report = classification_report(
            y_true,
            y_pred,
            labels=unique_labels,
            target_names=[eval_class_names[i] for i in unique_labels],
            zero_division=0
        )
        report_path = os.path.join(RESULTS_DIR, "classification_report.txt")
        with open(report_path, 'w') as f:
            f.write("RAPPORT DE CLASSIFICATION DÉTAILLÉ\n")
            f.write("="*70 + "\n\n")
            f.write(report)
        print(f"[Saved] {report_path}")

        # Résumé final
        print("\n" + "="*70)
        print("  RÉSULTATS FINAUX")
        print("="*70)
        for k, v in metrics.items():
            print(f"  {k:25s}: {v:.2f}%")
        print()

    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n[SOLUTION] Assurez-vous que:")
        print("  1. Le modèle a été entraîné: python 02_model_training.py")
        print("  2. Les données sont dans ./data/")
        print("  3. TensorFlow et dépendances sont installées")
