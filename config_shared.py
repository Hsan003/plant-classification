"""
Configuration Shared – Détection des Maladies des Plantes
Centralise tous les paramètres et lit dynamiquement les classes du dataset réel.
"""

import os
import json

# ============================================================
# CONFIGURATION GLOBALE
# ============================================================
CONFIG = {
    "img_size": (224, 224),
    "img_channels": 3,
    "batch_size": 64,  # Augmenté pour accélérer (test rapide) - initialement 32
    "epochs_frozen": 2,  # Réduit pour accélérer (test rapide) - initialement 5
    "epochs_finetune": 2,  # Réduit pour accélérer (test rapide) - initialement 10
    "learning_rate": 1e-3,
    "lr_finetune": 1e-5,
    "validation_split": 0.2,
    "test_split": 0.1,
    "model_name": "ResNet50",
    "transfer_learning": True,
    "seed": 42,
}

# ============================================================
# RÉSOLUTION DYNAMIQUE DES CLASSES
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "./data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "./models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "./results")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def resolve_classes(data_dir):
    """
    Scan le dossier data/ pour récupérer les vraies classes présentes localement.
    Retourne la liste triée des noms de dossiers (classes).
    """
    if not os.path.exists(data_dir):
        return []
    classes = []
    for name in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, name)
        if os.path.isdir(path):
            # Vérifier qu'il y a au moins une image
            has_img = any(
                f.lower().endswith((".jpg", ".jpeg", ".png"))
                for f in os.listdir(path)
            )
            if has_img:
                classes.append(name)
    return classes


CLASS_NAMES = resolve_classes(DATA_DIR)
CONFIG["num_classes"] = len(CLASS_NAMES)

if CONFIG["num_classes"] == 0:
    raise RuntimeError(
        "Aucune classe détectée dans data/. Veuillez ajouter le dataset PlantVillage localement."
    )

CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: cls for i, cls in enumerate(CLASS_NAMES)}

# ============================================================
# PATHS
# ============================================================
PATHS = {
    "data": DATA_DIR,
    "models": MODELS_DIR,
    "results": RESULTS_DIR,
    "model_final": os.path.join(MODELS_DIR, "final_model.keras"),
    "model_phase1": os.path.join(MODELS_DIR, "best_model_phase1.keras"),
    "model_phase2": os.path.join(MODELS_DIR, "best_model_phase2.keras"),
    "class_names": os.path.join(MODELS_DIR, "class_names.json"),
    "label_map": os.path.join(MODELS_DIR, "label_map.json"),
    "training_results": os.path.join(MODELS_DIR, "training_results.json"),
}


def save_class_metadata():
    """Sauvegarde les mappings de classes pour une utilisation postérieure."""
    with open(PATHS["class_names"], "w", encoding="utf-8") as f:
        json.dump(CLASS_NAMES, f, indent=2, ensure_ascii=False)
    with open(PATHS["label_map"], "w", encoding="utf-8") as f:
        json.dump(CLASS_TO_IDX, f, indent=2, ensure_ascii=False)


def load_class_metadata(models_dir):
    """Charge les mappings sauvegardés (utile pour l'inférence)."""
    cpath = os.path.join(models_dir, "class_names.json")
    lpath = os.path.join(models_dir, "label_map.json")
    if os.path.exists(cpath) and os.path.exists(lpath):
        with open(cpath, "r", encoding="utf-8") as f:
            classes = json.load(f)
        with open(lpath, "r", encoding="utf-8") as f:
            label_map = json.load(f)
        return classes, label_map
    return None, None


# ============================================================
# INFORMATIONS SUR LES MALADIES (15 classes réelles)
# ============================================================
DISEASE_INFO = {
    "Pepper__bell___Bacterial_spot": {
        "agent": "Xanthomonas campestris pv. vesicatoria (bactérie)",
        "symptomes": "Petites taches circulaires huileuses sur les feuilles, jaunâtres à brunes.",
        "traitement": "Semences certifiées, fongicides cuivriques, rotation des cultures.",
        "urgence": "Modérée",
        "color": "#FF9800",
    },
    "Pepper__bell___healthy": {
        "agent": "—",
        "symptomes": "Feuilles vertes, tiges robustes, croissance normale.",
        "traitement": "Bonnes pratiques agricoles, irrigation régulière.",
        "urgence": "Aucune",
        "color": "#4CAF50",
    },
    "Potato___Early_blight": {
        "agent": "Alternaria solani (champignon)",
        "symptomes": "Taches brunes concentriques sur feuilles âgées, jaunissement.",
        "traitement": "Fongicides protégeants, élimination des débris de récolte.",
        "urgence": "Modérée",
        "color": "#FF9800",
    },
    "Potato___healthy": {
        "agent": "—",
        "symptomes": "Plantes vertes, tubercules sains sans lésions.",
        "traitement": "Rotation, sol drainé, semences certifiées.",
        "urgence": "Aucune",
        "color": "#4CAF50",
    },
    "Potato___Late_blight": {
        "agent": "Phytophthora infestans (oomycète)",
        "symptomes": "Taches aqueuses grises, pourriture rapide, mycélium blanc.",
        "traitement": "Fongicides systémiques urgents, arrachage des plants infectés.",
        "urgence": "Critique",
        "color": "#F44336",
    },
    "Tomato_Bacterial_spot": {
        "agent": "Xanthomonas perforans / X. euvesicatoria",
        "symptomes": "Taches sombres angulaires sur feuilles et fruits.",
        "traitement": "Cuivre + mancozèbe, semences traitées, éviter arrosage feuillage.",
        "urgence": "Modérée",
        "color": "#FF9800",
    },
    "Tomato_Early_blight": {
        "agent": "Alternaria solani",
        "symptomes": "Taches brunes concentriques, feuilles jaunissent du bas vers le haut.",
        "traitement": "Fongicides, mulch, taille des feuilles inférieures.",
        "urgence": "Modérée",
        "color": "#FF9800",
    },
    "Tomato_healthy": {
        "agent": "—",
        "symptomes": "Feuilles vertes, floraison normale, fruits sains.",
        "traitement": "Bonnes pratiques, engrais équilibrés.",
        "urgence": "Aucune",
        "color": "#4CAF50",
    },
    "Tomato_Late_blight": {
        "agent": "Phytophthora infestans",
        "symptomes": "Taches brunes sur feuilles et tiges, pourriture des fruits.",
        "traitement": "Fongicides systémiques, aération, hygiène du sol.",
        "urgence": "Critique",
        "color": "#F44336",
    },
    "Tomato_Leaf_Mold": {
        "agent": "Passalora fulva (champignon)",
        "symptomes": "Taches jaunes sur le dessus, moisissure olive en dessous.",
        "traitement": "Ventilation, fongicides, variétés résistantes.",
        "urgence": "Modérée",
        "color": "#FF9800",
    },
    "Tomato_Septoria_leaf_spot": {
        "agent": "Septoria lycopersici",
        "symptomes": "Petites taches circulaires avec centre gris et bord brun.",
        "traitement": "Suppression feuilles infectées, fongicides, espacement.",
        "urgence": "Modérée",
        "color": "#FF9800",
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "agent": "Tetranychus urticae (acarien)",
        "symptomes": "Jaunissement ponctué, toilettes fines sous les feuilles.",
        "traitement": "Acaricides, humidité ambiante, prédateurs naturels.",
        "urgence": "Modérée",
        "color": "#FF9800",
    },
    "Tomato__Target_Spot": {
        "agent": "Corynespora cassiicola",
        "symptomes": "Taches brunes avec cercles concentriques, défoliation.",
        "traitement": "Fongicides, élimination résidus, variétés résistantes.",
        "urgence": "Modérée",
        "color": "#FF9800",
    },
    "Tomato__Tomato_mosaic_virus": {
        "agent": "Tomato mosaic virus (ToMV)",
        "symptomes": "Mosaïque jaune-vert, feuilles frisées, ralentissement.",
        "traitement": "Aucun traitement curatif. Hygiène, semences saines.",
        "urgence": "Élevée",
        "color": "#F44336",
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "agent": "Tomato yellow leaf curl virus (TYLCV)",
        "symptomes": "Feuilles en cuillère, jaunissement, arrêt de croissance.",
        "traitement": "Lutte anti-mouches blanches, variétés tolérantes.",
        "urgence": "Critique",
        "color": "#F44336",
    },
}

# ============================================================
# VÉRIFICATION DE COHÉRENCE
# ============================================================
def verify_config():
    """Vérifie la cohérence de la configuration."""
    issues = []
    if len(CLASS_NAMES) != CONFIG["num_classes"]:
        issues.append(
            f"Nombre de classes incohérent: {len(CLASS_NAMES)} vs CONFIG['num_classes']={CONFIG['num_classes']}"
        )
    if len(CLASS_NAMES) != len(set(CLASS_NAMES)):
        issues.append("Classes dupliquées trouvées!")
    if not os.path.exists(DATA_DIR):
        issues.append(f"Chemin manquant: {DATA_DIR}")
    return issues


if __name__ == "__main__":
    print("Configuration Shared – Validation")
    print("=" * 60)
    print(f"Classes détectées: {len(CLASS_NAMES)}")
    for i, c in enumerate(CLASS_NAMES):
        print(f"  {i:2d}: {c}")
    print(f"\nConfig: {CONFIG}")
    issues = verify_config()
    if issues:
        print("\n⚠️  Problèmes détectés:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ Configuration valide")
    save_class_metadata()
    print(f"\n💾 Mappings sauvegardés dans: {PATHS['class_names']} et {PATHS['label_map']}")
