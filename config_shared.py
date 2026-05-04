"""
Configuration Shared – Détection des Maladies des Plantes
Centralise tous les paramètres pour assurer la cohérence entre les modules
"""

import os

# ============================================================
# CONFIGURATION GLOBALE
# ============================================================
CONFIG = {
    # Chemins
    "data_dir": "./data",
    "models_dir": "./models",
    "results_dir": "./results",
    
    # Images
    "img_size": (224, 224),
    "img_channels": 3,
    
    # Training
    "batch_size": 32,
    "epochs_frozen": 10,
    "epochs_finetune": 10,
    "learning_rate": 1e-3,
    "lr_finetune": 1e-5,
    "validation_split": 0.2,
    "test_split": 0.1,
    
    # Model
    "num_classes": 38,
    "model_name": "ResNet50",
    "transfer_learning": True,
    
    # Seed
    "seed": 42,
}

# ============================================================
# CLASSES PLANTVILLAGE (38 classes)
# ============================================================
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

assert len(CLASS_NAMES) == CONFIG["num_classes"], f"Nombre de classes incohérent: {len(CLASS_NAMES)} vs {CONFIG['num_classes']}"

# ============================================================
# INFORMATIONS SUR LES MALADIES
# ============================================================
DISEASE_INFO = {
    "Apple___Apple_scab": {
        "agent": "Venturia inaequalis (champignon)",
        "symptomes": "Taches olive à brunes sur feuilles et fruits. Déformation des fruits.",
        "traitement": "Fongicides préventifs. Taille des branches infectées.",
        "urgence": "Modérée",
        "color": "#FF9800",
    },
    "Apple___Black_rot": {
        "agent": "Botryosphaeria obtusa (champignon)",
        "symptomes": "Taches noires circulaires sur fruits. Chancres sur branches.",
        "traitement": "Suppression des parties affectées. Fongicides de contact.",
        "urgence": "Modérée",
        "color": "#FF9800",
    },
    "Apple___healthy": {
        "agent": "—",
        "symptomes": "Aucun symptôme visible. Feuilles vertes et saines.",
        "traitement": "Continuer les bonnes pratiques agricoles.",
        "urgence": "Aucune",
        "color": "#4CAF50",
    },
    "Tomato___Early_blight": {
        "agent": "Alternaria solani (champignon)",
        "symptomes": "Taches brunes concentriques sur feuilles âgées.",
        "traitement": "Fongicides mancozèbe. Rotation des cultures.",
        "urgence": "Modérée",
        "color": "#FF9800",
    },
    "Tomato___Late_blight": {
        "agent": "Phytophthora infestans (oomycète)",
        "symptomes": "Taches vertes-grises, puis brunes-noires.",
        "traitement": "Fongicides systémiques urgents.",
        "urgence": "Critique",
        "color": "#F44336",
    },
    "Tomato___healthy": {
        "agent": "—",
        "symptomes": "Aucun symptôme visible. Feuilles vertes et saines.",
        "traitement": "Continuer les bonnes pratiques agricoles.",
        "urgence": "Aucune",
        "color": "#4CAF50",
    },
    "Potato___Early_blight": {
        "agent": "Alternaria solani (champignon)",
        "symptomes": "Taches brunes concentriques.",
        "traitement": "Fongicides protégeants.",
        "urgence": "Modérée",
        "color": "#FF9800",
    },
    "Potato___Late_blight": {
        "agent": "Phytophthora infestans (oomycète)",
        "symptomes": "Taches aqueuses grises.",
        "traitement": "Fongicides systémiques urgents.",
        "urgence": "Critique",
        "color": "#F44336",
    },
    "Potato___healthy": {
        "agent": "—",
        "symptomes": "Aucun symptôme visible. Feuilles vertes et saines.",
        "traitement": "Continuer les bonnes pratiques agricoles.",
        "urgence": "Aucune",
        "color": "#4CAF50",
    },
    "Corn___Common_rust": {
        "agent": "Puccinia sorghi (champignon rouille)",
        "symptomes": "Pustules ovales brun-rougeâtre.",
        "traitement": "Variétés résistantes.",
        "urgence": "Modérée",
        "color": "#FF9800",
    },
    "Corn___healthy": {
        "agent": "—",
        "symptomes": "Aucun symptôme visible. Feuilles vertes et saines.",
        "traitement": "Continuer les bonnes pratiques agricoles.",
        "urgence": "Aucune",
        "color": "#4CAF50",
    },
}

# ============================================================
# PATHS ABSOLUS
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

PATHS = {
    "data": os.path.join(PROJECT_ROOT, CONFIG["data_dir"]),
    "models": os.path.join(PROJECT_ROOT, CONFIG["models_dir"]),
    "results": os.path.join(PROJECT_ROOT, CONFIG["results_dir"]),
    "model_final": os.path.join(PROJECT_ROOT, CONFIG["models_dir"], "final_model.h5"),
    "model_phase1": os.path.join(PROJECT_ROOT, CONFIG["models_dir"], "best_model_phase1.h5"),
    "model_phase2": os.path.join(PROJECT_ROOT, CONFIG["models_dir"], "best_model_phase2.h5"),
}

# Créer les répertoires
for path in [PATHS["data"], PATHS["models"], PATHS["results"]]:
    os.makedirs(path, exist_ok=True)

# ============================================================
# VÉRIFICATION DE COHÉRENCE
# ============================================================
def verify_config():
    """Vérifie la cohérence de la configuration."""
    issues = []
    
    # Vérifier nombre de classes
    if len(CLASS_NAMES) != CONFIG["num_classes"]:
        issues.append(f"Nombre de classes incohérent: {len(CLASS_NAMES)} vs CONFIG['num_classes']={CONFIG['num_classes']}")
    
    # Vérifier les doublons
    if len(CLASS_NAMES) != len(set(CLASS_NAMES)):
        issues.append("Classes dupliquées trouvées!")
    
    # Vérifier les chemins
    for name, path in PATHS.items():
        if name.startswith("model_"):
            continue  # Les modèles ne doivent pas exister au démarrage
        if not os.path.exists(path):
            issues.append(f"Chemin manquant: {path}")
    
    return issues


if __name__ == "__main__":
    print("Configuration Shared – Validation")
    print("=" * 60)
    print(f"Classes: {len(CLASS_NAMES)}")
    print(f"Config: {CONFIG}")
    print(f"Chemins: {PATHS}")
    
    issues = verify_config()
    if issues:
        print("\n⚠️  Problèmes détectés:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ Configuration valide")
