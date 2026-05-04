#!/usr/bin/env python
"""
Script de vérification rapide de la cohérence du projet
"""

import os
import sys

def check_files():
    """Vérifie l'existence des fichiers essentiels."""
    files = [
        "01_preprocessing.py",
        "02_model_training.py",
        "03_evaluation.py",
        "04_app_streamlit.py",
        "config_shared.py",
        "README.md",
        "TRANSFORMATION_SUMMARY.py",
    ]
    
    print("\n📂 Vérification des fichiers:")
    all_ok = True
    for f in files:
        exists = os.path.exists(f)
        status = "✅" if exists else "❌"
        print(f"  {status} {f}")
        all_ok = all_ok and exists
    
    return all_ok

def check_config():
    """Vérifie la cohérence de la configuration."""
    try:
        from config_shared import CLASS_NAMES, CONFIG, verify_config
        
        print("\n⚙️  Vérification de la configuration:")
        print(f"  ✅ Classes: {len(CLASS_NAMES)}")
        print(f"  ✅ Num classes CONFIG: {CONFIG['num_classes']}")
        
        if len(CLASS_NAMES) != CONFIG["num_classes"]:
            print(f"  ❌ ERREUR: Incohérence du nombre de classes!")
            return False
        
        issues = verify_config()
        if issues:
            print("  ⚠️  Problèmes détectés:")
            for issue in issues:
                print(f"    - {issue}")
            return False
        
        print("  ✅ Configuration valide")
        return True
    except Exception as e:
        print(f"  ❌ Erreur: {str(e)}")
        return False

def check_imports():
    """Vérifie que les imports essentiels sont disponibles."""
    print("\n📦 Vérification des dépendances:")
    
    imports = {
        "tensorflow": "TensorFlow",
        "keras": "Keras",
        "cv2": "OpenCV",
        "numpy": "NumPy",
        "pandas": "Pandas",
        "sklearn": "Scikit-learn",
        "streamlit": "Streamlit",
        "PIL": "Pillow",
    }
    
    all_ok = True
    for module, name in imports.items():
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} NOT FOUND")
            all_ok = False
    
    return all_ok

def check_directories():
    """Vérifie l'existence des répertoires."""
    dirs = {
        "./data": "Dataset",
        "./models": "Modèles",
        "./results": "Résultats",
    }
    
    print("\n📁 Vérification des répertoires:")
    
    for path, desc in dirs.items():
        exists = os.path.exists(path)
        status = "✅" if exists else "⚠️ (créé automatiquement)"
        print(f"  {status} {path} ({desc})")
    
    return True

def main():
    print("="*70)
    print("  VÉRIFICATION DU PROJET – PlantDoc")
    print("="*70)
    
    results = {
        "Files": check_files(),
        "Config": check_config(),
        "Imports": check_imports(),
        "Directories": check_directories(),
    }
    
    print("\n" + "="*70)
    print("  RÉSUMÉ")
    print("="*70)
    
    all_ok = all(results.values())
    
    for check, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check}")
    
    print()
    
    if all_ok:
        print("✅ Tous les vérifications sont passées!")
        print("\n🚀 Prêt à démarrer:")
        print("  1. Entraîner: python 02_model_training.py")
        print("  2. Évaluer:   python 03_evaluation.py")
        print("  3. Web:       streamlit run 04_app_streamlit.py")
        return 0
    else:
        print("❌ Certaines vérifications ont échoué.")
        print("\n💡 Conseil: installer les dépendances")
        print("  pip install tensorflow keras opencv-python scikit-learn streamlit")
        return 1

if __name__ == "__main__":
    sys.exit(main())
