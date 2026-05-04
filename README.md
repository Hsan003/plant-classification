# 🌿 PlantDoc – Diagnostic de Maladies des Plantes

Détection automatique des maladies des plantes par **intelligence artificielle** et **traitement d'images**.

---

## 📊 Résumé rapide

| Aspect | Détail |
|--------|--------|
| **Dataset** | 20,638 images réelles (PlantVillage) |
| **Classes** | 15 (poivron, pomme de terre, tomate + maladies) |
| **Modèle** | ResNet50 (transfer learning) |
| **Précision** | **90%** sur le test set |
| **Architecture** | Prétraitement → Features extraction → Classification |

---

## 🚀 Installation rapide

```bash
cd "/home/hsan/Documents/Projets/Traitement d'images"
source venv/bin/activate
pip install tensorflow opencv-python scikit-learn streamlit matplotlib numpy
```

Vérifier :
```bash
python verify_project.py
```

---

## 📁 Structure

```
├── 01_preprocessing.py       # Visualisation du pipeline d'images
├── 02_model_training.py      # Entraînement ResNet50 (2 phases)
├── 03_evaluation.py          # Métriques et résultats
├── 04_app_streamlit.py       # Interface web interactive
├── config_shared.py          # Configuration centralisée
├── data/                     # Dataset (15 classes)
├── models/                   # Modèle entraîné + métadonnées
└── results/                  # Figures et rapports
```

---

## ⚡ Utilisation

### 1. Générer les visualisations
```bash
python 01_preprocessing.py
```
Génère 3 figures : preprocessing, filtres, histogrammes

### 2. Entraîner le modèle
```bash
python 02_model_training.py
```
Durée : ~45 min (GPU) | ~4h (CPU)  
Crée : `models/final_model.keras`

### 3. Évaluer
```bash
python 03_evaluation.py
```
Génère : Métriques, matrice de confusion, rapport détaillé

### 4. Lancer l'interface web
```bash
streamlit run 04_app_streamlit.py
```
Accès : http://localhost:8501  
Upload une image de feuille → Diagnostic automatique

---

## 📈 Résultats

### Performance globale
```
Accuracy:  90.0%
Precision: 88%
Recall:    91%
F1-score:  89%
```

### Meilleurs cas (>95%)
- Poivron sain : 99%
- Pomme de terre mildiou précoce : 97%
- Tomate virus TYLCV : 95%

### Plus délicats
- Tomate tache bactérienne : 83% (confusions avec autres)
- Tomate mildiou précoce : 74% (très similaire au tardif)

---

## 🧠 Architecture

### Pipeline
```
Image JPG/PNG
    ↓
Redimensionnement (224×224) + Normalisation ImageNet
    ↓
ResNet50 - Extraction de features (48M params pré-entraînés)
    ↓
Tête de classification (Dense 512 → Dense 256 → Softmax 15)
    ↓
Probabilités pour 15 maladies → Diagnostic final
```

### Pourquoi ResNet50 ?
- **Équilibre** : 25M params vs VGG (138M) ou ViT (300M)
- **Transfer learning** : Pré-entraîné sur ImageNet (1.2M images)
- **Vitesse** : Entraîne en 1-2h vs semaines pour from scratch
- **Précision** : 90% achievable vs 70% without transfer learning

### Entraînement 2 phases

**Phase 1** (Backbone figé)
- Seule la tête s'entraîne
- Learning rate : 1e-3
- Durée : 2 epochs (~20 min)
- Résultat : val_accuracy = 90%

**Phase 2** (Fine-tuning)
- 30 dernières couches ResNet50 se dégelent
- Learning rate : 1e-5 (100× plus bas)
- Durée : 2 epochs (~20 min)
- Résultat : val_accuracy = 91%

**Justification :** 
- Phase 1 : adaptation rapide à nos classes
- Phase 2 : affinage fin sans oublier ImageNet (catastrophic forgetting)

---

## 🖼️ Traitement d'images

### Pipeline principal

1. **Redimensionnement** → 224×224 (standard ResNet50)
2. **Normalisation ImageNet** → (img - mean) / std
   - ⚠️ **Obligatoire** : sans cela, accuracy = 50%
3. **Augmentation de données** (entraînement uniquement)
   - Flip horizontal, rotation ±20°, zoom ±20%, translation
   - **Effet** : +15% accuracy (75% → 90%)
4. **Class weighting** (compensation déséquilibre)
   - Classes rares → poids 6.3× | Classes fréquentes → poids 0.29×

### Gestion mémoire

**Défi :** 20k images × 224² × 3 = 15 GB

**Solution : tf.data pipeline**
- Charge seulement 64 images à la fois (1 batch)
- Lit depuis disque à la volée
- Mémoire constante : ~2-3 GB max
- Aucune OOM error

---

## 📚 Modules

### `config_shared.py`
- Détection automatique des 15 classes
- Hyperparamètres centralisés
- Base de données des maladies

### `01_preprocessing.py`
- Inspection dataset
- Visualisation du pipeline d'images
- Génère 3 figures de démo

### `02_model_training.py`
- Construit ResNet50 + tête classification
- Crée dataset tf.data
- Entraîne 2 phases
- Sauvegarde modèle final

### `03_evaluation.py`
- Charge modèle + test set
- Génère prédictions
- Calcule métriques (Accuracy, Precision, Recall, F1)
- Génère matrice de confusion et graphiques

### `04_app_streamlit.py`
- Upload image JPG/PNG
- Diagnostic en temps réel
- Affiche classe, confiance, top 5
- Heatmap des zones critiques
- Recommandations de traitement

---

## 🎯 Choix techniques

### Transfer learning vs from scratch
✅ **Transfer learning (choisi)**
- 20k images suffisent (vs 500k+)
- 1-2h entraînement (vs 2-3 semaines)
- 90% accuracy (vs 70%)

### ResNet50 vs autres architectures
✅ **ResNet50 (choisi)**
- Équilibre optimal précision/vitesse/taille
- Standard depuis 2015, très documenté
- VGG : trop lourd (138M), AlexNet : obsolète, ViT : overkill

### Normalisation ImageNet vs MinMax
✅ **ImageNet (obligatoire)**
- Modèles pré-entraînés l'exigent
- MinMax : -10-15% accuracy

### Data augmentation : dans ou hors du modèle
✅ **Dans le modèle (moderne)**
- GPU-accélérée
- Augmentation différente chaque epoch
- Pas de dupliquage disque

### Class weighting vs Oversampling
✅ **Class weighting (mathématique)**
- Pas duplication données
- GPU-friendly
- Évite overfitting

---

## ⚠️ Limitations

1. **Confusion Early/Late blight** (Recall: 74%)
   - Cause : Symptômes visuellement très similaires
   - Solution : Ajouter 400+ images

2. **Classes rares peu représentées**
   - Potato_healthy : 15 images (min)
   - Solution : Augmenter à 200+ par classe

3. **Pas de détection d'objets**
   - Modèle classe uniquement (pas de localisation)
   - Amélioration : YOLOv8 pour localiser les symptômes

---



## 📚 Technos utilisées

- **TensorFlow/Keras** : Deep learning
- **OpenCV** : Traitement d'images
- **Scikit-learn** : Métriques, preprocessing
- **Streamlit** : Interface web
- **Matplotlib** : Visualisations
- **NumPy** : Calculs numériques

---

## 👥 Auteurs

- Hsan Ellouze
- Mohamed Kmiha
- Amir Mallek
- Moncef Koubaa

**Année :** 2025-2026

---

## 📝 Licence

Projet pédagogique. Dataset PlantVillage : licence publique (citation requise).
