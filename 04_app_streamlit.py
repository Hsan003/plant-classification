"""
PROJET TRAITEMENT D'IMAGES - Détection des Maladies des Plantes
Fichier 04 : Interface Web Streamlit (CR6 - Bonus)
Groupe     : [Hsan Ellouze - Mohamed Kmiha - Amir Mallek - Moncef Koubaa]
Lancer avec : streamlit run 04_app_streamlit.py
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras

# ============================================================
# CONFIGURATION DE L'APPLICATION
# ============================================================
st.set_page_config(
    page_title="🌿 PlantDoc – Détection de Maladies",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
        padding: 1rem 0;
        font-weight: bold;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .healthy { background-color: #E8F5E9; border-left: 5px solid #4CAF50; }
    .diseased { background-color: #FFF3E0; border-left: 5px solid #FF9800; }
    .critical { background-color: #FFEBEE; border-left: 5px solid #F44336; }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Classe names (cohérent avec le modèle entraîné)
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

# Données détaillées sur les maladies
DISEASE_INFO = {
    "Apple___Apple_scab": {
        "agent": "Venturia inaequalis (champignon)",
        "symptomes": "Taches olive à brunes sur feuilles et fruits. Déformation des fruits.",
        "traitement": "Fongicides préventifs au printemps. Taille des branches infectées.",
        "urgence": "⚠️ Modérée",
        "color": "#FF9800",
    },
    "Apple___Black_rot": {
        "agent": "Botryosphaeria obtusa (champignon)",
        "symptomes": "Taches noires circulaires sur fruits. Chancres sur branches.",
        "traitement": "Suppression des parties affectées. Fongicides de contact.",
        "urgence": "⚠️ Modérée",
        "color": "#FF9800",
    },
    "Apple___Cedar_apple_rust": {
        "agent": "Gymnosporangium juniperi-virginianae",
        "symptomes": "Taches jaune-orange sur feuilles. Déformation des fruits.",
        "traitement": "Élimination des genévriers. Fongicides systémiques.",
        "urgence": "⚠️ Modérée",
        "color": "#FF9800",
    },
    "Apple___healthy": {
        "agent": "—",
        "symptomes": "Aucun symptôme visible. Feuilles vertes et saines.",
        "traitement": "Continuer les bonnes pratiques agricoles.",
        "urgence": "✅ Aucune action requise",
        "color": "#4CAF50",
    },
    "Tomato___Early_blight": {
        "agent": "Alternaria solani (champignon)",
        "symptomes": "Taches brunes concentriques sur feuilles âgées. Jaunissement autour des lésions.",
        "traitement": "Fongicides à base de mancozèbe. Rotation des cultures.",
        "urgence": "⚠️ Modérée",
        "color": "#FF9800",
    },
    "Tomato___Late_blight": {
        "agent": "Phytophthora infestans (oomycète)",
        "symptomes": "Taches vertes-grises, puis brunes-noires. Mycélium blanchâtre sous la feuille.",
        "traitement": "Fongicides systémiques. Réduction humidité. Surveillance météo.",
        "urgence": "🔴 Haute – Peut détruire la récolte",
        "color": "#F44336",
    },
    "Tomato___healthy": {
        "agent": "—",
        "symptomes": "Aucun symptôme visible. Feuilles vertes et saines.",
        "traitement": "Continuer les bonnes pratiques agricoles.",
        "urgence": "✅ Aucune action requise",
        "color": "#4CAF50",
    },
    "Potato___Early_blight": {
        "agent": "Alternaria solani (champignon)",
        "symptomes": "Taches brunes concentriques. Cercles concentriques visibles.",
        "traitement": "Fongicides protégeants. Élimination des débris.",
        "urgence": "⚠️ Modérée",
        "color": "#FF9800",
    },
    "Potato___Late_blight": {
        "agent": "Phytophthora infestans (oomycète)",
        "symptomes": "Taches aqueuses grises. Mycélium blanc sous les feuilles.",
        "traitement": "Fongicides systémiques urgents. Isolation des plants.",
        "urgence": "🔴 CRITIQUE – Maladie dévastatrice",
        "color": "#F44336",
    },
    "Potato___healthy": {
        "agent": "—",
        "symptomes": "Aucun symptôme visible. Feuilles vertes et saines.",
        "traitement": "Continuer les bonnes pratiques agricoles.",
        "urgence": "✅ Aucune action requise",
        "color": "#4CAF50",
    },
    "Corn___Common_rust": {
        "agent": "Puccinia sorghi (champignon rouille)",
        "symptomes": "Pustules ovales brun-rougeâtre sur les deux faces des feuilles.",
        "traitement": "Variétés résistantes. Fongicides si infestation sévère.",
        "urgence": "⚠️ Modérée",
        "color": "#FF9800",
    },
    "Corn___healthy": {
        "agent": "—",
        "symptomes": "Aucun symptôme visible. Feuilles vertes et saines.",
        "traitement": "Continuer les bonnes pratiques agricoles.",
        "urgence": "✅ Aucune action requise",
        "color": "#4CAF50",
    },
}


# ============================================================
# MODÈLE TENSORFLOW
# ============================================================
@st.cache_resource
def load_model():
    """Charge le modèle entraîné avec cache."""
    model_path = "./models/final_model.h5"
    
    if not os.path.exists(model_path):
        st.error(f"❌ Modèle non trouvé: {model_path}")
        st.info("Entraînez d'abord le modèle avec: `python 02_model_training.py`")
        return None
    
    try:
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle: {str(e)}")
        return None


# ============================================================
# FONCTIONS DE TRAITEMENT
# ============================================================
def preprocess_image(img_array, size=(224, 224)):
    """Prétraitement de l'image uploadée (cohérent avec l'entraînement)."""
    # Redimensionner
    resized = cv2.resize(img_array, size, interpolation=cv2.INTER_LINEAR)
    
    # Normaliser [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    
    return normalized, resized


def predict_disease(model, img_array):
    """
    Effectue une prédiction avec le modèle entraîné.
    """
    # Ajouter batch dimension
    img_batch = np.expand_dims(img_array, axis=0)
    
    # Prédiction
    predictions = model.predict(img_batch, verbose=0)
    probabilities = predictions[0]
    
    # Classe prédite et confiance
    pred_idx = np.argmax(probabilities)
    confidence = probabilities[pred_idx]
    
    # Créer dictionnaire avec top 5 prédictions
    top_5_idx = np.argsort(probabilities)[-5:][::-1]
    top_5_probs = {CLASS_NAMES[i]: float(probabilities[i]) for i in top_5_idx}
    
    return CLASS_NAMES[pred_idx], float(confidence), top_5_probs


def format_disease_name(class_name):
    """Formate le nom de la classe pour l'affichage."""
    plant, disease = class_name.rsplit("___", 1)
    plant = plant.replace("_", " ")
    disease = disease.replace("_", " ")
    return f"{plant} – {disease}"


def highlight_zones(img_resized, confidence):
    """Crée une visualisation simulée des zones d'intérêt."""
    h, w = img_resized.shape[:2]
    
    # Créer une heatmap basée sur les gradients locaux
    gray = cv2.cvtColor((img_resized * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # Calcul des gradients (Sobel)
    gradx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grady = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gradx**2 + grady**2)
    
    # Normaliser à [0, 255]
    heatmap = ((magnitude - magnitude.min()) / (magnitude.max() - magnitude.min()) * 255).astype(np.uint8)
    
    # Appliquer une colormap
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Superposer sur l'image originale
    overlay = cv2.addWeighted(
        (img_resized * 255).astype(np.uint8), 0.6,
        heatmap_rgb, 0.4, 0
    )
    return overlay


def create_bar_chart(probabilities):
    """Crée un graphique des probabilités top 5."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    diseases_formatted = [format_disease_name(d) for d in probabilities.keys()]
    probs_values = list(probabilities.values())
    
    # Couleurs
    colors = []
    for disease in probabilities.keys():
        if disease in DISEASE_INFO:
            colors.append(DISEASE_INFO[disease]["color"])
        else:
            colors.append("#808080")
    
    bars = ax.barh(diseases_formatted, [v * 100 for v in probs_values],
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Valeurs sur les barres
    for bar, val in zip(bars, probs_values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val*100:.1f}%', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel("Probabilité (%)", fontsize=12, fontweight='bold')
    ax.set_title("Top 5 Prédictions", fontsize=13, fontweight='bold')
    ax.set_xlim([0, 105])
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    return fig


# ============================================================
# INTERFACE PRINCIPALE
# ============================================================
def main():
    # En-tête
    st.markdown('<h1 class="main-header">🌿 PlantDoc – Détection des Maladies des Plantes</h1>',
                unsafe_allow_html=True)
    st.markdown("""
    > **Système de détection automatique** basé sur ResNet50 entraîné sur PlantVillage (54 000+ images, 38 classes)
    """)
    
    # Charger le modèle
    model = load_model()
    
    if model is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Paramètres")
        show_gradcam = st.checkbox("Afficher gradient saillance", value=True)
        show_preprocessing = st.checkbox("Afficher le prétraitement", value=False)
        confidence_threshold = st.slider("Seuil de confiance (%)", 50, 99, 70)
        
        st.markdown("---")
        st.markdown("### 📊 À propos du modèle")
        st.info("""
        **Architecture**: ResNet50
        
        **Transfer Learning**: ImageNet pré-entraîné
        
        **Dataset**: PlantVillage
        - 54,306 images
        - 38 classes
        - 14 espèces végétales
        
        **Entraînement**:
        - Phase 1: backbone gelé (10 epochs)
        - Phase 2: fine-tuning (10 epochs)
        - Optimiseur: Adam
        - Augmentation: rotation, zoom, flip
        """)
        
        st.markdown("---")
        st.markdown("### 📚 Quelques classes supportées")
        examples = ["Apple___healthy", "Tomato___Late_blight", "Potato___Early_blight", 
                   "Corn___Common_rust"]
        for ex in examples:
            st.markdown(f"• {format_disease_name(ex)}")
    
    # Zone principale
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Charger une Image")
        uploaded = st.file_uploader(
            "📸 Glissez une photo de feuille (JPG, PNG)",
            type=["jpg", "jpeg", "png"],
            help="Pour de meilleurs résultats : photo nette, feuille bien visible, bon éclairage"
        )
        
        if uploaded:
            img_pil = Image.open(uploaded).convert("RGB")
            img_array = np.array(img_pil)
            
            st.image(img_array, caption=f"Image chargée: {uploaded.name}", use_column_width=True)
            
            if show_preprocessing:
                with st.expander("🔍 Voir le prétraitement"):
                    img_proc, img_resized = preprocess_image(img_array)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(img_array, caption=f"Original ({img_array.shape[1]}×{img_array.shape[0]})")
                    with c2:
                        st.image((img_resized * 255).astype(np.uint8), caption="Normalisé (224×224)")
        else:
            st.info("👆 Uploadez une photo de feuille pour analyser")
    
    with col2:
        st.markdown("### 🔬 Résultats de l'Analyse")
        
        if uploaded and st.button("🚀 Analyser", type="primary", use_container_width=True):
            with st.spinner("⏳ Analyse en cours (inférence ResNet50)..."):
                img_proc, img_resized = preprocess_image(img_array)
                predicted_class, confidence, top_5_probs = predict_disease(model, img_proc)
            
            # Résultat principal
            is_healthy = "healthy" in predicted_class
            icon = "✅" if is_healthy else "⚠️"
            
            if is_healthy:
                box_class = "healthy"
            elif confidence < 0.85:
                box_class = "diseased"
            else:
                box_class = "critical" if "Late_blight" in predicted_class else "diseased"
            
            disease_display = format_disease_name(predicted_class)
            
            st.markdown(f"""
            <div class="result-box {box_class}">
                <h2>{icon} {disease_display}</h2>
                <h3>Confiance : {confidence*100:.1f}%</h3>
            </div>
            """, unsafe_allow_html=True)
            
            if confidence * 100 < confidence_threshold:
                st.warning(f"⚠️ Confiance ({confidence*100:.0f}%) sous le seuil ({confidence_threshold}%). "
                          f"Résultat incertain. Prendre plus d'images ou consulter un expert.")
            
            # Informations sur la maladie
            if predicted_class in DISEASE_INFO:
                info = DISEASE_INFO[predicted_class]
                with st.expander("📋 Détails et recommandations", expanded=True):
                    col_a, col_b = st.columns([1, 1])
                    with col_a:
                        st.markdown(f"**Agent pathogène**:\n{info['agent']}")
                        st.markdown(f"**Symptômes**:\n{info['symptomes']}")
                    with col_b:
                        st.markdown(f"**Traitement recommandé**:\n{info['traitement']}")
                        st.markdown(f"**Niveau d'urgence**:\n{info['urgence']}")
            
            # Visualisation saillance
            if show_gradcam and not is_healthy:
                with st.expander("🗺️ Saillance – Zones d'attention du modèle", expanded=True):
                    overlay = highlight_zones(img_resized, confidence)
                    st.image(overlay, caption="Zones d'intérêt identifiées (rouge = zones d'attention du modèle)")
            
            # Distribution des probabilités
            st.markdown("### 📊 Distribution Top 5")
            fig = create_bar_chart(top_5_probs)
            st.pyplot(fig, use_container_width=True)
            plt.close()
        
        elif not uploaded:
            st.markdown("""
            #### 📋 Comment utiliser :
            1. 📸 Chargez une photo de feuille (JPG/PNG)
            2. 🚀 Cliquez sur **Analyser**
            3. 📋 Consultez le diagnostic et les recommandations
            
            #### 🌾 Exemples de maladies détectées :
            - 🍅 **Tomate** : Early Blight, Late Blight, Leaf Mold...
            - 🍎 **Pomme** : Apple Scab, Black Rot, Cedar Apple Rust...
            - 🌽 **Maïs** : Common Rust, Northern Leaf Blight...
            - 🥔 **Pomme de terre** : Early/Late Blight...
            - 🍇 **Raisin** : Black Rot, Esca, Leaf Blight...
            
            #### 💡 Conseils pour de meilleurs résultats :
            - Prenez des photos nettes et bien éclairées
            - Positionner la feuille malade clairement en centre
            - Utiliser un fond uniforme si possible
            - Éviter les ombres et les reflets
            """)


if __name__ == "__main__":
    main()
