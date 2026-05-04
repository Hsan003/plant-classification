"""
PROJET TRAITEMENT D'IMAGES - Détection des Maladies des Plantes
Fichier 04 : Interface Web Streamlit
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
from tensorflow.keras.applications.resnet50 import preprocess_input

# ============================================================
# IMPORT CONFIG CENTRALISÉE
# ============================================================
from config_shared import CONFIG, CLASS_NAMES, DISEASE_INFO, PATHS, load_class_metadata

NUM_CLASSES = len(CLASS_NAMES)

# ============================================================
# CONFIGURATION DE L'APPLICATION
# ============================================================
st.set_page_config(
    page_title="PlantDoc – Détection de Maladies",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

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


# ============================================================
# MODÈLE TENSORFLOW
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model():
    """Charge le modèle entraîné avec cache. Fallback .h5 pour rétrocompatibilité."""
    model_path = PATHS["model_final"]

    # Fallback si un ancien modèle .h5 existe encore
    if not os.path.exists(model_path):
        fallback_path = model_path.replace(".keras", ".h5")
        if os.path.exists(fallback_path):
            model_path = fallback_path
        else:
            st.error(f"❌ Modèle non trouvé: {PATHS['model_final']}")
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
    """
    Prétraitement cohérent avec l'entraînement:
    - Redimensionnement
    - preprocess_input de ResNet50 (centrage ImageNet, PAS de /255)
    """
    resized = cv2.resize(img_array, size, interpolation=cv2.INTER_LINEAR)
    processed = preprocess_input(resized.astype(np.float32))
    return processed, resized


def predict_disease(model, img_array):
    """Effectue une prédiction avec le modèle entraîné."""
    img_batch = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_batch, verbose=0)
    probabilities = predictions[0]

    pred_idx = np.argmax(probabilities)
    confidence = probabilities[pred_idx]

    top_5_idx = np.argsort(probabilities)[-5:][::-1]
    top_5_probs = {CLASS_NAMES[i]: float(probabilities[i]) for i in top_5_idx}

    return CLASS_NAMES[pred_idx], float(confidence), top_5_probs


def format_disease_name(class_name):
    """Formate le nom de la classe pour l'affichage."""
    parts = class_name.replace("___", " – ").replace("__", " – ").replace("_", " ")
    return parts


def highlight_zones(img_resized):
    """Crée une visualisation des zones d'intérêt via gradients Sobel."""
    h, w = img_resized.shape[:2]
    gray = cv2.cvtColor((img_resized * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    gradx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grady = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gradx**2 + grady**2)

    rng = magnitude.max() - magnitude.min()
    heatmap = ((magnitude - magnitude.min()) / (rng if rng > 0 else 1) * 255).astype(np.uint8)

    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

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

    colors = []
    for disease in probabilities.keys():
        if disease in DISEASE_INFO:
            colors.append(DISEASE_INFO[disease]["color"])
        else:
            colors.append("#808080")

    bars = ax.barh(diseases_formatted, [v * 100 for v in probs_values],
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1)

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
    st.markdown('<h1 class="main-header">🌿 PlantDoc – Détection des Maladies des Plantes</h1>',
                unsafe_allow_html=True)
    st.markdown(f"""
    > **Système de détection automatique** basé sur ResNet50 entraîné sur un sous-ensemble
    > PlantVillage ({NUM_CLASSES} classes : Pepper, Potato, Tomato)
    """)

    model = load_model()
    if model is None:
        st.stop()

    with st.sidebar:
        st.markdown("### ⚙️ Paramètres")
        show_gradcam = st.checkbox("Afficher gradient saillance", value=True)
        show_preprocessing = st.checkbox("Afficher le prétraitement", value=False)
        confidence_threshold = st.slider("Seuil de confiance (%)", 50, 99, 70)

        st.markdown("---")
        st.markdown("### 📊 À propos du modèle")
        st.info(f"""
        **Architecture**: ResNet50

        **Transfer Learning**: ImageNet pré-entraîné

        **Dataset**: PlantVillage (sous-ensemble)
        - {NUM_CLASSES} classes détectées
        - Pepper, Potato, Tomato

        **Entraînement**:
        - Phase 1: backbone gelé ({CONFIG['epochs_frozen']} epochs)
        - Phase 2: fine-tuning ({CONFIG['epochs_finetune']} epochs)
        - Optimiseur: Adam
        - Class weights pour déséquilibre
        """)

        st.markdown("---")
        st.markdown("### 📚 Classes supportées")
        for ex in CLASS_NAMES[:5]:
            st.markdown(f"• {format_disease_name(ex)}")
        if len(CLASS_NAMES) > 5:
            st.markdown(f"*... et {len(CLASS_NAMES)-5} autres*")

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
            st.image(img_array, caption=f"Image chargée: {uploaded.name}", use_container_width=True)

            if show_preprocessing:
                with st.expander("🔍 Voir le prétraitement"):
                    img_proc, img_resized = preprocess_image(img_array)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(img_array, caption=f"Original ({img_array.shape[1]}×{img_array.shape[0]})")
                    with c2:
                        st.image((img_resized * 255).astype(np.uint8), caption="Redimensionnée (224×224)")
        else:
            st.info("👆 Uploadez une photo de feuille pour analyser")

    with col2:
        st.markdown("### 🔬 Résultats de l'Analyse")

        if uploaded and st.button("🚀 Analyser", type="primary", use_container_width=True):
            with st.spinner("⏳ Analyse en cours (inférence ResNet50)..."):
                img_proc, img_resized = preprocess_image(img_array)
                predicted_class, confidence, top_5_probs = predict_disease(model, img_proc)

            is_healthy = "healthy" in predicted_class
            icon = "✅" if is_healthy else "⚠️"

            if is_healthy:
                box_class = "healthy"
            elif confidence < 0.85:
                box_class = "diseased"
            else:
                box_class = "critical" if "Late_blight" in predicted_class or "virus" in predicted_class else "diseased"

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

            if show_gradcam and not is_healthy:
                with st.expander("🗺️ Saillance – Zones d'attention", expanded=True):
                    overlay = highlight_zones(img_resized)
                    st.image(overlay, caption="Zones d'intérêt identifiées (rouge = zones d'attention)")

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

            #### 🌾 Classes détectées :
            - 🫑 **Pepper** : Bacterial spot, Healthy
            - 🥔 **Potato** : Early blight, Late blight, Healthy
            - 🍅 **Tomato** : Bacterial spot, Early/Late blight, Leaf Mold,
              Septoria, Spider mites, Target Spot, Mosaic virus, Yellow Leaf Curl Virus, Healthy

            #### 💡 Conseils :
            - Photos nettes et bien éclairées
            - Feuille malade au centre
            - Fond uniforme si possible
            """)


if __name__ == "__main__":
    main()
