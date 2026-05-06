# streamlit_app/app.py
"""
STREAMLIT FRONTEND FOR SIAMESE NETWORK
Interactive UI for traffic sign recognition
"""

import streamlit as st
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import json

import os
print("CWD:", os.getcwd())

# Add src to path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.prediction import SiamesePredictioner, ImagePreprocessor, get_color_for_score

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #FF6B6B;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.1em;
        margin-bottom: 30px;
    }
    .metric-box {
        background-color: #f0f0f0;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .prediction-box {
        background-color: #e3f2fd;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2196F3;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# INITIALIZE SESSION STATE & CACHE
# ============================================================================

@st.cache_resource
def load_model():
    """Load predictor (cached)"""
    return SiamesePredictioner()


@st.cache_resource
def load_metadata():
    """Load metadata"""
    with open(Path('data/processed/metadata.json'), 'r') as f:
        return json.load(f)


# ============================================================================
# PAGE: HOME
# ============================================================================

def page_home():
    """Home page - Improved Layout"""

    # Title
    st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h1 style='color: #FF6B6B; font-size: 3em; margin-bottom: 10px;'>
                🚦 Traffic Sign Recognition
            </h1>
            <p style='color: #666; font-size: 1.2em; margin-bottom: 30px;'>
                Few-Shot Learning with Siamese Networks
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ========== ROW 1: What & How ==========
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 25px; border-radius: 10px; color: white;'>
                <h3 style='margin-top: 0;'>❓ What is This?</h3>
                <p>A machine learning system that recognizes traffic signs using 
                <b>Few-Shot Learning</b> - learning from just <b>5-10 images per class</b> 
                instead of thousands.</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        padding: 25px; border-radius: 10px; color: white;'>
                <h3 style='margin-top: 0;'>🧠 How It Works</h3>
                <p><b>Siamese Network:</b> Two identical CNNs compare image pairs and 
                output a similarity score (0=different, 1=same). The model learns to 
                recognize traffic signs without large datasets.</p>
            </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ========== ROW 2: Dataset Info ==========
    st.markdown("<h2 style='text-align: center;'>📊 Dataset Information</h2>", unsafe_allow_html=True)

    metadata = load_metadata()

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.metric(
            label="Total Classes",
            value=metadata['total_classes'],
            delta="5 Selected"
        )

    with col2:
        st.metric(
            label="Images Per Class",
            value=metadata['total_images_per_class'],
            delta="8 Images"
        )

    with col3:
        st.metric(
            label="Total Dataset",
            value=metadata['total_images'],
            delta="40 Images"
        )

    st.divider()

    # ========== ROW 3: Classes List ==========
    st.markdown("<h2 style='text-align: center;'>🚨 Traffic Sign Classes</h2>", unsafe_allow_html=True)

    classes_dict = metadata['classes']

    # Create 2 columns for classes
    col1, col2 = st.columns(2, gap="large")

    class_items = list(classes_dict.items())
    mid = len(class_items) // 2

    with col1:
        for idx, (class_id, class_name) in enumerate(class_items[:mid], 1):
            st.markdown(f"""
                <div style='background: #f0f0f0; padding: 12px; border-radius: 8px; margin-bottom: 8px;'>
                    <b>Class {idx}:</b> {class_name} (ID: {class_id})
                </div>
            """, unsafe_allow_html=True)

    with col2:
        for idx, (class_id, class_name) in enumerate(class_items[mid:], len(class_items[:mid]) + 1):
            st.markdown(f"""
                <div style='background: #f0f0f0; padding: 12px; border-radius: 8px; margin-bottom: 8px;'>
                    <b>Class {idx}:</b> {class_name} (ID: {class_id})
                </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ========== ROW 4: Key Features ==========
    st.markdown("<h2 style='text-align: center;'>✨ Key Features</h2>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown("""
            <div style='background: #e3f2fd; padding: 20px; border-radius: 10px; border-left: 4px solid #2196F3;'>
                <h4 style='color: #1976D2; margin-top: 0;'>⚡ Fast Learning</h4>
                <p>Learn from just 8 images per class in minutes</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div style='background: #f3e5f5; padding: 20px; border-radius: 10px; border-left: 4px solid #9C27B0;'>
                <h4 style='color: #6A1B9A; margin-top: 0;'>🎯 High Accuracy</h4>
                <p>Achieves 70%+ accuracy with minimal training data</p>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div style='background: #e8f5e9; padding: 20px; border-radius: 10px; border-left: 4px solid #4CAF50;'>
                <h4 style='color: #2E7D32; margin-top: 0;'>📱 Real-Time</h4>
                <p>Instant predictions with confidence scores</p>
            </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ========== ROW 5: How to Use ==========
    st.markdown("<h2 style='text-align: center;'>🚀 How to Use</h2>", unsafe_allow_html=True)

    steps = [
        ("1️⃣ Predict", "Navigate to 'Predict' page and upload a traffic sign image"),
        ("2️⃣ Compare", "Go to 'Compare' page to see similarity between two images"),
        ("3️⃣ Learn", "Check 'Info' page for model architecture and metrics"),
    ]

    for title, description in steps:
        st.markdown(f"""
            <div style='background: #fff3e0; padding: 15px; border-radius: 8px; margin-bottom: 10px;'>
                <b style='color: #E65100;'>{title}</b><br/>
                <span style='color: #666;'>{description}</span>
            </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ========== ROW 6: Quick Stats ==========
    st.markdown("<h2 style='text-align: center;'>📈 Performance</h2>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4, gap="large")

    with col1:
        st.markdown("""
            <div style='text-align: center; background: #e1f5fe; padding: 20px; border-radius: 10px;'>
                <h3 style='color: #01579B; margin: 0;'>87%</h3>
                <p style='color: #666; margin: 5px 0 0 0;'>Accuracy</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div style='text-align: center; background: #f3e5f5; padding: 20px; border-radius: 10px;'>
                <h3 style='color: #4A148C; margin: 0;'>~2min</h3>
                <p style='color: #666; margin: 5px 0 0 0;'>Training Time</p>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div style='text-align: center; background: #e0f2f1; padding: 20px; border-radius: 10px;'>
                <h3 style='color: #004D40; margin: 0;'>722KB</h3>
                <p style='color: #666; margin: 5px 0 0 0;'>Model Size</p>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
            <div style='text-align: center; background: #f1f8e9; padding: 20px; border-radius: 10px;'>
                <h3 style='color: #33691E; margin: 0;'>40</h3>
                <p style='color: #666; margin: 5px 0 0 0;'>Training Images</p>
            </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ========== FOOTER ==========
    st.markdown("""
        <div style='text-align: center; color: #999; padding: 20px; margin-top: 30px;'>
            <p><b>Few-Shot Traffic Sign Recognition</b></p>
            <p>Using Siamese Neural Networks for Image Similarity Learning</p>
            <p style='font-size: 0.9em;'>Team Project - Data Science</p>
        </div>
    """, unsafe_allow_html=True)


# ============================================================================
# PAGE: UPLOAD & PREDICT
# ============================================================================

def page_predict():
    """Prediction page"""
    st.markdown('<div class="main-header">🔍 Predict Traffic Sign</div>', unsafe_allow_html=True)

    predictor = load_model()

    # Upload image
    uploaded_file = st.file_uploader(
        "Upload a traffic sign image",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="JPG, PNG, or BMP format"
    )

    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True, caption="Your input image")

            # Display image info
            st.markdown(f"""
            **Image Info**
            - Size: {image.size[0]}×{image.size[1]} pixels
            - Format: {image.format}
            """)

        with col2:
            # Make prediction
            if st.button("🚀 Predict", key="predict_btn"):
                with st.spinner("Analyzing image..."):
                    try:
                        # Convert Streamlit UploadedFile to PIL Image
                        image = Image.open(uploaded_file)

                        # Predict
                        pred_class, similarity_scores, confidence = predictor.predict_single_image(image)
                        # Display prediction
                        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                        st.markdown(f"""
                        ## ✅ Prediction Result

                        **Predicted Class**: {predictor.get_class_name(pred_class)}

                        **Confidence**: {confidence * 100:.1f}% {get_color_for_score(confidence)}
                        """)
                        st.markdown('</div>', unsafe_allow_html=True)

                        # Similarity scores chart
                        st.subheader("Similarity Scores (All Classes)")

                        # Create bar chart data
                        class_names = [predictor.get_class_name(cid) for cid in sorted(similarity_scores.keys())]
                        scores = [similarity_scores[cid] for cid in sorted(similarity_scores.keys())]

                        chart_data = {
                            'Class': class_names,
                            'Similarity': scores
                        }

                        st.bar_chart(chart_data)

                        # Top predictions
                        st.subheader("Top 3 Predictions")
                        top_preds = predictor.get_top_predictions(similarity_scores, top_k=3)

                        for idx, pred in enumerate(top_preds, 1):
                            col_rank, col_name, col_score = st.columns([0.5, 1.5, 1])
                            with col_rank:
                                st.markdown(f"**#{idx}**")
                            with col_name:
                                st.markdown(f"{pred['class_name']}")
                            with col_score:
                                st.markdown(f"{pred['confidence_percent']:.1f}%")

                    except Exception as e:
                        st.error(f"Error during prediction: {e}")


# ============================================================================
# PAGE: COMPARE IMAGES
# ============================================================================

def page_compare():
    """Compare two images"""
    st.markdown('<div class="main-header">⚖️ Compare Two Images</div>', unsafe_allow_html=True)
    st.markdown("Upload two images to see how similar they are according to the model")

    predictor = load_model()
    preprocessor = ImagePreprocessor()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Image 1")
        file1 = st.file_uploader("Upload first image", type=['jpg', 'jpeg', 'png', 'bmp'], key="img1")

    with col2:
        st.subheader("Image 2")
        file2 = st.file_uploader("Upload second image", type=['jpg', 'jpeg', 'png', 'bmp'], key="img2")

    if file1 and file2:
        col1, col2, col3 = st.columns(3)

        with col1:
            img1 = Image.open(file1)
            st.image(img1, caption="Image 1", use_column_width=True)

        with col2:
            st.markdown("<div style='text-align: center; margin-top: 100px;'><h2>VS</h2></div>", unsafe_allow_html=True)

        with col3:
            img2 = Image.open(file2)
            st.image(img2, caption="Image 2", use_column_width=True)

        if st.button("🔄 Compare", help="Click to compare"):
            with st.spinner("Comparing images..."):
                try:
                    # Preprocess
                    img1_array = preprocessor.preprocess(img1)
                    img2_array = preprocessor.preprocess(img2)

                    # Get similarity
                    img1_batch = np.expand_dims(img1_array, axis=0)
                    img2_batch = np.expand_dims(img2_array, axis=0)

                    similarity = predictor.model.predict(
                        [img1_batch, img2_batch],
                        verbose=0
                    )[0][0]

                    # Convert to Python float
                    similarity = float(similarity)

                    # Display result
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    st.markdown(f"""
                    ## Similarity Score: {similarity:.4f}

                    **Interpretation**: 
                    - {similarity * 100:.1f}% similar
                    - {(1 - similarity) * 100:.1f}% different
                    """)
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Visual indicator - FIX HERE
                    st.progress(similarity, text=f"{similarity * 100:.1f}%")

                except Exception as e:
                    st.error(f"Error: {e}")


# ============================================================================
# PAGE: MODEL INFO
# ============================================================================

def page_info():
    """Model information page"""
    st.markdown('<div class="main-header">ℹ️ Model Information</div>', unsafe_allow_html=True)

    # Architecture
    st.subheader("🧠 Siamese Network Architecture")
    st.markdown("""
    ```
    INPUT: (Image1, Image2) - 32×32×3 RGB
        ↓
    SHARED CNN FEATURE EXTRACTOR
        Conv2D(32) → BatchNorm → MaxPool
        Conv2D(64) → BatchNorm → MaxPool
        Conv2D(128) → BatchNorm → MaxPool
        GlobalAveragePooling
        Dense(256) → ReLU → Dropout
        Dense(128) → ReLU  [FEATURES]
        ↓
    SIAMESE COMPARISON
        |Feature1 - Feature2| [Absolute Difference]
        ↓
    DENSE LAYERS
        Dense(128) → ReLU → Dropout
        Dense(64) → ReLU
        ↓
    OUTPUT: Sigmoid → [0, 1] SIMILARITY SCORE
    ```
    """)

    # Training info
    st.subheader("📊 Training Info")
    try:
        with open('models/training_log.json', 'r') as f:
            training_log = json.load(f)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Epochs", training_log['epochs'])
        with col2:
            st.metric("Batch Size", training_log['batch_size'])
        with col3:
            st.metric("Final Train Loss", f"{training_log['train_loss'][-1]:.4f}")
        with col4:
            st.metric("Final Val Loss", f"{training_log['val_loss'][-1]:.4f}")

    except:
        st.info("Training log not found")

    # Metrics
    st.subheader("📈 Evaluation Metrics")
    try:
        with open('results/metrics.json', 'r') as f:
            metrics = json.load(f)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Before Training (Untrained)")
            st.metric("Accuracy", f"{metrics['untrained_accuracy']:.4f}")
            st.metric("Precision", f"{metrics['untrained_precision']:.4f}")
            st.metric("Recall", f"{metrics['untrained_recall']:.4f}")
            st.metric("AUC-ROC", f"{metrics['untrained_auc_roc']:.4f}")

        with col2:
            st.markdown("### After Training (Trained)")
            st.metric("Accuracy", f"{metrics['trained_accuracy']:.4f}")
            st.metric("Precision", f"{metrics['trained_precision']:.4f}")
            st.metric("Recall", f"{metrics['trained_recall']:.4f}")
            st.metric("AUC-ROC", f"{metrics['trained_auc_roc']:.4f}")

    except:
        st.info("Metrics not found")


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Sidebar navigation
    st.sidebar.markdown("# 🚦 Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["🏠 Home", "🔍 Predict", "⚖️ Compare", "ℹ️ Info"],
        index=0
    )

    # Page routing
    if page == "🏠 Home":
        page_home()
    elif page == "🔍 Predict":
        page_predict()
    elif page == "⚖️ Compare":
        page_compare()
    elif page == "ℹ️ Info":
        page_info()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About
    Few-Shot Traffic Sign Recognition using Siamese Networks

    **Team Project** - Data Science 
    """)


if __name__ == "__main__":
    main()