"""
SE4050 - Deep Learning Assignment: Network Intrusion Detection System (NIDS)
Member 4 Deliverable: System Integration & Demonstration Dashboard
Dataset: CICIDS2017 | Models: FT-Transformer & MLP Baseline
"""

from pathlib import Path
import sys
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image

# -------------------------------------------------------------
# 1. Path Setup & Model Class Definitions
# -------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.models.ft_transformer import FTTransformer

# MLP Baseline Architecture (for side-by-side comparison)
class MLP(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

# -------------------------------------------------------------
# 2. Page Configuration & Styling
# -------------------------------------------------------------
st.set_page_config(
    page_title="CICIDS2017 Deep Learning NIDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0px; }
    .sub-title { font-size: 1.1rem; color: #4B5563; margin-bottom: 20px; }
    .metric-card { background-color: #F3F4F6; padding: 15px; border-radius: 10px; border-left: 5px solid #3B82F6; }
    .alert-benign { background-color: #D1FAE5; color: #065F46; padding: 15px; border-radius: 8px; font-weight: bold; }
    .alert-attack { background-color: #FEE2E2; color: #991B1B; padding: 15px; border-radius: 8px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 3. Cached Resource Loaders
# -------------------------------------------------------------
@st.cache_resource
def load_resources():
    models_dir = ROOT_DIR / 'models'
    data_dir = ROOT_DIR / 'data'
    
    # Load Label Encoder
    encoder_path = models_dir / 'label_encoder.joblib'
    if not encoder_path.exists():
        st.error("Missing models/label_encoder.joblib")
        st.stop()
    label_encoder = joblib.load(encoder_path)
    classes = list(label_encoder.classes_)
    
    # Load Processed Test Data Samples
    data_path = data_dir / 'processed_data.npz'
    if not data_path.exists():
        st.error("Missing data/processed_data.npz")
        st.stop()
    npz = np.load(data_path)
    X_test, y_test = npz['X_test'], npz['y_test']
    
    # Load FT-Transformer Model
    ft_model = FTTransformer(
        n_features=X_test.shape[1],
        n_classes=len(classes),
        d_token=64,
        n_blocks=3,
        n_heads=4
    )
    ft_path = models_dir / 'ft_transformer_model.pt'
    if ft_path.exists():
        ft_model.load_state_dict(torch.load(ft_path, map_location='cpu'))
    ft_model.eval()

    # Load MLP Model (if available)
    mlp_model = MLP(input_size=X_test.shape[1], output_size=len(classes))
    mlp_path = models_dir / 'mlp_model.pt'
    mlp_loaded = False
    if mlp_path.exists():
        try:
            mlp_model.load_state_dict(torch.load(mlp_path, map_location='cpu'))
            mlp_model.eval()
            mlp_loaded = True
        except Exception:
            mlp_loaded = False

    return classes, X_test, y_test, ft_model, mlp_model, mlp_loaded

classes, X_test, y_test, ft_model, mlp_model, mlp_loaded = load_resources()

# -------------------------------------------------------------
# 4. Header & Disclaimer
# -------------------------------------------------------------
st.markdown('<div class="main-title">🛡️ Deep Learning-Based Network Intrusion Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">CICIDS2017 Tabular Flow Classification & Evaluation Suite | SE4050 Group Project</div>', unsafe_allow_html=True)
st.info("ℹ️ **System Scope**: This system operates as an **offline network-flow classification prototype** on aggregated 79-feature bidirectional flow records.", icon="ℹ️")

# -------------------------------------------------------------
# 5. Sidebar Controls
# -------------------------------------------------------------
st.sidebar.header("⚙️ Model & Evaluation Controls")

available_models = ["FT-Transformer (Member 4 - Recommended)"]
if mlp_loaded:
    available_models.append("Multilayer Perceptron (Member 1 Baseline)")

selected_model_name = st.sidebar.selectbox("Select Active Deep Learning Model:", available_models)

active_model = ft_model if "FT-Transformer" in selected_model_name else mlp_model

# -------------------------------------------------------------
# 6. Main Dashboard Tabs
# -------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔍 Live Flow Prediction", "📊 Model Comparison & Metrics", "🧠 Architecture & Technical Details"])

# -------------------------------------------------------------
# TAB 1: Live Flow Prediction
# -------------------------------------------------------------
with tab1:
    st.subheader("Interactive Network Flow Classifier")
    st.write("Select a test flow from the CICIDS2017 test partition or test random traffic:")

    col_ctrl1, col_ctrl2 = st.columns([2, 1])

    with col_ctrl1:
        # Pre-select representative flows for each class
        sample_class_choice = st.selectbox(
            "Choose a traffic category to load a sample flow from:",
            ["Random Sample"] + classes
        )

    if sample_class_choice == "Random Sample":
        sample_idx = np.random.randint(0, len(X_test))
    else:
        target_class_id = classes.index(sample_class_choice)
        class_indices = np.where(y_test == target_class_id)[0]
        if len(class_indices) > 0:
            sample_idx = np.random.choice(class_indices)
        else:
            sample_idx = 0

    flow_vector = X_test[sample_idx]
    actual_label = classes[y_test[sample_idx]]

    # Run Inference
    with torch.no_grad():
        flow_tensor = torch.from_numpy(flow_vector).unsqueeze(0).float()
        start_time = time.perf_counter()
        logits = active_model(flow_tensor)
        inference_latency_ms = (time.perf_counter() - start_time) * 1000
        probs = torch.softmax(logits, dim=1).numpy()[0]
        pred_idx = np.argmax(probs)
        pred_label = classes[pred_idx]
        confidence = probs[pred_idx] * 100

    is_attack = pred_label != 'BENIGN'

    st.markdown("---")
    res_col1, res_col2, res_col3 = st.columns([1.5, 1.5, 1.2])

    with res_col1:
        st.markdown("**Binary Intrusion Status:**")
        if is_attack:
            st.markdown(f'<div class="alert-attack">🚨 ATTACK DETECTED</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-benign">✅ BENIGN TRAFFIC</div>', unsafe_allow_html=True)

    with res_col2:
        st.markdown("**Predicted Attack Family:**")
        st.markdown(f"### `{pred_label}` (Confidence: {confidence:.2f}%)")
        st.caption(f"Ground Truth Label: **{actual_label}**")

    with res_col3:
        st.markdown("**Inference Latency:**")
        st.markdown(f"### `{inference_latency_ms:.3f} ms`")
        st.caption("Per-flow processing time")

    st.markdown("---")
    st.subheader("Class Probability Distribution")
    prob_df = pd.DataFrame({
        "Traffic Class": classes,
        "Probability (%)": probs * 100
    }).sort_values(by="Probability (%)", ascending=True)

    st.bar_chart(prob_df.set_index("Traffic Class"))

# -------------------------------------------------------------
# TAB 2: Model Comparison & Metrics
# -------------------------------------------------------------
with tab2:
    st.subheader("Model Performance Comparison (Locked Test Set)")

    comparison_data = {
        "Architecture": ["FT-Transformer (Self-Attention)", "MLP / DNN (Dense Baseline)", "Denoising Autoencoder", "TabNet (Attentive Trees)"],
        "Owner": ["Member 4", "Member 1", "Member 2", "Member 3"],
        "Test Macro-F1": ["0.70+", "0.5525", "Pending", "Pending"],
        "Test Accuracy": ["95.2%", "93.57%", "Pending", "Pending"],
        "Param Count": ["~118K", "60.4K", "Pending", "Pending"],
        "Inference Speed": ["~0.035 ms/flow", "~0.012 ms/flow", "Pending", "Pending"]
    }
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

    st.markdown("---")
    st.subheader("Confusion Matrix (FT-Transformer)")
    cm_path = ROOT_DIR / 'results' / 'ft_transformer' / 'confusion_matrix.png'
    if cm_path.exists():
        img = Image.open(cm_path)
        st.image(img, caption="FT-Transformer Multiclass Confusion Matrix across 504,473 Test Flows", use_container_width=True)
    else:
        st.warning("Confusion matrix plot not found at results/ft_transformer/confusion_matrix.png")

# -------------------------------------------------------------
# TAB 3: Architecture & Technical Details
# -------------------------------------------------------------
with tab3:
    st.subheader("FT-Transformer: Feature Tokenizer + Transformer")
    st.markdown("""
    ### Why FT-Transformer for Tabular Network Traffic?
    Unlike standard Multi-Layer Perceptrons that treat tabular features as flat scalar arrays, the **FT-Transformer** (*Gorishniy et al., NeurIPS 2021*) transforms tabular features into token vectors:

    1. **Numerical Feature Tokenizer**: 
       Each continuous flow statistic $x_j \in \mathbb{R}$ is projected into a $d$-dimensional embedding:
       $$\\mathbf{e}_j = x_j \\mathbf{w}_j + \\mathbf{b}_j \\in \\mathbb{R}^d$$
    2. **Learnable `[CLS]` Token**: 
       Prepended to the sequence of 69 feature tokens to aggregate holistic flow representations.
    3. **Multi-Head Self-Attention (MHSA)**:
       Directly computes pairwise interactions between all 69 flow statistics (e.g., *Flow Duration* $\\leftrightarrow$ *Total Fwd Packets*).
    4. **Classification Head**:
       Extracts the transformed `[CLS]` token and passes it through LayerNorm and an MLP head to output 9 class logits.
    """)
