import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH  = "custom_cnn_best.keras"
IMG_SIZE    = (224, 224)
THRESHOLD   = 0.6
CLASS_NAMES = ["cracked", "not_cracked"]   # must match training class order

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Concrete Crack Detector",
    layout="centered",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: #0f1117; }

    .hero {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem;
    }
    .hero h1 {
        font-size: 2.2rem;
        font-weight: 700;
        color: #f0f0f0;
        letter-spacing: -0.5px;
        margin-bottom: 0.4rem;
    }
    .hero p {
        color: #9ca3af;
        font-size: 0.95rem;
        margin: 0;
    }

    .result-box {
        border-radius: 12px;
        padding: 1.6rem 2rem;
        margin: 1.5rem 0;
        text-align: center;
    }
    .result-cracked {
        background: #1f0a0a;
        border: 1.5px solid #ef4444;
    }
    .result-safe {
        background: #071a10;
        border: 1.5px solid #22c55e;
    }
    .result-unknown {
        background: #1a1a0a;
        border: 1.5px solid #eab308;
    }
    .result-label {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .result-cracked .result-label { color: #ef4444; }
    .result-safe    .result-label { color: #22c55e; }
    .result-unknown .result-label { color: #eab308; }
    .result-sub {
        font-size: 0.85rem;
        color: #9ca3af;
    }

    .score-mono {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.6rem;
        font-weight: 500;
    }
    .result-cracked .score-mono { color: #f87171; }
    .result-safe    .score-mono { color: #4ade80; }
    .result-unknown .score-mono { color: #facc15; }

    .metric-row {
        display: flex;
        gap: 0.75rem;
        justify-content: center;
        margin-top: 1.2rem;
        flex-wrap: wrap;
    }
    .metric-pill {
        background: #1e2130;
        border: 1px solid #2d3148;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-size: 0.8rem;
        color: #d1d5db;
        text-align: center;
    }
    .metric-pill span {
        display: block;
        font-size: 1rem;
        font-weight: 600;
        color: #f0f0f0;
        font-family: 'JetBrains Mono', monospace;
    }

    .upload-hint {
        color: #6b7280;
        font-size: 0.82rem;
        text-align: center;
        margin-top: 0.5rem;
    }

    .divider {
        border: none;
        border-top: 1px solid #1e2130;
        margin: 2rem 0;
    }

    .footer {
        text-align: center;
        color: #374151;
        font-size: 0.75rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

# ── Preprocessing (same pattern as the working potato app) ────────────────────
def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)   # NO /255 — match training pipeline
    return np.expand_dims(arr, axis=0)

model = load_model()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>Concrete Crack Detector</h1>
    <p>Upload a concrete surface image or take a photo — the model will tell you whether it is cracked.</p>
</div>
""", unsafe_allow_html=True)

# ── Input — upload or camera ──────────────────────────────────────────────────
tab_upload, tab_camera = st.tabs(["Upload Image", "Take Photo"])

image = None
caption = None

with tab_upload:
    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed",
    )
    st.markdown('<p class="upload-hint">Supported formats: JPG · PNG · BMP · WEBP</p>', unsafe_allow_html=True)
    if uploaded:
        image = Image.open(uploaded)
        caption = uploaded.name

with tab_camera:
    captured = st.camera_input("Point camera at a concrete surface")
    if captured:
        image = Image.open(captured)
        caption = "Camera capture"

# ── Inference ─────────────────────────────────────────────────────────────────
if image:
    col_img, _ = st.columns([1, 0.05])
    with col_img:
        st.image(image, width="stretch", caption=caption)

    with st.spinner("Analysing..."):
        tensor = preprocess(image)
        raw    = model.predict(tensor, verbose=0)[0]   # shape (2,)

    # Softmax: index order must match CLASS_NAMES / training class_indices
    p_cracked     = float(raw[0])
    p_not_cracked = float(raw[1])
    max_prob      = max(p_cracked, p_not_cracked)
    pred_idx      = int(np.argmax(raw))

    if max_prob < THRESHOLD:
        label = "unrecognised"
        score = max_prob
    elif pred_idx == 0:          # CLASS_NAMES[0] == "cracked"
        label = "cracked"
        score = p_cracked
    else:
        label = "not_cracked"
        score = p_not_cracked

    if label == "cracked":
        st.markdown(f"""
        <div class="result-box result-cracked">
            <div class="result-label">Cracked</div>
            <div class="score-mono">{score:.4f}</div>
            <div class="result-sub">Crack probability (threshold = {THRESHOLD})</div>
        </div>
        """, unsafe_allow_html=True)
    elif label == "not_cracked":
        st.markdown(f"""
        <div class="result-box result-safe">
            <div class="result-label">Not Cracked</div>
            <div class="score-mono">{score:.4f}</div>
            <div class="result-sub">Healthy probability (threshold = {THRESHOLD})</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-box result-unknown">
            <div class="result-label">Unrecognised</div>
            <div class="score-mono">{score:.4f}</div>
            <div class="result-sub">Max confidence below threshold ({THRESHOLD})</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-pill">P(Cracked)<span>{p_cracked*100:.1f}%</span></div>
        <div class="metric-pill">P(Not Cracked)<span>{p_not_cracked*100:.1f}%</span></div>
        <div class="metric-pill">Threshold<span>{THRESHOLD:.1f}</span></div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Model details"):
        st.markdown(f"""
        **Raw softmax:** `[{raw[0]:.6f}, {raw[1]:.6f}]`

        **Class order:** `{CLASS_NAMES}`

        | Metric | Value |
        |---|---|
        | Accuracy | 96.47% |
        | Precision | 96.37% |
        | Recall | 96.47% |
        | F1-Score | 96.38% |

        **Per-class performance**

        | Class | Precision | Recall | F1 |
        |---|---|---|---|
        | cracked | 97.32% | 98.72% | 98.01% |
        | not_cracked | 89.31% | 79.78% | 84.27% |

        *Evaluated on 1,502 test samples.*
        """)

else:
    st.markdown("""
    <div style="text-align:center; padding: 2rem 0; color: #4b5563;">
        <div style="font-size: 0.9rem;">No image yet. Upload one or take a photo to get started.</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown('<div class="footer">Custom CNN · Trained on Concrete Crack Dataset · 96.47% accuracy</div>', unsafe_allow_html=True)