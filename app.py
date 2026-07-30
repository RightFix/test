import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH  = "tl_feature_extraction_best.keras"
IMG_SIZE    = (224, 224)
CLASS_NAMES = ["orange", "rottenoranges"]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Orange Freshness Classifier",
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
    .result-fresh {
        background: #071a10;
        border: 1.5px solid #22c55e;
    }
    .result-rotten {
        background: #1f0a0a;
        border: 1.5px solid #ef4444;
    }
    .result-label {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .result-fresh  .result-label { color: #22c55e; }
    .result-rotten .result-label { color: #ef4444; }
    .result-sub {
        font-size: 0.85rem;
        color: #9ca3af;
    }

    .score-mono {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.6rem;
        font-weight: 500;
    }
    .result-fresh  .score-mono { color: #4ade80; }
    .result-rotten .score-mono { color: #f87171; }

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

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)   # NO /255
    return np.expand_dims(arr, axis=0)

model = load_model()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>Orange Freshness Classifier</h1>
    <p>Upload an image of an orange — the model will tell you whether it is fresh or rotten.</p>
</div>
""", unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
tab_upload, tab_camera = st.tabs(["Upload Image", "Take Photo"])

image  = None
caption = None

with tab_upload:
    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed",
    )
    st.markdown(
        '<p class="upload-hint">Supported formats: JPG · PNG · BMP · WEBP</p>',
        unsafe_allow_html=True,
    )
    if uploaded:
        image   = Image.open(uploaded)
        caption = uploaded.name

with tab_camera:
    captured = st.camera_input("Point camera at an orange")
    if captured:
        image   = Image.open(captured)
        caption = "Camera capture"

# ── Inference ─────────────────────────────────────────────────────────────────
if image:
    st.image(image, width="stretch", caption=caption)

    with st.spinner("Analysing..."):
        tensor = preprocess(image)
        raw    = model.predict(tensor, verbose=0)[0]

    p_fresh  = float(raw[0])
    p_rotten = float(raw[1])
    pred_idx = int(np.argmax(raw))

    if pred_idx == 0:
        label = "orange"
        score = p_fresh
    else:
        label = "rottenoranges"
        score = p_rotten

    if label == "orange":
        st.markdown(f"""
        <div class="result-box result-fresh">
            <div class="result-label">Fresh Orange</div>
            <div class="score-mono">{score:.4f}</div>
            <div class="result-sub">Fresh probability</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-box result-rotten">
            <div class="result-label">Rotten Orange</div>
            <div class="score-mono">{score:.4f}</div>
            <div class="result-sub">Rotten probability</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-pill">P(Fresh)<span>{p_fresh*100:.1f}%</span></div>
        <div class="metric-pill">P(Rotten)<span>{p_rotten*100:.1f}%</span></div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Model details"):
        st.markdown(f"""
        **Raw softmax:** `[{raw[0]:.6f}, {raw[1]:.6f}]`

        **Class order:** `{CLASS_NAMES}`

        | Metric | Value |
        |---|---|
        | Accuracy | 96.47% |
        | F1-Score | 96.38% |

        *Model: MobileNetV2 Transfer Learning · Input: 224 × 224 px*
        """)

else:
    st.markdown("""
    <div style="text-align:center; padding: 2rem 0; color: #4b5563;">
        <div style="font-size: 0.9rem;">No image yet. Upload one or take a photo to get started.</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown(
    '<div class="footer">MobileNetV2 Transfer Learning · Trained on Fruit Freshness Dataset · 96.47% accuracy</div>',
    unsafe_allow_html=True,
)
