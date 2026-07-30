import streamlit as st
import numpy as np
import tensorflow as tf

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH  = "mobilenetv2_transfer.keras"
IMG_SIZE    = (224, 224)
THRESHOLD   = 0.6
CLASS_NAMES = ["orange", "rotten_oranges"]   # must match training class order

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Orange Quality Classifier",
    layout="centered",
)


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize(IMG_SIZE)
    #img = tf.keras.utils.load_img(img_path, target_size=image_size)
    #img_array = tf.keras.utils.img_to_array(img)
    #img_array = np.expand_dims(img_array, axis=0) 
    arr = np.array(img, dtype=np.float32)   # NO /255
    return np.expand_dims(arr, axis=0)

model = load_model()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>Orange Quality Classifier</h1>
    <p>Upload a concrete surface image or take a photo — the model will tell you whether it is cracked.</p>
</div>
""", unsafe_allow_html=True)

# ── Input — upload or camera ──────────────────────────────────────────────────
tab_upload = st.tabs(["Upload Image"])

image = None
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
        image = Image.open(uploaded)
        caption = uploaded.name


# ── Inference ─────────────────────────────────────────────────────────────────
if image:
    st.image(image, width="stretch", caption=caption)

    with st.spinner("Analysing..."):
        tensor = preprocess(image)
        raw = model.predict(tensor, verbose=0)[0][0]

    # p_cracked = float(raw[0])
    # p_not_cracked = float(raw[1])
    # max_prob = max(p_cracked, p_not_cracked)
    # pred_idx = int(np.argmax(raw))
    label = class_names[int(raw <= 0.5)]

    # if max_prob <:
    #     label = "unrecognised"
    #     score = max_prob
    if raw >= 0.5 :
        label = "orange"
        score = raw
    else:
        label = "rotten_oranges"
        score = 1 - raw

    if label == "orange":
        st.markdown(f"""
        <div class="result-box result-cracked">
            <div class="result-label">Orange</div>
            <div class="score-mono">{score:.4f}</div>
            <div class="result-sub">Crack probability (threshold = {THRESHOLD})</div>
        </div>
        """, unsafe_allow_html=True)
    elif label == "rotten_oranges":
        st.markdown(f"""
        <div class="result-box result-safe">
            <div class="result-label">Rotten Orange</div>
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


st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown(
    '<div class="footer">Custom CNN · Trained on Concrete Crack Dataset · 96.47% accuracy</div>',
    unsafe_allow_html=True,
)
