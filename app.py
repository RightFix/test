import streamlit as st
import numpy as np
import tensorflow as tf
import tempfile

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH  = "mobilenetv2_transfer.keras"
IMG_SIZE    = (224, 224)
THRESHOLD   = 0.6
CLASS_NAMES = ["orange", "rottenoranges"]   # must match training class order

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Orange Quality Classifier",
    layout="centered",
)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>Orange Quality Classifier</h1>
    <p>Upload a concrete surface image or take a photo — the model will tell you whether it is cracked.</p>
</div>
""", unsafe_allow_html=True)


image = None
# caption = None

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
        # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded.getbuffer())
        image = tmp.name
        # image = Image.open(uploaded)
        # caption = uploaded.name


# ── Inference ─────────────────────────────────────────────────────────────────
if image:
    st.image(image, width="stretch", caption='upload image')

    with st.spinner("Analysing..."):
        model =  tf.keras.models.load_model(MODEL_PATH)
        img = tf.keras.utils.load_img(image, target_size=IMG_SIZE)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) 
        raw = model.predict(img_array, verbose=0)[0][0]

    # p_cracked = float(raw[0])
    # p_not_cracked = float(raw[1])
    # max_prob = max(p_cracked, p_not_cracked)
    # pred_idx = int(np.argmax(raw))
    label = CLASS_NAMES[int(raw <= 0.5)]
    st.write(label)
    # if max_prob <:
    #     label = "unrecognised"
    #     score = max_prob
    if label == "orange":
        label = label
        score = raw
    else:
        label = label
        score = 1 - raw
    st.write(raw)
    st.write(image)
    if label == "orange":
        st.markdown(f"""
        <div class="result-box result-cracked">
            <div class="result-label">Orange</div>
            <div class="score-mono">{score:.4f}</div>
            <div class="result-sub">Healthy probability (threshold = {THRESHOLD})</div>
        </div>
        """, unsafe_allow_html=True)
    elif label == "rottenoranges":
        st.markdown(f"""
        <div class="result-box result-safe">
            <div class="result-label">Rotten Orange</div>
            <div class="score-mono">{score:.4f}</div>
            <div class="result-sub">Rotten probability (threshold = {THRESHOLD})</div>
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
