import streamlit as st
import numpy as np
import tensorflow as tf
import tempfile

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH  = "mobilenetv3_transfer.keras"
IMG_SIZE    = (224, 224)
CLASS_NAMES = ["Healthy", "Vitiligo"]   # must match training class order

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Skin Classifier",
    layout="centered",
)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>Skin Classifier</h1>
    <p>Upload a picture of the skin so it can be analyzed.</p>
</div>
""", unsafe_allow_html=True)


image = None

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


# ── Inference ─────────────────────────────────────────────────────────────────
if image:
    st.image(image, width="stretch", caption='upload image')

    with st.spinner("Analysing..."):
        model =  tf.keras.models.load_model(MODEL_PATH)
        img = tf.keras.utils.load_img(image, target_size=IMG_SIZE)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) 
        raw = model.predict(img_array, verbose=0)[0][0]

   
    label = CLASS_NAMES[int(raw <= 0.5)]
  
    if label == "Healthy":
        label = label
        score = raw
    else:
        label = label
        score = 1 - raw
        
    
    if label == "Healthy":
        st.markdown(f"""
        <div class="result-box result-cracked">
            <div class="result-label">Orange</div>
            <div class="score-mono">{score:.4f}</div>
            <div class="result-sub">Healthy probability </div>
        </div>
        """, unsafe_allow_html=True)
    elif label == "Vitiligo":
        st.markdown(f"""
        <div class="result-box result-safe">
            <div class="result-label">Rotten Orange</div>
            <div class="score-mono">{score:.4f}</div>
            <div class="result-sub">Vitiligo probability </div>
        </div>
        """, unsafe_allow_html=True)
    


st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown(
    '<div class="footer">Custom CNN · Trained on Fruit Dataset · 96.47% accuracy</div>',
    unsafe_allow_html=True,
)
