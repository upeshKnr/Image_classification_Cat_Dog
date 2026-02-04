import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json

st.set_page_config(page_title="Cat vs Dog AI", page_icon="🐾", layout="centered")

# Load Model
model = tf.keras.models.load_model("model/cat_dog_model.keras")

# Labels
with open("model/labels.json") as f:
    labels = json.load(f)

classes = list(labels.keys())

# ----------- CUSTOM CSS -----------

st.markdown("""
<style>
.stApp {
    background: radial-gradient(circle at top, #0f172a, #020617);
}

.card {
    background: rgba(255,255,255,0.06);
    padding: 40px;
    border-radius: 25px;
    box-shadow: 0 0 40px rgba(0,0,0,0.7);
    backdrop-filter: blur(15px);
    border: 1px solid rgba(255,255,255,0.1);
}

.title {
    text-align:center;
    font-size:48px;
    font-weight:800;
    color:#e5e7eb;
}

.badge {
    padding:10px 20px;
    border-radius:20px;
    font-weight:bold;
    font-size:18px;
    display:inline-block;
}
</style>
""", unsafe_allow_html=True)

# ---------- END CSS ----------


# st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<p class="title">🐾 Cat vs Dog AI</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg","png","jpeg"])

if uploaded_file:
    col1, col2 = st.columns(2)

    with col1:
        image = Image.open(uploaded_file).resize((150,150))
        st.image(image, width="stretch")

    img_array = np.array(image)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        label = "DOG 🐶"
        conf = prediction
        color = "#16a34a"
    else:
        label = "CAT 🐱"
        conf = 1 - prediction
        color = "#2563eb"

    with col2:

    # mini columns for badge + button
     mini1, mini2 = st.columns([1,1])

    with mini1:
        st.markdown(
            f'<div class="badge" style="background:{color};">{label}</div>',
            unsafe_allow_html=True
        )

    with mini2:
        show_conf = st.button("🔍 Confidence")

    if show_conf:
        st.progress(float(conf))
        st.caption(f"Confidence: {conf*100:.2f}%")

st.markdown('</div>', unsafe_allow_html=True)

