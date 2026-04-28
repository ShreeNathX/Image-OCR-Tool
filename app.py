import streamlit as st
from PIL import Image
import easyocr
import numpy as np

st.set_page_config(page_title="OCR Tool", layout="centered")
st.title("Image OCR Tool")

@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'], gpu=False)

reader = load_reader()

f = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if f:
    img = Image.open(f)
    st.image(img, caption="Uploaded Image")

    if st.button("Extract Text"):
        with st.spinner("Reading text..."):
            result = reader.readtext(np.array(img), detail=0)

        if result:
            st.subheader("Extracted Text")
            st.write(" ".join(result))
        else:
            st.warning("No text found in the image.")