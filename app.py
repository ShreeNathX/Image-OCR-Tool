import streamlit as st
import cv2
import easyocr
import numpy as np

st.set_page_config(page_title="Robust OCR", layout="wide")
st.title("Image OCR Tool")

@st.cache_resource
def load(): return easyocr.Reader(['en'])
reader = load()

f = st.file_uploader("Upload image", ['jpg', 'jpeg', 'png'])

def preprocess(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=1.5, fy=1.5)         # handle small/stretched text
    g = cv2.GaussianBlur(g, (5, 5), 0)              # noise removal
    g = cv2.adaptiveThreshold(g, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2)                    # contrast boost
    return g

if f:
    # FIX 1: width='stretch' is invalid — removed it (uses full column width by default)
    img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), 1)
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if st.button("Extract Text"):
        with st.spinner("Processing..."):
            proc = preprocess(img)
            res = reader.readtext(proc)

            out = img.copy()
            txt = []

            for (b, t, p) in res:
                tl, br = tuple(map(int, b[0])), tuple(map(int, b[2]))
                cv2.rectangle(out, tl, br, (0, 255, 0), 2)
                cv2.putText(out, t, tl, 0, 0.8, (0, 0, 255), 2)
                txt.append((t, p))

        # FIX 2: These lines are now correctly indented inside the if st.button block
        st.success("Done")
        c1, c2 = st.columns(2)

        with c1:
            st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

        with c2:
            box = st.container(border=True)
            for i, (t, p) in enumerate(txt, 1):
                box.markdown(f"**Text {i}:** {t}")
                box.write(f"Confidence: {p:.2%}")
                box.divider()
            if not txt:
                box.warning("No text found")