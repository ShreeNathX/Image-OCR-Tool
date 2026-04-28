import streamlit as st
import cv2
import easyocr
import numpy as np

st.set_page_config(page_title="Robust OCR", layout="wide")
st.title("Image OCR Tool")


@st.cache_resource(show_spinner="Loading OCR model (first run may take ~2 min)...")
def load():
    return easyocr.Reader(['en'], gpu=False)


reader = load()

f = st.file_uploader("Upload image", type=['jpg', 'jpeg', 'png'])


def preprocess(img):
    """
    Improved pipeline for handwritten / cursive / signature images.

    Key changes vs original:
    - Convert to grayscale then INVERT so dark ink = white on black
      (EasyOCR detects light text on dark backgrounds more reliably)
    - Use morphological closing BEFORE threshold to reconnect broken
      cursive strokes (critical for joined handwriting)
    - Replace GaussianBlur+adaptiveThreshold with bilateral filter
      (preserves ink edges while removing noise)
    - Remove ruled notebook lines via horizontal morphological erosion
    """
    # Step 1: Grayscale
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 2: Upscale for small/low-res images
    g = cv2.resize(g, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    # Step 3: Remove horizontal ruled lines (notebook paper)
    # Erode horizontally to isolate lines, then subtract them
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    h_lines = cv2.morphologyEx(g, cv2.MORPH_OPEN, h_kernel)
    g = cv2.subtract(g, h_lines)  # remove the lines from image

    # Step 4: Bilateral filter — reduces noise but keeps ink stroke edges sharp
    g = cv2.bilateralFilter(g, 9, 75, 75)

    # Step 5: Morphological closing — reconnects broken cursive strokes
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, close_kernel)

    # Step 6: Otsu thresholding — works better than adaptive for ink on white paper
    _, g = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 7: Invert — EasyOCR expects dark background, light text
    g = cv2.bitwise_not(g)

    return g


if f:
    file_bytes = f.read()
    img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        st.error("Could not decode the image. Please upload a valid JPG or PNG.")
        st.stop()

    st.subheader("Original Image")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if st.button("Extract Text"):
        with st.spinner("Processing..."):
            proc = preprocess(img)

            # Show preprocessed image for debugging
            with st.expander("View preprocessed image (debug)"):
                st.image(proc, caption="After preprocessing")

            # EasyOCR config tuned for handwriting:
            # - paragraph=False: treat each word/stroke separately
            # - contrast_ths lowered: catch low-contrast cursive
            # - adjust_contrast higher: boost faint ink
            res = reader.readtext(
                proc,
                paragraph=False,
                contrast_ths=0.05,
                adjust_contrast=0.7,
                text_threshold=0.5,   # lower = more detections (catches cursive)
                low_text=0.3,
            )

            out = img.copy()
            txt = []

            for (b, t, p) in res:
                # Scale bounding box back to original image size (we upscaled 2x)
                scale = 0.5
                tl = tuple(map(lambda x: int(x * scale), b[0]))
                br = tuple(map(lambda x: int(x * scale), b[2]))
                cv2.rectangle(out, tl, br, (0, 255, 0), 2)
                cv2.putText(out, t, tl, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                txt.append((t, p))

        st.success("Done")

        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Annotated Result")
            st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

        with c2:
            st.subheader("Extracted Text")
            box = st.container(border=True)
            if txt:
                for i, (t, p) in enumerate(txt, 1):
                    box.markdown(f"**Text {i}:** {t}")
                    box.write(f"Confidence: {p:.2%}")
                    box.divider()
                # Full combined output
                full_text = " ".join(t for t, _ in txt)
                st.text_area("Combined output", full_text, height=100)
            else:
                box.warning(
                    "No text detected. Tips:\n"
                    "- Ensure good lighting and contrast\n"
                    "- Avoid heavy cursive if possible\n"
                    "- Try a cleaner scan/photo"
                )