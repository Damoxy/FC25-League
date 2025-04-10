import streamlit as st
from PIL import Image
import easyocr
import io
import numpy as np

# Must be the FIRST Streamlit command
st.set_page_config(page_title="OCR Score Extractor", layout="centered")

# Lazy load the OCR reader
@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'], gpu=False)

reader = load_reader()

def resize_image(image, max_width=800):
    if image.width > max_width:
        ratio = max_width / float(image.width)
        new_height = int((float(image.height) * float(ratio)))
        return image.resize((max_width, new_height), Image.Resampling.LANCZOS)
    return image

def ocr_from_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = resize_image(image)
    image_np = np.array(image)
    result = reader.readtext(image_np)
    return result

def extract_home_away_scores(text_lines):
    if len(text_lines) >= 6:
        return text_lines[4], text_lines[5]
    return None, None

# Streamlit UI
st.title("📷 OCR Score Extractor")

uploaded_file = st.file_uploader("Upload a screenshot (e.g. FIFA score screen)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Extracting text..."):
        result = ocr_from_image(uploaded_file.read())
        extracted_text = [detection[1] for detection in result]

    if extracted_text:
        home_score, away_score = extract_home_away_scores(extracted_text)

        st.subheader("🏆 Detected Scores")
        if home_score and away_score:
            st.markdown(f"**Home Score:** {home_score}")
            st.markdown(f"**Away Score:** {away_score}")
        else:
            st.warning("Could not find scores in the expected lines.")
    else:
        st.error("No text detected.")
