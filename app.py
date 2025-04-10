import streamlit as st
import easyocr
from PIL import Image
import io
import numpy as np

# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Function to perform OCR on the uploaded image
def ocr_from_image(image_bytes):
    # Convert the bytes to a numpy array
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)
    result = reader.readtext(image_np)
    return result

# Function to extract score from specific lines (line 5 for home, line 6 for away)
def extract_home_away_scores(text_lines):
    # Make sure we have at least 6 lines
    if len(text_lines) >= 6:
        home_score = text_lines[4]  # Line 5 in the list (index 4)
        away_score = text_lines[5]  # Line 6 in the list (index 5)
        return home_score, away_score
    return None, None

# Streamlit app layout
st.title("📷 OCR Score Extractor with EasyOCR")

# Upload image
uploaded_file = st.file_uploader("Upload an image with a score line (e.g. FIFA screenshot)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Perform OCR on the image
    result = ocr_from_image(uploaded_file.read())
    extracted_text = [detection[1] for detection in result]

    # Extract home and away scores from line 5 and line 6
    home_score, away_score = extract_home_away_scores(extracted_text)

    if home_score and away_score:
        st.subheader("🏆 Extracted Game Information")
        st.markdown(f"**Home Score :** {home_score}")
        st.markdown(f"**Away Score :** {away_score}")
    else:
        st.error("⚠️ Could not find scores on line 5 and 6.")
