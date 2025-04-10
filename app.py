import streamlit as st
from PIL import Image
import easyocr
import io
import numpy as np
import re
import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials

# --- STREAMLIT SETUP ---
st.set_page_config(page_title="NIUK FC 25 Fixtures Extractor", layout="centered")


# --- OCR SETUP ---
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
    scores = [text for text in text_lines if re.fullmatch(r"\d+", text)]
    if len(scores) >= 2:
        return scores[0], scores[1]
    return None, None


# --- GOOGLE SHEETS SETUP ---
@st.cache_resource
def load_gsheet():
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name("service.json", scope)
    client = gspread.authorize(creds)

    # Use Spreadsheet ID instead of the sheet name
    spreadsheet_id = "1RnNitFtsNaebQ_j0ed-LjOMzqPzBatjE5kuqHmgr6g8"  # Replace with your actual Spreadsheet ID
    sheet = client.open_by_key(spreadsheet_id).sheet1
    return sheet


sheet = load_gsheet()
data = sheet.get_all_records()
df = pd.DataFrame(data)

# --- STREAMLIT UI ---
st.title("🇳🇬NIUK FC 25 Fixtures Extractor🇬🇧")

# 1. Upload and OCR (Only if no score has been detected already)
if 'home_score' not in st.session_state or 'away_score' not in st.session_state:
    uploaded_file = st.file_uploader("Upload a clear picture", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Extracting score line..."):
            result = ocr_from_image(uploaded_file.read())
            extracted_text = [detection[1] for detection in result]

        if extracted_text:
            home_score, away_score = extract_home_away_scores(extracted_text)

            if home_score and away_score:
                st.session_state.home_score = home_score
                st.session_state.away_score = away_score
                st.success(f"Detected Score: {home_score} - {away_score}")
            else:
                st.warning("Could not find two valid numeric scores.")
        else:
            st.error("No text detected.")
else:
    st.success(f"Detected Score: {st.session_state.home_score} - {st.session_state.away_score}")

# 2. Select round and match to update
st.subheader("🔁 Update Fixture Score")

# Ensure that 'Round' exists in the DataFrame
df['Round'] = df['Round'].astype(str).str.strip()  # Strip any extra spaces
rounds = sorted(df['Round'].unique())
selected_round = st.selectbox("Select Round", rounds)

# Filter the DataFrame for the selected round
round_df = df[df['Round'] == selected_round]

# Debug: Display selected round and filtered DataFrame to check if filtering works
st.write(f"Selected Round: {selected_round}")
st.write("Filtered DataFrame for selected round:", round_df)

# Check if there are matches available for the selected round
if not round_df.empty:
    match_options = round_df.apply(lambda row: f"{row['Home']} vs {row['Away']}", axis=1).tolist()
    selected_match = st.selectbox("Select Match", match_options)

    # Check if the match already has a score recorded
    match_row = round_df[round_df.apply(lambda row: f"{row['Home']} vs {row['Away']}" == selected_match, axis=1)]
    if match_row.empty:
        st.warning("No match found for the selected round.")
    else:
        home_score_cell = match_row.iloc[0]['Home Score']
        away_score_cell = match_row.iloc[0]['Away Score']

        if home_score_cell and away_score_cell:  # If both home and away scores are already filled
            st.error(f"Scores have already been recorded for {selected_match}.")
        else:
            # When "Update Google Sheet" button is pressed, update the scores
            if st.button(
                    "Update Google Sheet") and 'home_score' in st.session_state and 'away_score' in st.session_state:
                # Locate the row of the selected match
                row_idx = match_row.index[0]  # Get the row index of the match
                sheet_row = row_idx + 2  # +2 because of header and 0-based index

                # Update the home and away scores in the sheet (Columns 4 and 5 for scores)
                sheet.update_cell(sheet_row, 4, str(st.session_state.home_score))  # Column D = Home Score
                sheet.update_cell(sheet_row, 5, str(st.session_state.away_score))  # Column E = Away Score

                st.success("✅ Scores updated in Google Sheet!")
else:
    st.warning("No matches found for the selected round.")
