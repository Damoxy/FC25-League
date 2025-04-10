import streamlit as st
from PIL import Image
import easyocr
import io
import numpy as np
import re
import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
import base64

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
    encoded_service_json = st.secrets["general"]["google_service_json"]
    decoded_service_json = base64.b64decode(encoded_service_json)
    with open("service.json", "wb") as f:
        f.write(decoded_service_json)

    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name("service.json", scope)
    client = gspread.authorize(creds)

    spreadsheet_id = st.secrets["general"]["spreadsheet_id"]
    sheet = client.open_by_key(spreadsheet_id).sheet1
    return sheet

sheet = load_gsheet()
data = sheet.get_all_records()
df = pd.DataFrame(data)

# --- STREAMLIT UI ---
st.title("🇳🇬NIUK FC 25 Fixtures Extractor🇬🇧")

# 1. Upload and OCR
flagged_for_admin = False

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

                score_validity = st.radio(
                    "Is this detected score line correct?",
                    ('Yes', 'No')
                )

                if score_validity == 'No':
                    st.warning("You may want to re-upload or manually verify the image.")
                    if st.button("🚩 Flag Admin"):
                        flagged_for_admin = True
                        st.info("Flag has been sent to admin (functionality coming soon).")
                    st.stop()  # Stop the execution if the score is flagged

            else:
                st.warning("Could not find two valid numeric scores.")
        else:
            st.error("No text detected.")
else:
    st.success(f"Detected Score: {st.session_state.home_score} - {st.session_state.away_score}")

# 2. Select round and match to update
st.subheader("🔁 Update Fixture Score")

df['Round'] = df['Round'].astype(str).str.strip()
rounds = [round for round in df['Round'].unique() if round.isdigit()]
selected_round = st.selectbox("Select Round", rounds)

round_df = df[df['Round'] == selected_round]

st.write(f"ROUND {selected_round} Fixtures:")
st.dataframe(round_df)

if not round_df.empty:
    match_options = round_df.apply(lambda row: f"{row['Home']} vs {row['Away']}", axis=1).tolist()
    selected_match = st.selectbox("Select Match", match_options)

    match_row = round_df[round_df.apply(lambda row: f"{row['Home']} vs {row['Away']}" == selected_match, axis=1)]
    if match_row.empty:
        st.warning("No match found for the selected round.")
    else:
        home_team = match_row.iloc[0]['Home']
        away_team = match_row.iloc[0]['Away']
        home_score_cell = match_row.iloc[0]['Home Score']
        away_score_cell = match_row.iloc[0]['Away Score']

        if home_score_cell and away_score_cell:
            st.error(f"Scores have already been recorded for {selected_match}.")
        else:
            # ✅ Show score line with team names
            st.markdown("### 📊 Confirm Score Direction")
            st.info(f"Detected: **{home_team} {st.session_state.home_score} - {st.session_state.away_score} {away_team}**  \nSwapped: **{home_team} {st.session_state.away_score} - {st.session_state.home_score} {away_team}**")

            score_direction = st.radio(
                "Which score line is correct for this fixture?",
                ('Yes, keep as shown (Home - Away)', 'No, swap the scores')
            )

            if st.button("Update Google Sheet"):
                if score_direction == 'Yes, keep as shown (Home - Away)':
                    final_home_score = st.session_state.home_score
                    final_away_score = st.session_state.away_score
                else:
                    final_home_score = st.session_state.away_score
                    final_away_score = st.session_state.home_score

                row_idx = match_row.index[0]
                sheet_row = row_idx + 2  # +2 for header and 0-index

                sheet.update_cell(sheet_row, 4, str(final_home_score))  # Column D
                sheet.update_cell(sheet_row, 5, str(final_away_score))  # Column E

                st.success(f"✅ Scores updated: {home_team} {final_home_score} - {final_away_score} {away_team}")
else:
    st.warning("No matches found for the selected round.")
