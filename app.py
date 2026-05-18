# app.py
import streamlit as st
import pandas as pd
import time
import re
import random
import os

# Custom Imports
import data_manager
import ai_engine
import pdf_generator
from progress_manager import save_progress, load_progress, restore_progress

st.set_page_config(page_title="Trainer", page_icon=" 🩺 ", layout="wide")

# Styling Fixes
st.markdown("""
<style>
.stButton > button { width: 100%; }
.stSelectbox > div > div > select { width: 100%; }
</style>
""", unsafe_allow_html=True)

API_KEYS = [st.secrets["GENAI_KEY_1"]]
MASTER_CSV = "learning_objectives_informative_reports.csv"
NOTES_FILE = "lecture_notes.csv"
JOIN_COLUMN = "lecture_id"

EXAM_WEIGHTS = {
    "Anatomy": 0.28, "Physiology": 0.40, "Pharmacology": 0.15,
    "Nutrition": 0.06, "Microbiology": 0.06, "Immunology": 0.01, "Uncategorized": 0.04
}

# --- PROFILE MANAGEMENT ---
if 'username' not in st.session_state:
    st.session_state.username = "Default"

new_user = st.sidebar.text_input("Enter your username:", st.session_state.username)
if st.sidebar.button("Switch / Create Profile"):
    st.session_state.username = new_user.strip()
    for k in ['current_level', 'num_questions', 'missed_questions', 'current_exam', 'current_key', 'samples_df']:
        if k in st.session_state: del st.session_state[k]
    st.rerun()

active_user = st.session_state.username
st.sidebar.success(f"Logged in as: **{active_user}**")

USER_CSV = f"{active_user}_objectives.csv"
data_manager.backup_user_data(USER_CSV)

# Sync Profile System
try:
    if data_manager.synchronize_profile(MASTER_CSV, USER_CSV, JOIN_COLUMN):
        st.toast(" 🔄 Profile updated with latest curriculum structure!")
except Exception as e:
    st.error(f"Error handling profile sync: {e}")

# --- INITIALIZE SESSION STATES & RESTORE PROGRESS ---
active_user = st.session_state.get("username", "Default")
loaded_progress = load_progress(active_user)

# 1. Establish basic defaults
if 'current_level' not in st.session_state: st.session_state.current_level = 1
if 'num_questions' not in st.session_state: st.session_state.num_questions = 5
if 'last_score' not in st.session_state: st.session_state.last_score = 0
if 'missed_questions' not in st.session_state: st.session_state.missed_questions = []
if 'last_user_input' not in st.session_state: st.session_state.last_user_input = ""
if 'last_correct_key' not in st.session_state: st.session_state.last_correct_key = ""
if 'exam_submitted' not in st.session_state: st.session_state.exam_submitted = False
if 'current_categories' not in st.session_state: st.session_state.current_categories = []
if 'samples_df' not in st.session_state: st.session_state.samples_df = None
if 'current_exam' not in st.session_state: st.session_state.current_exam = ""
if 'current_key' not in st.session_state: st.session_state.current_key = []
if 'key_index' not in st.session_state: st.session_state.key_index = random.randint(0, len(API_KEYS) - 1)
if 'previous_test_data' not in st.session_state: st.session_state.previous_test_data = {}

# 2. Hand off restoration to progress_manager.py (wipes out missing fields)
if loaded_progress:
    restore_progress(st.session_state, loaded_progress)

# 3. FORCE INITIALIZE UNTRACKED CONFIGS HERE (Safe from step 2!)
if 'exam_model' not in st.session_state or not st.session_state.exam_model:
    st.session_state.exam_model = 'gemini-2.5-flash-lite'

if 'use_search' not in st.session_state:
    st.session_state.use_search = False  # Default Google Search option to False

if st.sidebar.button("Save Progress", help="Manually save your current progress"):
    # Pass the session state context and active username string explicitly
    if save_progress(st.session_state, st.session_state.username): 
        st.sidebar.success("Progress saved successfully!")
    else:
        st.sidebar.error("Failed to save progress")

# --- FILTERS AND CONTROLS ---
st.title("Trainer")
st.sidebar.header("Stats & Controls")
st.sidebar.metric("Active Level", f"{st.session_state.current_level}/50")

df_sidebar = pd.read_csv(USER_CSV)
categories = df_sidebar['category'].fillna("Uncategorized").astype(str).unique().tolist()
focus_mode = st.sidebar.selectbox("Focus Mode:", ["All Topics"] + sorted(categories))

exam_filter = st.sidebar.selectbox("Exam Filter:", ["All Exams"] + sorted(df_sidebar['exam'].fillna("Uncategorized").astype(str).unique().tolist())) if 'exam' in df_sidebar.columns else "All Exams"
system_filter = st.sidebar.selectbox("System Filter:", ["All Systems"] + sorted(df_sidebar['system'].fillna("Uncategorized").astype(str).unique().tolist())) if 'system' in df_sidebar.columns else "All Systems"
mastery_mode = "on" if st.sidebar.checkbox("Mastery Mode", value=False) else "off"

# Backup/Restore Layouts
if st.session_state.get('previous_test_data') and st.sidebar.button("Load Previous Exam", use_container_width=True):
    current_backup = {k: st.session_state.get(k) for k in ['current_exam', 'current_key', 'user_selections', 'exam_submitted', 'last_score', 'last_user_input', 'last_correct_key', 'last_user_answers_list', 'current_categories', 'samples_df']}
    for k, v in st.session_state.previous_test_data.items(): st.session_state[k] = v
    st.session_state.previous_test_data = current_backup
    st.rerun()

if st.sidebar.button("Settings", use_container_width=True):
    st.session_state.show_settings = not st.session_state.get('show_settings', False)

if st.session_state.get('show_settings', False):
    st.session_state.current_level = st.sidebar.slider("Starting Level", 1, 50, st.session_state.current_level)
    st.session_state.num_questions = st.sidebar.slider("Number of Questions", 1, 50, st.session_state.num_questions)

    model_choice = st.sidebar.radio("Speed", options=["Fast", "Slow & Smart"], index=1, horizontal=True, label_visibility="collapsed")
    st.session_state.exam_model = 'gemini-2.5-flash' if "Slow" in model_choice else 'gemini-3.1-flash-lite-preview'
    st.session_state.use_search = st.sidebar.toggle("Enable Google Search", value=False)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Knowledge Bank")
    if 'mastery_score' in df_sidebar.columns:
        df_sidebar['mastery_score'] = pd.to_numeric(df_sidebar['mastery_score'], errors='coerce').fillna(1).astype(int)
        total_objs = len(df_sidebar)
        mastered = len(df_sidebar[df_sidebar['mastery_score'] == 5])
        progress_val = mastered / total_objs if total_objs > 0 else 0
        
        st.sidebar.write(f"Mastered: {mastered} / {total_objs}")
        st.sidebar.progress(progress_val)
        st.sidebar.caption(f"{progress_val*100:.1f}% of curriculum at Mastery Level 5")

if st.sidebar.button("Reset Progress"):
    if os.path.exists("user_progress.json"): os.remove("user_progress.json")
    st.session_state.current_level, st.session_state.num_questions = 10, 10
    st.session_state.missed_questions, st.session_state.current_key, st.session_state.current_categories = [], [], []
    st.session_state.current_exam, st.session_state.samples_df = None, None
    st.session_state.exam_submitted = False
    st.rerun()

# --- EXAM GENERATION CORE ---
if st.sidebar.button("Generate New Exam"):
    if st.session_state.get('current_exam'):
        st.session_state.previous_test_data = {k: st.session_state.get(k) for k in ['current_exam', 'current_key', 'user_selections', 'exam_submitted', 'last_score', 'last_user_input', 'last_correct_key', 'last_user_answers_list', 'current_categories', 'samples_df']}
    
    # 1. Clear submission tracking and data tables immediately
    st.session_state.exam_submitted = False
    st.session_state.last_score = 0
    st.session_state.user_selections = {}
    
    # 2. Obliterate Streamlit's internal radio widget value cache
    for key in list(st.session_state.keys()):
        if key.startswith("qr_"):
            del st.session_state[key]
    
    samples_df, err = data_manager.get_weighted_sample(USER_CSV, NOTES_FILE, JOIN_COLUMN, focus_mode, exam_filter, system_filter, EXAM_WEIGHTS, mastery_mode, st.session_state.num_questions)
    if err:
        st.error(err)
        st.stop()
        
    st.session_state.samples_df = samples_df
    st.session_state.current_categories = samples_df['category'].fillna('General').tolist() if 'category' in samples_df.columns else ['General'] * len(samples_df)
    samples = (samples_df['explanation'] + "\n[Notes: " + samples_df['content'].fillna('') + "]").tolist()

    with st.spinner("Generating exam items..."):
        max_retries = 3
        for attempt in range(max_retries):
            raw_res, new_idx = ai_engine.get_blind_exam(samples, st.session_state.current_level, st.session_state.num_questions, st.session_state.exam_model, API_KEYS, st.session_state.key_index, use_search=st.session_state.use_search)
            st.session_state.key_index = new_idx
            is_valid, _ = ai_engine.validate_exam_format(raw_res, st.session_state.num_questions)
            if is_valid: break
            st.toast(f"Attempt {attempt + 1} failed validation. Retrying...")
            time.sleep(1)

        if is_valid and "[KEY:" in raw_res:
            text, key_part = raw_res.split("[KEY:")
            st.session_state.current_exam = text.strip()
            st.session_state.current_key = re.findall(r'[A-D]', key_part)
            st.rerun()
        else:
            st.error("Failed to generate a cleanly formatted exam item. Please retry.")

# --- RENDER MAIN INTERFACE EXAM FORM ---
if st.session_state.current_exam:
    st.info("Select the best answer for each clinical scenario below.")
    if pdf_generator.has_pdf_support():
        pdf_bytes = pdf_generator.create_exam_pdf(st.session_state.current_exam, st.session_state.current_key)
        if pdf_bytes: st.download_button(" 📄 Download Exam as PDF", data=pdf_bytes, file_name="practice_exam.pdf", mime="application/pdf")
    
    clean_text = re.sub(r"^(Here are|Based on|Sure|I have generated).*?\n", "", st.session_state.current_exam.strip(), flags=re.IGNORECASE)
    individual_questions = [q.strip() for q in re.split(r'\n(?=\d+\.\s)', clean_text) if q.strip()]
    
    if 'user_selections' not in st.session_state: st.session_state.user_selections = {}
    
    with st.form("grading_form"):
        for i, q_text in enumerate(individual_questions):
            st.subheader(f"Question {i+1}")
            formatted_q = re.sub(r"(\d+\.\s[^A-D]*?)(?=A\.)", r"\1<br>", q_text)
            formatted_q = re.sub(r"([A-D]\.\s[^A-D]*?)(?=[A-D]\.|$)", r"\1<br>", formatted_q).replace("\n", "<br>")
            st.markdown(re.sub(r"(<br>){2,}", "<br>", formatted_q), unsafe_allow_html=True)
            
            st.session_state.user_selections[i] = st.radio(f"Select answer {i+1}", ["A", "B", "C", "D"], key=f"qr_{i}_{abs(hash(st.session_state.current_exam)) % 1000}", horizontal=True, index=None, label_visibility="collapsed")
            st.write("---")
            
        if st.form_submit_button("Submit for Grading"):
            num_actual = len(individual_questions)
            
            # CORRECTED LOOKUP: Match the exact 'qr_' state keys generated by your radio buttons
            user_answers = []
            for i in range(num_actual):
                radio_state_key = f"qr_{i}"
                # Retrieve the selection from session_state; default to "No Answer" if left blank
                ans = st.session_state.get(radio_state_key, "No Answer")
                user_answers.append(ans)
                
            correct_key = st.session_state.current_key[:num_actual]
            
            # Track user choices for the feedback renderer to consume
            st.session_state.user_selections = {i: user_answers[i] for i in range(num_actual)}
            st.session_state.last_user_answers_list = user_answers
            
            st.session_state.last_user_input = "\n".join([f"Q{i+1}: {ans}" for i, ans in enumerate(user_answers)])
            st.session_state.last_correct_key = "\n".join([f"Q{i+1}: {ans}" for i, ans in enumerate(correct_key)])
            
            score = 0
        
            df_main = pd.read_csv(USER_CSV)
            df_main.columns = df_main.columns.str.strip().str.lower()
            
            for i, (u_ans, correct) in enumerate(zip(user_answers, correct_key)):
                # Match the question safely by its unique lecture_id string instead of integer position
                current_lecture_id = st.session_state.samples_df.iloc[i]['lecture_id']
                row_mask = df_main['lecture_id'] == current_lecture_id
                
                if row_mask.any():
                    target_idx = df_main[row_mask].index[0]
                    c_score = int(df_main.at[target_idx, 'mastery_score']) if 'mastery_score' in df_main.columns else 1
                    
                    if u_ans == correct:
                        score += 1
                        if 'mastery_score' in df_main.columns and mastery_mode == "on": 
                            df_main.at[target_idx, 'mastery_score'] = min(5, c_score + 1)
                    else:
                        if 'mastery_score' in df_main.columns and mastery_mode == "on": 
                            df_main.at[target_idx, 'mastery_score'] = max(1, c_score - 1)
                        st.session_state.missed_questions.append({
                            "question": individual_questions[i], 
                            "correct": correct, 
                            "yours": u_ans, 
                            "category": st.session_state.current_categories[i]
                        })
                else:
                    if u_ans == correct:
                        score += 1
            
            if 'mastery_score' in df_main.columns: 
                df_main['mastery_score'] = df_main['mastery_score'].astype(int)
            df_main.to_csv(USER_CSV, index=False)
            
            st.session_state.exam_submitted = True
            st.session_state.last_score = score
            
            # Level adjustments calculation
            pct = (score / num_actual) * 100
            if (num_actual - score) <= 1 or pct >= 80: 
                st.session_state.current_level = min(50, st.session_state.current_level + 1)
            elif pct < 50: 
                st.session_state.current_level = max(1, st.session_state.current_level - 1)
                
            st.rerun()

# --- GRADING INTERFACE OUTSIDE FORM ---
if st.session_state.get('exam_submitted'):
    st.subheader(f"Results: {st.session_state.last_score}/{st.session_state.num_questions}")
    with st.spinner("Instructor processing evaluation..."):
        feedback, new_idx = ai_engine.get_ai_grading(st.session_state.current_exam, st.session_state.last_user_input, st.session_state.last_correct_key, st.session_state.last_score, 'gemini-3.1-flash-lite-preview', API_KEYS, st.session_state.key_index, use_search=st.session_state.use_search)
        st.session_state.key_index = new_idx
        st.markdown(feedback)
        
    if pdf_generator.has_pdf_support():
        pdf_bytes_graded = pdf_generator.create_exam_pdf(st.session_state.current_exam, st.session_state.current_key, user_answers=st.session_state.get('last_user_answers_list', []), score=st.session_state.last_score, max_score=st.session_state.num_questions)
        if pdf_bytes_graded: st.download_button(" 📄 Download Results as PDF", data=pdf_bytes_graded, file_name="graded_exam.pdf", mime="application/pdf", key="download_graded_pdf")


# Missed Questions Bank in Sidebar
if st.session_state.missed_questions:
    # 1. THE HEATMAP (Visual)
    st.write("---")
    st.header("Weakness Heatmap")
    m_df = pd.DataFrame(st.session_state.missed_questions)
    if 'category' in m_df.columns:
        st.bar_chart(m_df['category'].value_counts(), color="#ff4b4b")

    # 2. YOUR ORIGINAL SIDEBAR EXPORT (Preserved)
    st.sidebar.markdown("---")
    st.sidebar.subheader(f"Missed Questions ({len(st.session_state.missed_questions)})")
    if st.sidebar.button("Export Mistakes to .txt"):
        with open("missed_questions.txt", "a") as f:
            f.write(f"\n\n=== WEB SESSION: {time.strftime('%Y-%m-%d %H:%M')} ===\n")
            for item in st.session_state.missed_questions:
                # Upgraded text format to include category in the file
                cat = item.get('category', 'General')
                f.write(f"\n[{cat}] {item['question']}\n[CORRECT: {item['correct']} | YOURS: {item['yours']}]\n")
        st.sidebar.success("Saved to missed_questions.txt")
