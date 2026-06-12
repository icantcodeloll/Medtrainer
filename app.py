import streamlit as st
import pandas as pd
from google import genai
from google.genai import types 
import time
import re
import os
from progress_manager import save_progress, load_progress
import tempfile
import random
import datetime
from zoneinfo import ZoneInfo
import zipfile
import io
import glob

try:
    from fpdf import FPDF
except ImportError:
    FPDF = None

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="Trainer", page_icon="🩺", layout="wide")

API_KEYS = [st.secrets["GENAI_KEY_1"]]#, st.secrets["GENAI_KEY_2"], st.secrets["GENAI_KEY_3"]] # (Keep your full list here)
NOTES_FILE = "lecture_notes.csv"
JOIN_COLUMN = "lecture_id"

# ==========================================
# 1A. PROFILE MANAGEMENT
# ==========================================
if 'username' not in st.session_state:
    st.session_state.username = "Default"

new_user = st.sidebar.text_input("Enter your username:", st.session_state.username)
if st.sidebar.button("Switch / Create Profile"):
    st.session_state.username = new_user.strip()
    
    # Wipe the screen clean so the new user's data can load
    keys_to_clear = ['current_level', 'num_questions', 'missed_questions', 'current_exam', 'current_key', 'samples_df']
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()

active_user = st.session_state.username
st.sidebar.success(f"Logged in as: **{active_user}**")

# ==========================================
# GLOBAL MASTER TEMPLATE CONFIGURATION
# ==========================================
CSV_FILE = "learning_objectives_informative_reports.csv" 

if not os.path.exists(CSV_FILE):
    st.error(f"Fatal Error: Master template file '{CSV_FILE}' not found!")
    st.stop()

# Models
EXAM_MODEL = 'gemini-3.1-flash-lite'
GRADER_MODEL = 'gemini-3.1-flash-lite'


# ==========================================
# 1B. EXAM WEIGHTINGS (Percentages or Relative Ratios)
# ==========================================
# Adjust these numbers to match your actual blueprint (e.g., USMLE, Board exams)
EXAM_WEIGHTS = {
    "Anatomy": 42, 
    "Physiology": 62,
    "Pharmacology": 23,
    "Nutrition": 6,
    "Microbiology": 9,
    "Immunology": 2,
    "Clinical skills": 36,
    "EBM": 14,
    "Int Med": 6
}


# Load saved progress on startup
loaded_progress = load_progress(active_user)

# Initialize Session States (with progress restoration)
if 'current_level' not in st.session_state:
    st.session_state.current_level = loaded_progress.get("current_level", 1)
if 'exam_model' not in st.session_state:
    st.session_state.exam_model = 'gemini-3.1-flash-lite'
if 'num_questions' not in st.session_state:
    st.session_state.num_questions = loaded_progress.get("num_questions", 5) 
if 'missed_questions' not in st.session_state:
    st.session_state.missed_questions = loaded_progress.get("missed_questions", [])
if 'current_exam' not in st.session_state:
    st.session_state.current_exam = loaded_progress.get("current_exam", "")
if 'current_key' not in st.session_state:
    st.session_state.current_key = loaded_progress.get("current_key", [])
if 'key_index' not in st.session_state:
    st.session_state.key_index = loaded_progress.get("key_index", random.randint(0, len(API_KEYS) - 1))
if 'current_categories' not in st.session_state:
    st.session_state.current_categories = loaded_progress.get("current_categories", [])
if 'samples_df' not in st.session_state:
    st.session_state.samples_df = loaded_progress.get("samples_df", pd.DataFrame())
if 'previous_test_data' not in st.session_state:
    st.session_state.previous_test_data = {}
if "use_search" not in st.session_state:
    st.session_state.use_search = False
if 'thinking_level' not in st.session_state:
    st.session_state.thinking_level = "MEDIUM"


if st.sidebar.button("Save Progress", help="Manually save your current progress"):
    try:
        if save_progress(st.session_state, active_user):
            st.sidebar.success("Progress saved successfully!")
        else:
            st.sidebar.error("Failed to save progress")
    except Exception as e:
        st.sidebar.error(f"Serialization Error: Could not save progress yet. ({e})")

def get_client():
    return genai.Client(api_key=API_KEYS[st.session_state.key_index])

def call_gemini_with_rotation(prompt, model_to_use, use_search=False):
    keys_tried = 0
    
    # 1. Establish tool rules: Google Search ONLY applies to Gemini 2.5 Flash
    tools = []
    if use_search and "3.5-flash" in model_to_use.lower():
        tools = [types.Tool(google_search=types.GoogleSearch())]
    
    # 2. Build configuration arguments dynamically
    config_args = {}
    
    if tools:
        config_args["tools"] = tools
        
    
    if "3.1-flash-lite" in model_to_use.lower() or "3.5-flash" in model_to_use.lower():
        # Safeguard default state if UI component hasn't rendered yet
        current_level = st.session_state.get("thinking_level", "MEDIUM")
        config_args["thinking_config"] = types.ThinkingConfig(thinking_level=current_level)

    # Pack arguments into the structural API configuration object
    generation_config = types.GenerateContentConfig(**config_args)

    # --- REST OF YOUR CONTINUOUS API LOOP ---
    while keys_tried < len(API_KEYS):
        try:
            client = get_client()
            response = client.models.generate_content(
                model=model_to_use,
                contents=prompt,
                config=generation_config
            )
            return response.text
        except Exception as e:
            if "429" in str(e):
                keys_tried += 1
                if keys_tried >= len(API_KEYS):
                    st.error("Reduce the question count.")
                    return None
                st.session_state.key_index = (st.session_state.key_index + 1) % len(API_KEYS)
                time.sleep(1)
            elif "503" in str(e):
                time.sleep(5)
            else:
                st.error(f"Error during generation: {e}")
                return None
# ==========================================
# 2. CORE LOGIC FUNCTIONS
# ==========================================
def create_exam_pdf(exam_text, answer_key, user_answers=None, score=None, max_score=None, metadata=None):
    """Generates a PDF containing the exam questions, answer key, and optionally user selections and filters."""
    if not FPDF:
        return None

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title
    pdf.set_font("Arial", "B", 12)
    if score is not None and max_score is not None:
        pdf.cell(0, 10, f"Practice Exam Results - Score: {score}/{max_score}", ln=True, align="C")
    else:
        pdf.cell(0, 10, "Practice Exam", ln=True, align="C")
    pdf.ln(2)

    # --- ADD METADATA BLOCK ---
    if metadata:
        pdf.set_font("Arial", "I", 9)
        melbourne_time = datetime.datetime.now(ZoneInfo("Australia/Melbourne")).strftime('%Y-%m-%d %H:%M:%S')
        meta_text = f"Level: {metadata.get('level', 'N/A')} | Subject: {metadata.get('subject', 'All')} | Exam Filter: {metadata.get('exam', 'All')} | System Filter: {metadata.get('system', 'All')}"
        time_text = f"Generated on (Melbourne Time): {melbourne_time}"
        pdf.cell(0, 6, meta_text, ln=True, align="C")
        pdf.cell(0, 6, time_text, ln=True, align="C")
        pdf.ln(5)
    else:
        pdf.ln(3)
    # --------------------------

    # Clean text to prevent Unicode encoding errors in FPDF
    clean_text = exam_text.replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')
    clean_text = clean_text.encode('latin-1', 'replace').decode('latin-1')

    # Print Questions
    pdf.set_font("Arial", "", 9)
    pdf.multi_cell(0, 7, clean_text)

    # Add Answer Key & User Answers on a new page
    pdf.add_page()
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Exam Summary", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Arial", "", 9)
    for i, ans in enumerate(answer_key):
        text = f"Question {i+1}: Correct Key: {ans}"
        if user_answers and i < len(user_answers):
            u_ans = user_answers[i] if user_answers[i] else "No Answer"
            match_text = " (CORRECT)" if u_ans == ans else " (INCORRECT)"
            text += f" | Your Answer: {u_ans}{match_text}"

        pdf.cell(0, 8, text, ln=True)

    # Save to temp file and return bytes
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf.output(tmp.name)
        tmp_path = tmp.name

    with open(tmp_path, "rb") as f:
        pdf_bytes = f.read()

    os.unlink(tmp_path) # Clean up temp file
    return pdf_bytes

def get_ai_grading(exam_text, user_answers, correct_key, score):
    prompt = f"""
    Here is the input:
    EXAM QUESTIONS: {exam_text}
    SCORE: {score}
    CORRECT KEY: {correct_key}
    STUDENT ANSWERS: {user_answers}
    

    You are a medical instructor. Grade the student's performance.
    
    ### GRADING PROTOCOL:
    1. Compare the student's answer for each question against the correct key.
    2. Verify with current medical knowledge.
    3. Focus ONLY on the questions the student got INCORRECT. 
    
    ### STRICT FORMATTING REQUIREMENTS:
    You MUST output your response in clean Markdown. If the student got 100%, congratulate them and provide one high-yield clinical pearl.
    Otherwise, for EVERY incorrect question, use this EXACT format:

    ### Question [Insert Question Number]
    **Your Answer:** [Letter] | **Correct Answer:** [Letter]
    
    * **Explanation:** [1-2 concise sentences explaining why the correct answer is right and the student's answer is wrong].
    * **Clinical Pearl:** [A short, high-yield tip or memory hook for board exams].
    ---
    """
    
    # Using search during grading ensures explanations match current guidelines
    return call_gemini_with_rotation(prompt, GRADER_MODEL, use_search=st.session_state.use_search)


def get_blind_exam(topics_list, level, num_questions):
    combined_content = "\n\n".join([f"Source {i+1}: {t}" for i, t in enumerate(topics_list)])

    # Difficulty calibration from intuitive basics to counterintuitive expert challenges
    if level <= 5:
        difficulty_desc = "intuitive basics - straightforward medical concepts that make logical sense"
        complexity_guide = "focus on intuitive anatomy, obvious physiology, simple definitions, core principles that follow common sense"
    elif level <= 15:
        difficulty_desc = "logical progression - clinical applications that follow standard patterns"
        complexity_guide = "include common diseases with predictable presentations, standard treatments, straightforward clinical reasoning"
    elif level <= 25:
        difficulty_desc = "complex but predictable - applied knowledge with some nuance"
        complexity_guide = "complex clinical cases with clear patterns, differential diagnosis with logical elimination, treatment with expected responses"
    elif level <= 35:
        difficulty_desc = "challenging patterns - specialized knowledge requiring deeper analysis"
        complexity_guide = "specialty-specific conditions with some counterintuitive elements, advanced therapeutics with unexpected side effects, presentations that deviate from textbook patterns"
    elif level <= 45:
        difficulty_desc = "counterintuitive expert - knowledge that defies common medical assumptions"
        complexity_guide = "subspecialty expertise where textbook knowledge fails, paradoxical treatment responses, rare conditions that present opposite to expected patterns, cutting-edge research that contradicts established dogma"
    else:  # 46-50
        difficulty_desc = "supreme counterintuition - advanced mastery of medical paradoxes and exceptions"
        complexity_guide = "multi-system integration where standard rules don't apply, latest research breakthroughs that overturn conventional wisdom, complex clinical reasoning requiring recognition of exceptions, niche subspecialty knowledge where intuitive answers are wrong, molecular-level pathophysiology that defies simple explanations, emerging treatment protocols with paradoxical mechanisms, rare disease patterns that mimic opposite conditions, advanced diagnostic challenges where the obvious answer is incorrect"

    # --- DYNAMIC RANDOM KEY GENERATION ---
    # Create an even pool of options and randomly shuffle them for this specific exam run
    options_pool = ['A', 'B', 'C', 'D'] * ((num_questions // 4) + 1)
    dynamic_keys = random.sample(options_pool, num_questions)
    formatted_key_string = ", ".join(dynamic_keys)
    # -------------------------------------

    prompt = f"""
You are a medical board examiner.
TASK: Generate EXACTLY {num_questions} Multiple Choice Questions (1 per snippet provided below).
DIFFICULTY LEVEL: {level}/50.
DIFFICULTY DESCRIPTION: {difficulty_desc}.
COMPLEXITY GUIDANCE: {complexity_guide}.

CRITICAL BIAS PREVENTION:
1. AVOID confirmation bias - Ensure each option could plausibly be correct
2. NO obvious "red herrings" - All distractors must be medically plausible
3. BALANCED difficulty - Correct answer should not be obviously easier/harder than others
4. MEDICAL ACCURACY - Verify all information with current medical guidelines
5. CLARITY over trickery - Questions should test knowledge, not reading comprehension

CRITICAL FORMATTING REQUIREMENTS:
1. START IMMEDIATELY with '1. ' followed by the question text. NO preamble.
2. ABSOLUTELY NO introductory text, explanations, or meta-commentary.
3. Each question MUST follow this EXACT format:
"X. [Question text]
A. [Option A]
B. [Option B]
C. [Option C]
D. [Option D]"
4. Every question MUST start with its number and period (e.g., '1.', '2.', '3.').
5. NO extra text, warnings, or formatting notes anywhere in the response.

CRITICAL ANSWER ASSIGNMENT:
6. You MUST design the questions so that the correct answer for each question follows this exact sequence: [KEY: {formatted_key_string}]
7. Arrange your option texts (A, B, C, D) manually so that the real, factual medical answer aligns perfectly with the matching letter in that designated sequence.
8. The VERY LAST line of your response must be exactly: [KEY: {formatted_key_string}]

CONTENT REQUIREMENTS:
7. Use the STUDY MATERIAL provided below as the base.
8. Ensure questions match difficulty level {level}/50

STUDY MATERIAL:
{combined_content}

REMEMBER: You must maintain a strict difficulty level of {level}/50 for ALL {num_questions} questions. Do not drop the complexity or become more intuitive on the later questions. Start with '1. ' immediately. No introduction. Match your questions to the exact key sequence provided, and end with the [KEY: format]. Make the questions creative
"""

    # Single call to the model using the TOGGLE'S value
    exam_text = call_gemini_with_rotation(prompt, st.session_state.exam_model, use_search=st.session_state.use_search)
    return exam_text

# =====================================================================
# PASSWORD-PROTECTED DATA PORTABILITY ENGINE
# =====================================================================
def render_data_portability_interface():
    """
    Renders password-protected download/upload tools in the sidebar to 
    safeguard user JSON progress profiles and tracking CSV matrices.
    """

    st.sidebar.markdown("---")
    st.sidebar.subheader("Admin")
    
    # 1. ENFORCE SECURITY PASSWORD
    # Change "YourSecurePassword123" to whatever admin password you prefer
    ADMIN_PASSWORD = "123456789" 
    
    user_password = st.sidebar.text_input(
        "Enter password", 
        type="password", 
        key="data_portability_pwd"
    )

    if not user_password:
        st.sidebar.caption("Enter password")
        return
        
    if user_password != ADMIN_PASSWORD:
        st.sidebar.error("Incorrect")
        return

    # --- EVERYTHING BELOW IS UNLOCKED ONLY IF THE PASSWORD IS CORRECT ---
    st.sidebar.success("Access Granted")
    st.sidebar.caption("Download data profiles before changing code, and reupload them afterward.")

# 2. SCAN AND ARCHIVE ALL PROGRESS DATA
    json_files = glob.glob("*_progress.json")
    if json_files:
        # Create an in-memory ZIP package
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for file_path in json_files:
                if os.path.exists(file_path):
                    zip_file.write(file_path, os.path.basename(file_path))
        
        zip_buffer.seek(0)
        
        timestamp_melb = datetime.datetime.now(ZoneInfo("Australia/Melbourne")).strftime('%Y%m%d_%H%M%S')
        st.sidebar.download_button(
            label="Download All User Data (.zip)",
            data=zip_buffer,
            file_name=f"backup_user_profiles_{timestamp_melb}.zip",
            mime="application/zip",
            key="download_all_data_zip"
        )
    else:
        st.sidebar.info("No user tracking files found to back up yet.")

    # 3. RESTORE AND UNPACK USER ARCHIVES
    uploaded_zip = st.sidebar.file_uploader(
        "Restore / Migrate Profiles (.zip)", 
        type=["zip"], 
        key="upload_migration_zip"
    )

    if uploaded_zip is not None:
        if st.sidebar.button("💥 Confirm Overwrite & Extract Data"):
            try:
                with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
                    file_list = zip_ref.namelist()
                    valid_extensions = ('.json', '.bak')

                    if not all(any(f.endswith(ext) for ext in valid_extensions) for f in file_list):
                        st.sidebar.error("Invalid archive payload. Must contain only .json progress profiles.")

                    # Unpack files into the execution root directory
                    zip_ref.extractall(".")
                
                st.sidebar.success(f"Successfully restored {len(file_list)} database tracking asset(s)!")
                st.sidebar.info("Please refresh or interact with the app to load profiles.")
                st.balloons()
            except Exception as e:
                st.sidebar.error(f"Migration processing error: {str(e)}")
# ==========================================
# 3. WEB INTERFACE
# ==========================================
st.title("Trainer")
st.sidebar.header("Stats & Controls")

# Move Active Level metric here
st.sidebar.metric("Active Level", f"{st.session_state.current_level}/50")

df_sidebar = pd.read_csv(CSV_FILE)

# --- FILTER MENU EXPANDER ---
with st.sidebar.expander("Filter Content", expanded=False):
    
    # --- Subject filter Checkboxes (Default: Checked) ---
    st.markdown("**Subject filter:**")
    categories = sorted(df_sidebar['category'].fillna("Uncategorized").astype(str).unique().tolist())
    subject_filter = []
    for cat in categories:
        if st.checkbox(cat, value=True, key=f"focus_{cat}"):
            subject_filter.append(cat)

    st.markdown("---")

    # --- Exam Filter Checkboxes (Default: Checked) ---
    exam_filter = []
    if 'exam' in df_sidebar.columns:
        st.markdown("**Exam Filter:**")
        exams = sorted(df_sidebar['exam'].fillna("Uncategorized").astype(str).unique().tolist())
        for ex in exams:
            if st.checkbox(ex, value=True, key=f"exam_{ex}"):
                exam_filter.append(ex)
    else:
        exam_filter = []

    st.markdown("---")

    # --- Systems Filter Checkboxes (Default: Checked) ---
    system_filter = []
    if 'system' in df_sidebar.columns:
        st.markdown("**System Filter:**")
        systems = sorted(df_sidebar['system'].fillna("Uncategorized").astype(str).unique().tolist())
        for sys in systems:
            if st.checkbox(sys, value=True, key=f"sys_{sys}"):
                system_filter.append(sys)
    else:
        system_filter = []

st.sidebar.markdown("---")


if st.sidebar.button("Generate New Exam"):
    # --- NEW: BACKUP THE CURRENT EXAM BEFORE OVERWRITING ---
    if st.session_state.get('current_exam'):
        st.session_state.previous_test_data = {
            'current_exam': st.session_state.current_exam,
            'current_key': st.session_state.current_key,
            'user_selections': st.session_state.get('user_selections', {}),
            'exam_submitted': st.session_state.get('exam_submitted', False),
            'last_score': st.session_state.get('last_score', 0),
            'last_user_input': st.session_state.get('last_user_input', ""),
            'last_correct_key': st.session_state.get('last_correct_key', ""),
            'last_user_answers_list': st.session_state.get('last_user_answers_list', []),
            'current_categories': st.session_state.get('current_categories', []),
            'samples_df': st.session_state.get('samples_df', None)
        }
    # -------------------------------------------------------
    st.session_state.exam_submitted = False  # Add this line
    st.session_state.last_score = 0
    st.session_state.user_selections = {}  # Clear previous selections
    st.session_state.last_user_answers_list = [] # Reset the answer history list
    st.session_state.samples_df = pd.DataFrame() # Wipe out old question indexes
    n = st.session_state.num_questions
    try:
        df_main = pd.read_csv(CSV_FILE)
        df_notes = pd.read_csv(NOTES_FILE)
        df = pd.merge(df_main, df_notes, on=JOIN_COLUMN, how='left')
                
        # --- Filter by subject ---
        if not subject_filter:
            st.error("Please select at least one Subject filter.")
            st.stop()
        else:
            df = df[df['category'].isin(subject_filter)]

        # --- Filter by exam ---
        if 'exam' in df.columns:
            if not exam_filter:
                st.error("Please select at least one Exam filter.")
                st.stop()
            else:
                df = df[df['exam'].isin(exam_filter)]

        # --- Filter by system ---
        if 'system' in df.columns:
            if not system_filter:
                st.error("Please select at least one System filter.")
                st.stop()
            else:
                df = df[df['system'].isin(system_filter)]
                
        # Final safety check to ensure data isn't missing
        if df.empty:
            st.error("No questions found matching your combined filter criteria. Check your CSV selections.")
            st.stop()
        
        # --- TOGGLE LOGIC ---
        if 'include' in df.columns:
            df = df[df['include'].astype(str).str.lower().str.strip() == 'y']
            
            if df.empty:
                st.error("No active objectives found. Mark some as 'y' in your CSV.")
                st.stop()
    
        # --- SMART SAMPLING (EXAM WEIGHTED ONLY) ---
        # Map exam weights to the dataframe based on category blueprints
        df['sampling_weight'] = df['category'].map(EXAM_WEIGHTS).fillna(0.05)
            

        # 3. Sample using the calculated weights
        try:
            # We use replace=False so we don't duplicate questions in the same exam
            st.session_state.samples_df = df.sample(min(n, len(df)), weights='sampling_weight', replace=False)
            st.sidebar.info("All content has been uploaded")
        except ValueError:
            # Fallback if weights math fails (e.g., all weights are zero)
            st.session_state.samples_df = df.sample(min(n, len(df)))
            st.sidebar.warning("Fallback Sampling: Standard random generation used.")
        
        samples_df = st.session_state.samples_df
        # ----------------------------

        if 'category' in samples_df.columns:
            st.session_state.current_categories = samples_df['category'].fillna('General').tolist()
        else:
            st.session_state.current_categories = ['General'] * n

        def randomize_paragraph_start(text):
            """Splits a paragraph into sentences and randomizes their order or starting point."""
            if not text or not isinstance(text, str):
                return ""
            
            # Split text into sentences using a simple regex (handles ., !, ?)
            sentences = re.split(r'(?<=[.!?])\s+', text.strip())
            if len(sentences) <= 1:
                return text # Return as-is if it's only one sentence
            
            # Option A: Completely shuffle the sentences
            #random.shuffle(sentences)
            #return " ".join(sentences)

            # ALTERNATIVE (Option B): If you prefer to keep the sequential order but just want 
            # to start at a random sentence and wrap around to the beginning, uncomment below:
            start_idx = random.randint(0, len(sentences) - 1)
            rotated_sentences = sentences[start_idx:] + sentences[:start_idx]
            return " ".join(rotated_sentences)

        # Apply the sentence-level randomization to your data columns
        randomized_explanations = samples_df['explanation'].fillna('').astype(str).apply(randomize_paragraph_start)
        randomized_content = samples_df['content'].fillna('').astype(str).apply(randomize_paragraph_start)
        randomized_flashcards = samples_df['flashcards'].fillna('').astype(str).apply(randomize_paragraph_start)

        # Compile into the final list for the prompt
        samples = (randomized_explanations + "\n[Notes: " + randomized_content + randomized_flashcards + "]").tolist()
        
        with st.spinner(f"Generating {n} questions at Level {st.session_state.current_level}..."):
            raw_response = ""
            raw_response = get_blind_exam(samples, st.session_state.current_level, n)

            if "[KEY:" in raw_response:
                # Use a split that keeps the questions separate from the key
                text, key_part = raw_response.split("[KEY:")
                
                # CLEANING: Remove the key section from the visible text 
                # so it doesn't show up in the last radio button question
                st.session_state.current_exam = text.strip() 
                
                st.session_state.current_key = re.findall(r'[A-D]', key_part)
                st.rerun()
            else:
                st.error(f"Failed to generate a perfectly formatted exam after {max_retries} attempts. Please click generate again.")
    except Exception as e:
        st.error(f"File Error: Ensure {CSV_FILE} and {NOTES_FILE} are in the folder. ({e})")

st.sidebar.markdown("---")


# --- NEW: RESTORE BACKUP BUTTON ---
if st.session_state.get('previous_test_data'):
    if st.sidebar.button("Load Previous Exam", help="Accidentally clicked generate? Restore the last exam.", use_container_width=True):
        
        # Take a snapshot of the active exam before swapping, so you can toggle back and forth!
        current_backup = {}
        if st.session_state.get('current_exam'):
            current_backup = {
                'current_exam': st.session_state.current_exam,
                'current_key': st.session_state.current_key,
                'user_selections': st.session_state.get('user_selections', {}),
                'exam_submitted': st.session_state.get('exam_submitted', False),
                'last_score': st.session_state.get('last_score', 0),
                'last_user_input': st.session_state.get('last_user_input', ""),
                'last_correct_key': st.session_state.get('last_correct_key', ""),
                'last_user_answers_list': st.session_state.get('last_user_answers_list', []),
                'current_categories': st.session_state.get('current_categories', []),
                'samples_df': st.session_state.get('samples_df', None)
            }
        
        # Load the backup into the live view
        backup = st.session_state.previous_test_data
        st.session_state.current_exam = backup.get('current_exam')
        st.session_state.current_key = backup.get('current_key')
        st.session_state.user_selections = backup.get('user_selections', {})
        st.session_state.exam_submitted = backup.get('exam_submitted', False)
        st.session_state.last_score = backup.get('last_score', 0)
        st.session_state.last_user_input = backup.get('last_user_input', "")
        st.session_state.last_correct_key = backup.get('last_correct_key', "")
        st.session_state.last_user_answers_list = backup.get('last_user_answers_list', [])
        st.session_state.current_categories = backup.get('current_categories', [])
        st.session_state.samples_df = backup.get('samples_df', None)
        
        # Make the old current exam the new backup
        st.session_state.previous_test_data = current_backup if current_backup else {}
            
        st.rerun()
# ----------------------------------

# Settings button for sliders
if st.sidebar.button("Settings", use_container_width=True):
    if 'show_settings' not in st.session_state:
        st.session_state.show_settings = False
    st.session_state.show_settings = not st.session_state.show_settings

# Show sliders only when settings is expanded
if st.session_state.get('show_settings', False):
    st.session_state.current_level = st.sidebar.slider("Starting Level", 1, 50, st.session_state.current_level)
    st.session_state.num_questions = st.sidebar.slider("Number of Questions", 1, 50, st.session_state.num_questions)
    st.session_state.thinking_level = st.sidebar.selectbox(
        "Gemini Thinking Level",
        options=["MINIMAL", "LOW", "MEDIUM", "HIGH"],
        index=["MINIMAL", "LOW", "MEDIUM", "HIGH"].index(st.session_state.thinking_level),
        help="Control how deeply the model deliberates before generating questions or grading."
    )

    st.sidebar.markdown("---")
    
    # --- NEW: Double-Sided Model Switch ---
    st.sidebar.markdown("**Speed:**")
    model_choice = st.sidebar.radio(
        label="Speed",
        label_visibility="collapsed", # Hides the label so it just looks like a switch
        options=["Fast", "Slow & Smart"],
        index=1 if st.session_state.get('exam_model', 'gemini-3.5-flash') == 'gemini-3.5-flash' else 0,
        horizontal=True, # This forces them side-by-side like a double switch!
        help="Fast uses Flash-Lite. Slow & Smart uses Flash."
    )

    # Update memory
    st.session_state.exam_model = 'gemini-3.5-flash' if "Slow" in model_choice else 'gemini-3.1-flash-lite'

    st.sidebar.markdown("**Grounding:**")
    st.session_state.use_search = st.sidebar.toggle(
        label="Enable Google Search",
        value=False,  # Sets the default state to ON
        help="Turn off on fast mode."
    )

    st.sidebar.markdown("---")
    if st.sidebar.button("Reset Progress", help="Clear all saved progress and reset to defaults"):
    # Target the specific user's progress file instead of a global file
        user_specific_progress = f"{active_user}_progress.json"
        if os.path.exists(user_specific_progress):
            os.remove(user_specific_progress)
            # Reset session state to defaults
            st.session_state.current_level = 10
            st.session_state.num_questions = 10
            st.session_state.missed_questions = []
            st.session_state.current_exam = None
            st.session_state.current_key = []
            st.session_state.last_score = 0
            st.session_state.user_selections = {}
            st.session_state.exam_submitted = False
            st.session_state.current_categories = []
            st.session_state.samples_df = None
            st.sidebar.success("Progress reset successfully!")
            st.rerun()
        else:
            st.sidebar.info("No saved progress to reset")
    
    st.sidebar.markdown("---")
    render_data_portability_interface()


# Display the Exam
if st.session_state.current_exam:
    st.info("Select the best answer for each clinical scenario below.")
    
    # --- ADDED: PDF Download Button ---
    if FPDF:
        # Package the metadata dictionary dynamically
        current_metadata = {
            "level": st.session_state.current_level,
            "subject": subject_filter if 'subject_filter' in locals() else "All Subjects",
            "exam": exam_filter if 'exam_filter' in locals() else "All Exams",
            "system": system_filter if 'system_filter' in locals() else "All Systems"
        }

        pdf_bytes = create_exam_pdf(
            st.session_state.current_exam,
            st.session_state.current_key,
            metadata=current_metadata
        )
        if pdf_bytes:
            st.download_button(
                label="  Download Exam as PDF",
                data=pdf_bytes,
                file_name="practice_exam.pdf",
                mime="application/pdf"
            )

    # --- NEW: TXT Download Button (Zero Blank Lines between Questions) ---
    if st.session_state.current_exam:
        # Compile clean plain text with regular expressions
        raw_exam_text = st.session_state.current_exam.strip()
        
        # 1. Collapse multi-newlines between Option D and the following question number down to a single line break
        # This matches Option D, grabs all trailing newlines/whitespace, and replaces them with a single \n
        clean_exam_text = re.sub(r'(D\.\s.*?)\n+(?=\d+\.\s)', r'\1\n', raw_exam_text)
        
        # 2. Package the metadata dictionary dynamically
        meta_subject = subject_filter if 'subject_filter' in locals() else "All Subjects"
        meta_exam = exam_filter if 'exam_filter' in locals() else "All Exams"
        meta_system = system_filter if 'system_filter' in locals() else "All Systems"
        
        melbourne_now = datetime.datetime.now(ZoneInfo("Australia/Melbourne")).strftime('%Y-%m-%d %H:%M:%S')
        txt_content = (
            f"Practice Exam (Level {st.session_state.current_level}/50)\n"
            f"Filters - Subject: {meta_subject} | Exam: {meta_exam} | System: {meta_system}\n"
            f"Generated on (Melbourne Time): {melbourne_now}\n"
            f"----------------------------------------------------------------------\n\n"
            f"{clean_exam_text}\n"
        )
        
        st.download_button(
            label="Download Exam as TXT",
            data=txt_content,
            file_name="practice_exam.txt",
            mime="text/plain",
            key="download_exam_txt_main"
        )
    # ----------------------------------
    # 1. CLEANING: Remove introductory fluff and trailing keys
    clean_text = st.session_state.current_exam.strip()
    # Remove common AI intros like "Here are your questions..."
    clean_text = re.sub(r"^(Here are|Based on|Sure|I have generated).*?\n", "", clean_text, flags=re.IGNORECASE)

    # 2. SPLITTING: Look for "1. ", "2. ", etc. at the START of a line only
    # This prevents it from splitting on "1." inside a sentence
    raw_questions = re.split(r'\n(?=\d+\.\s)', clean_text)
    
    # Remove any empty strings resulting from the split
    individual_questions = [q.strip() for q in raw_questions if q.strip()]

    if 'user_selections' not in st.session_state:
        st.session_state.user_selections = {}

    with st.form("grading_form"):
        for i, q_text in enumerate(individual_questions):
            st.subheader(f"Question {i+1}")
            
            # Enhanced formatting: Add line breaks after questions and options
            # 1. Add single HTML line break after question text before options
            formatted_q = re.sub(r"(\d+\.\s[^A-D]*?)(?=A\.)", r"\1<br>", q_text)
            # 2. Add single HTML line break after each option (A., B., C., D.)
            formatted_q = re.sub(r"([A-D]\.\s[^A-D]*?)(?=[A-D]\.|$)", r"\1<br>", formatted_q)
            # 3. Replace newlines with HTML breaks for better rendering
            formatted_q = formatted_q.replace("\n", "<br>")
            # 4. Clean up any multiple consecutive breaks to maximum 1
            formatted_q = re.sub(r"(<br>){2,}", "<br>", formatted_q)
            
            # Use markdown with HTML allowed for proper line breaks
            st.markdown(formatted_q, unsafe_allow_html=True)
            
            # This makes the radio buttons cleaner - use exam content for unique key
            exam_content = st.session_state.get('current_exam', '')
            exam_content_hash = hash(exam_content) if exam_content else 0
            st.session_state.user_selections[i] = st.radio(
                label=f"Select answer for Question {i+1}", # Provide a real label
                options=["A", "B", "C", "D"],
                key=f"q_radio_{i}_{abs(exam_content_hash) % 1000}",
                horizontal=True,
                index=None,
                label_visibility="collapsed" # This hides the label visually
            )
            st.write("---")
        
        submitted = st.form_submit_button("Submit for Grading")
        if submitted:
            # Use actual number of questions from current exam
            num_actual_questions = len(raw_questions)

            # Convert dictionary to a sorted list of answers
            user_answers = [st.session_state.user_selections[i] for i in range(num_actual_questions)]
            user_input = "\n".join([f"Q{i+1}: {ans if ans else 'No Answer'}" for i, ans in enumerate(user_answers)])

            # Use only the first num_actual_questions answers from current_key
            correct_key = st.session_state.current_key[:num_actual_questions]
            correct_key_formatted = "\n".join([f"Q{i+1}: {ans}" for i, ans in enumerate(correct_key)])
            
            # Save these formatted versions to session state
            st.session_state.last_user_input = user_input
            st.session_state.last_correct_key = correct_key_formatted
            st.session_state.last_user_answers_list = user_answers 

            if len(user_answers) != len(correct_key):
                st.error(f"Mismatch: The exam has {len(correct_key)} questions, but you entered {len(user_answers)} answers. Please fix your input.")
                st.stop() 

            # --- SIMPLIFIED GRADING SYSTEM (Now properly nested inside 'if submitted') ---
            score = 0
            for i, q_text in enumerate(individual_questions):
                if i >= len(user_answers):
                    break
                u_ans = user_answers[i]
                correct = correct_key[i] if i < len(correct_key) else None
                if u_ans == correct:
                    score += 1
                else:
                    if i < len(individual_questions):
                        st.session_state.missed_questions.append({
                            "question": individual_questions[i].strip(),
                            "correct": correct,
                            "yours": u_ans,
                            "category": st.session_state.current_categories[i] if i < len(st.session_state.current_categories) else "General",
                        })

            # Save state so the feedback stays visible after submission
            st.session_state.exam_submitted = True
            st.session_state.last_score = score
            
            # Update Level based on performance
            percentage_correct = (score / num_actual_questions) * 100
            questions_wrong = num_actual_questions - score

            # Level up if: only 1 question wrong OR 90%+ correct
            if questions_wrong <= 1 or percentage_correct >= 90:
                st.session_state.current_level = min(50, st.session_state.current_level + 1)
                st.success(f"Level Up! Now at Level {st.session_state.current_level}")
            # Level down if: less than 60% correct
            elif percentage_correct <= 60:
                st.session_state.current_level = max(1, st.session_state.current_level - 1)
                st.warning(f"Level Down. Now at Level {st.session_state.current_level}")
            else:
                st.info(f"Score: {score}/{st.session_state.num_questions} ({percentage_correct:.0f}%) - Level maintained")
            
            # Force a rerun to clean up widget displays and lock in state
            st.rerun()

    # --- 2. THE FEEDBACK (Fully outside the st.form block) ---
    if st.session_state.get('exam_submitted'):
        st.subheader(f"Results: {st.session_state.last_score}/{st.session_state.num_questions}")

        with st.spinner("Don't forget to click save progress after!!"):
            feedback = get_ai_grading(
                st.session_state.current_exam,
                st.session_state.last_user_input,
                st.session_state.last_correct_key,
                st.session_state.last_score
            )
        st.markdown(feedback)
        st.write("---")

        # --- Download Graded Exam Button ---
        # --- Download Graded Exam Button ---
        if FPDF:
            # Package the metadata dictionary dynamically here as well
            current_metadata = {
                "level": st.session_state.current_level,
                "subject": subject_filter if 'subject_filter' in locals() else "All Subjects",
                "exam": exam_filter if 'exam_filter' in locals() else "All Exams",
                "system": system_filter if 'system_filter' in locals() else "All Systems"
            }

            pdf_bytes_graded = create_exam_pdf(
                st.session_state.current_exam,
                st.session_state.current_key,
                user_answers=st.session_state.get('last_user_answers_list', []),
                score=st.session_state.last_score,
                max_score=st.session_state.num_questions,
                metadata=current_metadata
            )
        

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
    # Prepare the missed questions text content dynamically in memory
    melbourne_mistakes_time = datetime.datetime.now(ZoneInfo("Australia/Melbourne")).strftime('%Y-%m-%d %H:%M')
    export_text = f"=== WEB SESSION (Melbourne Time): {melbourne_mistakes_time} ===\n"    
    for item in st.session_state.missed_questions:
        cat = item.get('category', 'General')
        export_text += f"\n[{cat}] {item['question']}\n[CORRECT: {item['correct']} | YOURS: {item['yours']}]\n"

    # Offer the file directly as a local browser download
    st.sidebar.download_button(
        label="Download Mistakes to .txt",
        data=export_text,
        file_name=f"{active_user}_missed_questions.txt",
        mime="text/plain",
        key=f"download_mistakes_{active_user}"
    )

