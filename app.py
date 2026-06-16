import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import random
import datetime
from zoneinfo import ZoneInfo
import re
import os
import glob
import io
import zipfile
import tempfile
import time
import atexit
from google import genai
from google.genai import types
from progress_manager import save_progress, load_progress
from fpdf import FPDF
import shutil

def initialize_pwa_assets():
    """
    Automated hook ensuring manifest, service worker, and app tags are mapped
    directly into Streamlit Cloud's native frontend directory structure.
    """
    try:
        # Resolve Streamlit's underlying static frontend webroot folder path
        streamlit_static_path = os.path.join(os.path.dirname(st.__file__), "static")
        
        # Manifest assets that need to be public-facing at the root domain
        target_assets = ["manifest.json", "sw.js", "app-icon.png"]
        
        for asset in target_assets:
            if os.path.exists(asset):
                shutil.copy(asset, os.path.join(streamlit_static_path, asset))
        
        # Inject the mandatory PWA mobile tags directly into the core index.html head
        index_html_path = os.path.join(streamlit_static_path, "index.html")
        with open(index_html_path, "r") as f:
            html_content = f.read()
            
        if "manifest.json" not in html_content:
            pwa_tags = """
            <link rel="manifest" href="./manifest.json">
            <meta name="apple-mobile-web-app-capable" content="yes">
            <meta name="apple-mobile-web-app-status-bar-style" content="default">
            <link rel="apple-touch-icon" href="./app-icon.png">
            <script>
              if ('serviceWorker' in navigator) {
                navigator.serviceWorker.register('./sw.js');
              }
            </script>
            """
            # Splice our custom metadata right inside the main document head boundary
            updated_content = html_content.replace("</head>", f"{pwa_tags}</head>")
            with open(index_html_path, "w") as f:
                f.write(updated_content)
    except Exception:
        # Fails silently to prevent execution disruptions during local development environments
        pass

# Trigger asset pipeline validation prior to UI layout assembly
initialize_pwa_assets()

st.set_page_config(page_title="Trainer", page_icon="🩺", layout="wide")

API_KEYS = [st.secrets["GENAI_KEY_1"]]#, st.secrets["GENAI_KEY_2"], st.secrets["GENAI_KEY_3"]] # (Keep your full list here)
CSV_FILE = "learning_objectives_informative_reports.csv" 
NOTES_FILE = "lecture_notes.csv"
JOIN_COLUMN = "lecture_id"
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
# Models
EXAM_MODEL = 'gemini-3.1-flash-lite'
GRADER_MODEL = 'gemini-3.1-flash-lite'


# ==========================================
# 0. MULTI-PAGE CONFIGURATION & NAVIGATION
# ==========================================
def initialize_app(active_user, force_reset=False):
    """
    Handles all initial state configurations, progress restoration, 
    and systemic fallback settings in one central runtime hook.
    """
    # Load saved progress dynamically from local disk storage
    if force_reset:
        loaded_progress = {}
    else:
        loaded_progress = load_progress(active_user) # Existing Line 4595

    # Core parameters mapping dictionary 
    defaults = {
        "current_level": loaded_progress.get("current_level", 1),
        "exam_model": loaded_progress.get("exam_model", 'gemini-3.1-flash-lite'),
        "num_questions": loaded_progress.get("num_questions", 5),
        "semester": loaded_progress.get("semester", "y2s1"),
        "missed_questions": loaded_progress.get("missed_questions", []),
        "exam_history": loaded_progress.get("exam_history", []),
        "current_exam": loaded_progress.get("current_exam", ""),
        "current_key": loaded_progress.get("current_key", []),
        "key_index": min(loaded_progress.get("key_index", 0), max(0, len(API_KEYS) - 1)) if len(API_KEYS) > 0 else 0,
        "current_categories": loaded_progress.get("current_categories", []),
        "previous_test_data": {},
        "use_search": False,
        "thinking_level": "MEDIUM",
        "exam_submitted": False,
        "last_score": 0,
        "user_selections": {},
        "last_user_answers_list": [],
        "show_settings": False,
        "current_page": "Exam Trainer"
    }

    # Bulk assign missing parameters into active memory layout
    for key, value in defaults.items():
        if key not in st.session_state or force_reset:
            st.session_state[key] = value

    # Specialized block handler for reconstructing structure representations (DataFrames)
    if 'samples_df' not in st.session_state or force_reset:
        saved_samples = loaded_progress.get("samples_df", None)
        if isinstance(saved_samples, pd.DataFrame):
            st.session_state.samples_df = saved_samples
        elif isinstance(saved_samples, (dict, list)):
            st.session_state.samples_df = pd.DataFrame(saved_samples)
        else:
            st.session_state.samples_df = pd.DataFrame()

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


def render_data_portability_interface():
    """
    Renders password-protected download/upload tools in the sidebar to 
    safeguard user JSON progress profiles and tracking CSV matrices.
    """

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

# Define the view functions
def render_trainer_page():
    global CSV_FILE
    # ==========================================
    # 0. INITIALIZATION ENGINE
    # ==========================================

    def save_on_tab_close():
        """
        Background hook that intercepts Streamlit's session cleanup routine.
        If the tab is closed, this function executes on the server right before 
        the session memory is wiped, committing the latest state to disk.
        """
        try:
            # Check if we have an active username and valid exam data to back up
            if 'username' in st.session_state and st.session_state.username:
                active_user = st.session_state.username
                
                # Construct a clean dictionary snapshot of the active workspace
                state_snapshot = {
                    "current_level": st.session_state.get("current_level", 1),
                    "exam_model": st.session_state.get("exam_model", 'gemini-3.1-flash-lite'),
                    "num_questions": st.session_state.get("num_questions", 5),
                    "missed_questions": st.session_state.get("missed_questions", []),
                    "exam_history": st.session_state.get("exam_history", []),
                    "current_exam": st.session_state.get("current_exam", ""),
                    "current_key": st.session_state.get("current_key", []),
                }
                
                # Safely serialize the dataframe to standard records so the JSON manager handles it cleanly
                samples_df = st.session_state.get("samples_df", pd.DataFrame())
                if not samples_df.empty:
                    state_snapshot["samples_df"] = samples_df.to_dict(orient="records")
                else:
                    state_snapshot["samples_df"] = []
                    
                # Fire the save function directly to the disk
                save_progress(state_snapshot, active_user)
        except Exception:
            pass # Silently pass to ensure the server thread terminates smoothly

    # Register the background cleanup hook with the Python runtime engine
    atexit.register(save_on_tab_close)

    # ==========================================
    # 1. SETUP & CONFIGURATION
    # ==========================================
    st.set_page_config(page_title="Trainer", page_icon="🩺", layout="wide")



    # ==========================================
    # 1A. PROFILE MANAGEMENT
    # ==========================================
    if 'username' not in st.session_state:
        st.session_state.username = "Default"

    new_user = st.sidebar.text_input("Enter your username:", st.session_state.username)
    if st.sidebar.button("Switch / Create Profile"):
        st.session_state.username = new_user.strip()
        
        # Wipe the screen clean so the new user's data can load
        keys_to_clear = ['current_level', 'num_questions', 'missed_questions', 'exam_history', 'current_exam', 'current_key', 'samples_df']
        for k in keys_to_clear:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

    active_user = st.session_state.username
    st.sidebar.success(f"Logged in as: **{active_user}**")
    initialize_app(active_user)

    # Load saved progress on startup
    loaded_progress = load_progress(active_user)
    # ==========================================
    # GLOBAL MASTER TEMPLATE CONFIGURATION
    # ==========================================


    if not os.path.exists(CSV_FILE):
        st.error(f"Fatal Error: Master template file '{CSV_FILE}' not found!")
        st.stop()


    if st.sidebar.button("Save Progress", help="Manually save your current progress"):
        try:
            if save_progress(st.session_state, active_user):
                st.sidebar.success("Progress saved successfully!")
            else:
                st.sidebar.error("Failed to save progress")
        except Exception as e:
            st.sidebar.error(f"Serialization Error: Could not save progress yet. ({e})")


    # ==========================================
    # 2. CORE LOGIC FUNCTIONS
    # ==========================================
    # Dynamically pick the target CSV depending on the selectbox state
    if st.session_state.get("semester") == "Y2S2":
        CSV_FILE = "learning_objectives_y2s2.csv"
        NOTES_FILE = "lecture_notes.csv"
    else:
        CSV_FILE = "learning_objectives_informative_reports.csv"
        NOTES_FILE = "lecture_notes.csv"

    JOIN_COLUMN = "lecture_id"
    # ==========================================
    # 3. WEB INTERFACE
    # ==========================================
    st.markdown("<div id='top-anchor'></div>", unsafe_allow_html=True)
    if st.session_state.get('exam_submitted'):
        st.markdown("""
            <script>
                window.location.href = '#top-anchor';
            </script>
        """, unsafe_allow_html=True)
    st.title("Trainer")
    
    # NEW: Show exam breakdown at top if submitted
    if st.session_state.get('exam_submitted'):

        st.metric(
            label="Exam Performance", 
            value=f"{st.session_state.last_score} / {num_actual_questions}",
            delta=f"{(st.session_state.last_score / num_actual_questions * 100):.1f}% Correct",
            delta_color="normal" if st.session_state.last_score / num_actual_questions >= 0.7 else "inverse"
        )
        
        if st.session_state.get('level_message'):
            st.info(st.session_state.level_message)
        if st.session_state.get('immediate_wrong_breakdown'):
            st.markdown("### Answer Breakdown")
            st.markdown(st.session_state.immediate_wrong_breakdown)
        
        st.write("---")
    # This layout structure prevents the button from stretching awkwardly on wide screens
    gen_col1, gen_col2, gen_col3 = st.columns([1, 2, 1])

    with gen_col2:
        # use_container_width fills the middle column layout; type="primary" makes it high-contrast
        generate_clicked = st.button(
            "Generate New Exam", 
            type="primary", 
            use_container_width=True,
            help="Click here to compile a fresh customized exam based on your filter selections."
        )
    
    if generate_clicked:
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
        # --- NEW RESETS ---
        st.session_state.ai_feedback_clean = ""
        st.session_state.level_message = ""
        if 'immediate_wrong_breakdown' in st.session_state:
            st.session_state.immediate_wrong_breakdown = ""

        

        n = st.session_state.num_questions
        try:
            df_main = pd.read_csv(CSV_FILE)
            df_notes = pd.read_csv(NOTES_FILE)
            df = pd.merge(df_main, df_notes, on=JOIN_COLUMN, how='left')
            if "semester" in df.columns:
                df = df[df['semester'] == st.session_state.semester]

            # ────────── ADD THIS NEW BLOCK HERE TO INITIALIZE FILTERS ──────────
            # Reconstruct subject_filter from session state checkboxes
            categories = sorted(df_main['category'].fillna("Uncategorized").astype(str).unique().tolist())
            subject_filter = [cat for cat in categories if st.session_state.get(f"focus_{cat}", True)]
            
            # Reconstruct exam_filter from session state checkboxes
            exam_filter = []
            if 'exam' in df_main.columns:
                exams = sorted(df_main['exam'].fillna("Uncategorized").astype(str).unique().tolist())
                exam_filter = [ex for ex in exams if st.session_state.get(f"exam_{ex}", True)]
                
            # Reconstruct system_filter from session state checkboxes
            system_filter = []
            if 'system' in df_main.columns:
                systems = sorted(df_main['system'].fillna("Uncategorized").astype(str).unique().tolist())
                system_filter = [sys for sys in systems if st.session_state.get(f"sys_{sys}", True)]
            # ───────────────────────────────────────────────────────────────────

            # --- NEW LECTURE TAB OVERRIDE CHECK --- 
            # If the user selected specific lectures in Tab 2, use them instead of Blueprint filters
            if 'filter_by_lecture_ids' in st.session_state and st.session_state.filter_by_lecture_ids:
                df = df[df[JOIN_COLUMN].isin(st.session_state.filter_by_lecture_ids)]
            else:
                # --- Default: Fallback to your standard multi-checkbox filters ---
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
        
            # --- SMART SAMPLING (LECTURE WEIGHTED EQUALLY) ---
            # 1. Map the base category weights from your blueprint
            df['base_weight'] = df['category'].map(EXAM_WEIGHTS).fillna(0.05)

            # 2. Count how many rows (learning objectives) each unique lecture has
            # This uses your JOIN_COLUMN ("lecture_id") to find the size of each lecture
            lecture_counts = df.groupby(JOIN_COLUMN).size().to_dict()

            # 3. Normalize the weight: divide the base weight by the number of objectives in that lecture
            # This ensures a lecture's total weight is split equally among its rows
            df['sampling_weight'] = df.apply(
                lambda row: row['base_weight'] / lecture_counts[row[JOIN_COLUMN]] 
                if row[JOIN_COLUMN] in lecture_counts and lecture_counts[row[JOIN_COLUMN]] > 0 
                else 0.05, 
                axis=1
            )

            # 4. Sample using the normalized lecture-balanced weights
            try:
                # We use replace=False so we don't duplicate questions in the same exam
                st.session_state.samples_df = df.sample(min(n, len(df)), weights='sampling_weight', replace=False)
                st.sidebar.info("Y2S1 complete")
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

            # 1. Helper function to safely merge the columns into a single continuous string per row
            def combine_row_text(row):
                explanation = str(row.get('explanation', '')).strip()
                content = str(row.get('content', '')).strip()
                flashcards = str(row.get('flashcards', '')).strip()
                
                # Filter out empty fields so we don't introduce awkward spacing or isolated punctuation
                valid_segments = [seg for seg in [explanation, content, flashcards] if seg]
                
                # Join them with a space so they form a continuous stream of sentences
                return " ".join(valid_segments)

            # 2. Combine the fields first across the dataframe row-by-row
            combined_raw_text = samples_df.apply(combine_row_text, axis=1)

            # 3. Apply your rotation function to the entire combined block of sentences
            # This allows sentences from 'content' or 'flashcards' to seamlessly shift to the front!
            samples = combined_raw_text.apply(randomize_paragraph_start).tolist()
            
            with st.spinner(f"Generating {n} questions at Level {st.session_state.current_level}..."):
                raw_response = get_blind_exam(samples, st.session_state.current_level, n)

                if "[KEY:" in raw_response:
                    # Use a split that keeps the questions separate from the key
                    text, key_part = raw_response.split("[KEY:")

                    # CLEANING: Remove the key section from the visible text
                    st.session_state.current_exam = text.strip()
                    st.session_state.current_key = re.findall(r'[A-D]', key_part)
                    
                    # --- PERSISTENT SAVE AT MOMENT OF GENERATION ---
                    state_snapshot = {
                        "current_level": st.session_state.current_level,
                        "exam_model": st.session_state.get("exam_model", 'gemini-3.1-flash-lite'),
                        "num_questions": st.session_state.num_questions,
                        "missed_questions": st.session_state.missed_questions,
                        "current_exam": st.session_state.current_exam,
                        "current_key": st.session_state.current_key,
                        "current_categories": st.session_state.current_categories
                    }
                    
                    # Convert the sampling DataFrame to JSON records if it contains valid context data
                    if hasattr(st.session_state, 'samples_df') and isinstance(st.session_state.samples_df, pd.DataFrame) and not st.session_state.samples_df.empty:
                        state_snapshot["samples_df"] = st.session_state.samples_df.to_dict(orient="records")
                    else:
                        state_snapshot["samples_df"] = []

                    try:
                        save_progress(state_snapshot, active_user)
                    except Exception:
                        pass # Safeguard to ensure any serialization issues won't crash the UI runtime
                    # -----------------------------------------------

                    st.rerun()
                else:
                    st.error(f"Failed to generate a perfectly formatted exam. Please click generate again.")
        except Exception as e:
            st.error(f"File Error: Ensure {CSV_FILE} and {NOTES_FILE} are in the folder. ({e})")

    st.sidebar.header("Stats & Controls")

    # Move Active Level metric here
    st.sidebar.metric("Active Level", f"{st.session_state.current_level}/50")

    df_sidebar = pd.read_csv(CSV_FILE)

    # Create two tabs inside the sidebar
    filter_tab1, filter_tab2 = st.sidebar.tabs(["Exam Filter", "Lecture Filter"])

    # --- TAB 1: BLUEPRINT FILTERS (Original Logic) ---
    with filter_tab1:
        # --- Subject filter Checkboxes ---
        st.markdown("**Subjects:**")
        categories = sorted(df_sidebar['category'].fillna("Uncategorized").astype(str).unique().tolist())
        subject_filter = []
        for cat in categories:
            if st.checkbox(cat, value=True, key=f"focus_{cat}"):
                subject_filter.append(cat)
                
        st.markdown("---")
        
        # --- Exam Filter Checkboxes ---
        exam_filter = []
        if 'exam' in df_sidebar.columns:
            st.markdown("**Exam:**")
            exams = sorted(df_sidebar['exam'].fillna("Uncategorized").astype(str).unique().tolist())
            for ex in exams:
                if st.checkbox(ex, value=True, key=f"exam_{ex}"):
                    exam_filter.append(ex)
        else:
            exam_filter = []
            
        st.markdown("---")
        
        # --- Systems Filter Checkboxes ---
        system_filter = []
        if 'system' in df_sidebar.columns:
            st.markdown("**Systems:**")
            systems = sorted(df_sidebar['system'].fillna("Uncategorized").astype(str).unique().tolist())
            for sys in systems:
                if st.checkbox(sys, value=True, key=f"sys_{sys}"):
                    system_filter.append(sys)
        else:
            system_filter = []

    # --- TAB 2: LECTURE FILTERS (New Integrated Feature) ---
    with filter_tab2:
        
        # Identify your Join Column dynamically from the dataset
        if JOIN_COLUMN in df_sidebar.columns:
            # Get all unique lecture IDs available
            available_lectures = sorted(df_sidebar[JOIN_COLUMN].dropna().unique().tolist())
            
            # Use a multi-select box so users can select one or multiple lectures
            selected_lectures = st.multiselect(
                "Select Lecture:", 
                options=available_lectures,
                default=[],
                key="filter_by_lecture_ids"
            )
        else:
            st.warning(f"Join column '{JOIN_COLUMN}' not found in the dataset.")
            selected_lectures = []


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



    # Display the Exam
    if st.session_state.current_exam:

        

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

        # --- 1. INJECT ISOLATED OPTION STYLE MATRIX ONLY ---
        st.markdown("""
            <style>
            /* Target ONLY your quiz options, leaving all other site buttons untouched */
            .quiz-option-box {
                display: block;
                width: 100%;
                padding: 14px 20px;
                margin: 8px 0;
                border-radius: 8px;
                font-size: 15px;
                text-align: left;
                background-color: transparent;
                border: 2px solid #e0e0e0;
                color: #333333;
                transition: all 0.2s ease-in-out;
            }
            
            /* Elegant hover state strictly localized to quiz options */
            .quiz-option-box:hover {
                border-color: #4b6cb7;
                background-color: #f4f7fc;
                color: #4b6cb7;
            }
            
            /* Selected state has exact same structural dimensions to prevent shifting */
            .quiz-option-box-selected {
                display: block;
                width: 100%;
                padding: 14px 20px;
                margin: 8px 0;
                border-radius: 8px;
                font-size: 15px;
                text-align: left;
                border: 2px solid #4b6cb7;
                background-color: #eef2fa;
                color: #4b6cb7;
                font-weight: bold;
            }
            </style>
        """, unsafe_allow_html=True)

        # --- 2. RENDER THE QUESTIONS DYNAMICALLY (OUTSIDE st.form CONSTRAINTS FOR FAST RERUNS) ---
        for i, q_text in enumerate(individual_questions):
            st.subheader(f"Question {i+1}")
            
            # Extract clinical question text body before the option choices begin
            prompt_match = re.search(r"(\d+\s*\.\s*.*?)(?=A\s*\.\s*)", q_text, re.DOTALL)
            q_prompt = prompt_match.group(1).strip() if prompt_match else q_text
            
            # Keep the question at its native, original markdown text size
            st.markdown(q_prompt.replace("\n", "<br>"), unsafe_allow_html=True)
            st.write("") 

            # Clean option boundaries handling spacing nuances from API outputs
            opt_A = re.search(r"(A\s*\.\s*.*?)(?=[B-D]\s*\.\s*|$)", q_text, re.DOTALL)
            opt_B = re.search(r"(B\s*\.\s*.*?)(?=[A,C,D]\s*\.\s*|$)", q_text, re.DOTALL)
            opt_C = re.search(r"(C\s*\.\s*.*?)(?=[A,B,D]\s*\.\s*|$)", q_text, re.DOTALL)
            opt_D = re.search(r"(D\s*\.\s*.*?)(?=[A-C]\s*\.\s*|$)", q_text, re.DOTALL)

            options_dict = {
                "A": opt_A.group(1).strip() if opt_A else "A. Option A",
                "B": opt_B.group(1).strip() if opt_B else "B. Option B",
                "C": opt_C.group(1).strip() if opt_C else "C. Option C",
                "D": opt_D.group(1).strip() if opt_D else "D. Option D"
            }

            current_selection = st.session_state.user_selections.get(i, None)

            # Safely convert keys to list and track indexing position mapping
            choice_keys = list(options_dict.keys())

            # 1. Inject a style trick to completely hide this specific row of native radio selectors
            st.markdown(f"""
                <style>
                div[data-testid="stRadio"] {{
                    display: none !important;
                }}
                div[data-testid="stButton"] button {{
                    white-space: normal !important;
                    text-align: left !important;
                    height: auto !important;
                    word-break: break-word !important;
                    overflow-wrap: break-word !important;
                    word-wrap: break-word !important;
                    padding: 12px 16px !important;
                    line-height: 1.5 !important;
                    min-height: auto !important;
                    max-width: 100% !important;
                }}
                </style>
            """, unsafe_allow_html=True)

            # 2. Render an invisible radio button in the background to handle the variable state
            selected_letter = st.radio(
                label=f"Radio Select Q{i}",
                options=choice_keys,
                index=choice_keys.index(current_selection) if current_selection in choice_keys else None,
                key=f"native_radio_q_{i}",
                label_visibility="collapsed"
            )

            # 3. Render your custom styled choices as large, full-width clickable buttons
            for choice_letter, full_sentence_text in options_dict.items():
                is_selected = (current_selection == choice_letter)
                btn_type = "primary" if is_selected else "secondary"
                prefix = "➔   " if is_selected else "      "

                # Use use_container_width=True to make the buttons big and fill the space
                # Disable buttons after submission so users can't change answers while viewing feedback
                if st.button(f"{prefix}{full_sentence_text}", key=f"btn_q_{i}_{choice_letter}", use_container_width=True, type=btn_type, disabled=st.session_state.get('exam_submitted', False)):
                    st.session_state.user_selections[i] = choice_letter
                    st.rerun()

            # --- NEW: INJECT PER-QUESTION AI FEEDBACK RIGHT HERE ---
            if st.session_state.get('exam_submitted') and st.session_state.get('ai_feedback_clean'):
                # Look for the section matching "### Question X" or "### Question [X]"
                feedback_str = st.session_state.ai_feedback_clean
                pattern = rf"### Question \s*\[?{i+1}\]?.*?(?=### Question \s*\[?{i+2}\]?|---|$)"
                match = re.search(pattern, feedback_str, re.DOTALL | re.IGNORECASE)

                if match:
                    st.info("💡 **AI Grading Feedback:**")
                    st.markdown(match.group(0).strip())
                else:
                    # If no specific incorrect feedback is found, the question might be correct
                    correct_ans = st.session_state.current_key[i]
                    user_ans = st.session_state.user_selections.get(i, "No Answer")
                    if user_ans == correct_ans:
                        st.success(f"Correct! You answered `{user_ans}`.")

            st.write("---")

        # Standalone execution grading submission action button
        # --- NEW: SCORE & LEVELING FEEDBACK HIGHLIGHT ---
        if st.session_state.get('exam_submitted'):
            num_actual_questions = len(individual_questions)
            st.write("Click the button on the right to scroll up >>>")

        # Standalone execution grading submission action button (normalized look)
        submitted = st.button("Submit for Grading", type="primary")
        if submitted:
            num_actual_questions = len(individual_questions)
            user_answers = [st.session_state.user_selections.get(idx, None) for idx in range(num_actual_questions)]
            user_input = "\n".join([f"Q{idx+1}: {ans if ans else 'No Answer'}" for idx, ans in enumerate(user_answers)])
            
            correct_key = st.session_state.current_key[:num_actual_questions]
            correct_key_formatted = "\n".join([f"Q{idx+1}: {ans}" for idx, ans in enumerate(correct_key)])
            
            st.session_state.last_user_input = user_input
            st.session_state.last_correct_key = correct_key_formatted
            st.session_state.last_user_answers_list = user_answers
            
            if len(user_answers) != len(correct_key):
                st.error(f"Mismatch: The exam has {len(correct_key)} questions, but you entered {len(user_answers)} answers.")
                st.stop()
                
            score = 0
            incorrect_summary_markdown = ""
            for i, q_text in enumerate(individual_questions):
                if i >= len(user_answers):
                    break
                u_ans = user_answers[i] if user_answers[i] else "No Answer"
                correct = correct_key[i] if i < len(correct_key) else None
                
                # 1. ALWAYS track the total exam performance history (Correct AND Incorrect)
                st.session_state.exam_history.append({
                    "question": individual_questions[i].strip(),
                    "correct": correct,
                    "yours": u_ans,
                    "semester": st.session_state.get("semester", "Y2S1"),
                    "category": st.session_state.current_categories[i] if i < len(st.session_state.current_categories) else "General",
                })
                
                if u_ans == correct:
                    score += 1
                else:
                    clean_q_snippet = re.sub(r'<br\s*/?>', ' ', individual_questions[i].split('\n')[0][:120])
                    incorrect_summary_markdown += f"**Question {i+1}:** *{clean_q_snippet}...*\n"
                    incorrect_summary_markdown += f"&nbsp;&nbsp;&nbsp;&nbsp;• **Your Answer:** `{u_ans}` | **Correct Answer:** `{correct}`\n\n"
                    
                    # 2. ONLY add to the missed questions bank if it was wrong
                    st.session_state.missed_questions.append({
                        "question": individual_questions[i].strip(),
                        "correct": correct,
                        "yours": u_ans,
                        "semester": st.session_state.get("semester", "Y2S1"),
                        "category": st.session_state.current_categories[i] if i < len(st.session_state.current_categories) else "General",
                    })


                
            st.session_state.exam_submitted = True
            st.session_state.last_score = score
            st.session_state.immediate_wrong_breakdown = incorrect_summary_markdown if incorrect_summary_markdown else "  🎉   **Perfect score! You got every question right!**"

            # --- NEW: FETCH FEEDBACK BEFORE RERUN SO IT DISPLAYS UNDER QUESTIONS ---
            with st.spinner("Analyzing answers..."):
                try:
                    feedback = get_ai_grading(
                        st.session_state.current_exam,
                        user_input,
                        correct_key_formatted,
                        score
                    )
                    st.session_state.ai_feedback_clean = feedback
                except Exception as e:
                    st.session_state.ai_feedback_clean = f"Error generating explanation: {e}"

            # Auto-save progress after AI grading completes
            try:
                save_progress(st.session_state, active_user)
            except Exception as e:
                pass  # Silently fail to avoid disrupting the user experience

            percentage_correct = (score / num_actual_questions) * 100
            if (num_actual_questions - score) <= 1 or percentage_correct >= 90:
                next_level = min(50, st.session_state.current_level + 1)
                if next_level > st.session_state.current_level:
                    st.session_state.level_message = f"**Excellent performance ({percentage_correct:.0f}%)! You have leveled up to Level {next_level}!**"
                else:
                    st.session_state.level_message = f"**Fantastic score ({percentage_correct:.0f}%)! You are at the maximum mastery level (Level 50)!**"
                st.session_state.current_level = next_level
            elif percentage_correct <= 60:
                next_level = max(1, st.session_state.current_level - 1)
                if next_level < st.session_state.current_level:
                    st.session_state.level_message = f"**Score was {percentage_correct:.0f}%. The system adjusted your difficulty down to Level {next_level} to rebuild foundations.**"
                else:
                    st.session_state.level_message = f"**Score was {percentage_correct:.0f}%. You are at Level 1. Keep practicing to build confidence!**"
                st.session_state.current_level = next_level
            else:
                st.session_state.level_message = f"**Solid effort ({percentage_correct:.0f}%)! Remaining at Level {st.session_state.current_level} to lock in consistency.**"
                
            st.rerun()
            

    # Missed Questions Bank in Sidebar
    if st.session_state.missed_questions:
        # 2. YOUR ORIGINAL SIDEBAR EXPORT (Preserved)
        st.sidebar.subheader(f"Missed Questions ({len(st.session_state.missed_questions)})")
        # Prepare the missed questions text content dynamically in memory
        melbourne_mistakes_time = datetime.datetime.now(ZoneInfo("Australia/Melbourne")).strftime('%Y-%m-%d %H:%M')
        export_text = f"=== WEB SESSION (Melbourne Time): {melbourne_mistakes_time} ===\n"

        for item in st.session_state.missed_questions:
            if isinstance(item, dict):
                cat = item.get('category', 'General')
                q_text = item.get('question', 'Unknown Question') 
                # CHANGED: Match 'correct' key used in your dictionary snapshot
                ans_text = item.get('correct', 'N/A') 
                export_text += f"\n[{cat}] {q_text}\n[CORRECT: {ans_text}]\n"
            else:
                # Fallback safeguard
                export_text += f"\n[General] {str(item)}\n"

        # Offer the file directly as a local browser download
        st.sidebar.download_button(
            label="Download Mistakes",
            data=export_text,
            file_name=f"{active_user}_missed_questions.txt",
            mime="text/plain",
            key=f"download_mistakes_{active_user}"
        )

    pass

def render_stats_page():
    # ==========================================
    # INITIALIZATION FOR STATS PAGE
    # ==========================================
    if 'username' not in st.session_state:
        st.session_state.username = "Default"

    new_user = st.sidebar.text_input("Enter your username:", st.session_state.username)
    if st.sidebar.button("Switch / Create Profile"):
        st.session_state.username = new_user.strip()
        st.rerun()

    active_user = st.session_state.username
    st.sidebar.success(f"Logged in as: **{active_user}**")
    initialize_app(active_user)

    # Load saved progress on startup
    loaded_progress = load_progress(active_user)

    def display_analytics_dashboard():
        st.title("Stats")
        
        # Pull directly from the complete exam history pool
        history_data = st.session_state.get("exam_history", [])        
        if not history_data:
            st.info("No exam submissions recorded for this profile. Generate and grade an exam to unlock data insights!")
            return
            
        # Convert the history collection to a pandas DataFrame for calculation
        df_history = pd.DataFrame(history_data)
        
        # Apply the active semester filtering cleanly (Case-insensitive)
        current_sem = st.session_state.get("semester", "Y2S1").upper()
        if "semester" in df_history.columns:
            df_history = df_history[df_history['semester'].str.upper() == current_sem]
            
        if df_history.empty:
            st.info(f"No history records found matching active semester: {current_sem}")
            return
            
        # Calculate Core Performance Metrics
        total_completed = len(df_history)
        df_history['is_correct'] = df_history['yours'] == df_history['correct']
        total_correct = df_history['is_correct'].sum()
        total_incorrect = total_completed - total_correct
        overall_accuracy = (total_correct / total_completed) * 100 if total_completed > 0 else 0
        
        # ==========================================
        # KPI METRIC CARDS ROW
        # ==========================================
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(label="Total Questions Done", value=f"{total_completed} Qs")
        with col2:
            st.metric(
                label="Overall Accuracy", 
                value=f"{overall_accuracy:.1f}%",
                delta="Target: >75%" if overall_accuracy < 75 else "Excellent!",
                delta_color="normal" if overall_accuracy >= 75 else "inverse"
            )
        with col3:
            st.metric(label="Correct Answers", value=f"{total_correct}")
        with col4:
            st.metric(label="Current Level Tier", value=f"Lvl {st.session_state.get('current_level', 1)} / 50")
            
        st.markdown("---")
        
        # ==========================================
        # ADVANCED DIAGNOSTICS & CHARTS
        # ==========================================
        st.subheader("Performance Breakdown by Medical Specialty")
        
        # Group by category and compute accurate performance
        subject_stats = df_history.groupby('category').agg(
            Total=('is_correct', 'count'),
            Correct=('is_correct', 'sum')
        ).reset_index()
        
        subject_stats['Accuracy (%)'] = (subject_stats['Correct'] / subject_stats['Total']) * 100
        subject_stats = subject_stats.sort_values(by='Accuracy (%)', ascending=False)
        
        # Display as a clean, full-width data table with built-in progress bars
        st.dataframe(
            subject_stats,
            column_config={
                "category": "Medical Specialty",
                "Correct": "Correct Qs",
                "Total": "Total Faced",
                "Accuracy (%)": st.column_config.ProgressColumn(
                    "Accuracy Tracker",
                    help="Target 100% mastery tier across every board subject blueprint",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100
                )
            },
            hide_index=True,
            use_container_width=True  
        )
        
        # ==========================================
        # STUDY RECOMMENDATION ENGINE
        # ==========================================
        st.markdown("---")
        st.subheader("Adaptive Study Recommendation")
        
        # Identify lowest performing subject tracking metrics
        if not subject_stats.empty:
            weakest_subject = subject_stats.iloc[-1]
            if weakest_subject['Accuracy (%)'] < 70:
                st.warning(
                    f"Your performance metrics show a weakness in **{weakest_subject['category']}** "
                    f"({weakest_subject['Accuracy (%)']:.1f}% accuracy). Consider isolating this subject in your filter settings "
                    f"on the main menu."
                )
            else:
                st.success("All subject tracks are performing above standard target parameters. Keep testing up to Level 50!")
                
    display_analytics_dashboard()

def render_export_page():
    st.title("Export Questions")
    
    if not st.session_state.get('current_exam'):
        st.warning("No active exam found. Please generate an exam on the 'Exam Trainer' page first.")
        st.stop()
        
    st.info("Download your currently active exam in your preferred format below.")
    
    # 1. PDF Export Section
    try:
        from fpdf import FPDF
    except ImportError:
        FPDF = None
        
    if FPDF:
        # Recreate the metadata context
        current_metadata = {
            "level": st.session_state.get('current_level', 1),
            "subject": "Active Exam Filter", 
            "exam": "Active Exam Filter",
            "system": "Active Exam Filter"
        }
        
        pdf_bytes = create_exam_pdf(
            st.session_state.current_exam, 
            st.session_state.current_key, 
            metadata=current_metadata
        )
        
        if pdf_bytes:
            st.subheader("PDF Format")
            st.download_button(
                label="Download Exam as PDF", 
                data=pdf_bytes, 
                file_name="practice_exam.pdf", 
                mime="application/pdf",
                key="download_exam_pdf_page"
            )
            
    # 2. TXT Export Section
    raw_exam_text = st.session_state.current_exam.strip()
    clean_exam_text = re.sub(r'(D \. \s.*?) \n +(?=\d+ \. \s)', r'\1 \n ', raw_exam_text)
    
    from zoneinfo import ZoneInfo
    melbourne_now = datetime.datetime.now(ZoneInfo("Australia/Melbourne")).strftime('%Y-%m-%d %H:%M:%S')
    
    txt_content = (
        f"Practice Exam (Level {st.session_state.get('current_level', 1)}/50)\n"
        f"Generated on (Melbourne Time): {melbourne_now}\n"
        f"----------------------------------------------------------------------\n\n"
        f"{clean_exam_text}\n"
    )
    
    st.write("---")
    st.subheader("Plain Text Format")
    st.download_button(
        label="Download Exam as TXT", 
        data=txt_content, 
        file_name="practice_exam.txt", 
        mime="text/plain", 
        key="download_exam_txt_page"
    )

def render_settings_page():
    st.title("⚙️ Global Settings")
    st.write("---")
    
    if 'username' not in st.session_state:
        st.session_state.username = "Default"
    active_user = st.session_state.username
    
    # Run core parameter synchronization
    initialize_app(active_user)
    
    # 1. Main Configuration Sliders
    st.subheader("Manually adjust")
    st.session_state.current_level = st.slider("Starting Level", 1, 50, st.session_state.current_level)
    st.session_state.num_questions = st.slider("Number of Questions", 1, 50, st.session_state.num_questions)
    
    # Define the options exactly as you want them
    semester_options = ["Y2S1", "Y2S2"]

    # Look up what is currently saved in session state to determine the starting index (default to 0 if not found)
    current_semester = st.session_state.get("semester", "Y2S1")
    default_index = semester_options.index(current_semester) if current_semester in semester_options else 0

    # Your selectbox, dynamically setting its initial value based on session state
    st.session_state.semester = st.selectbox(
        "Active Semester", 
        options=semester_options, 
        index=default_index
    )


    st.session_state.thinking_level = st.selectbox(
        "Gemini Thinking Level",
        options=["MINIMAL", "LOW", "MEDIUM", "HIGH"],
        index=["MINIMAL", "LOW", "MEDIUM", "HIGH"].index(st.session_state.thinking_level),
        help="Control how deeply the model deliberates before generating questions or grading."
    )
    
    st.write("---")
    
    # 2. Speed Switch
    st.subheader("Model Selection")
    model_choice = st.radio(
        label="Speed Selection",
        options=["3.1 flash lite", "3.5 flash"],
        index=1 if st.session_state.get('exam_model', 'gemini-3.5-flash') == 'gemini-3.5-flash' else 0,
        horizontal=True,
        help="Fast uses Flash-Lite. Slow & Smart uses Flash."
    )
    st.session_state.exam_model = 'gemini-3.5-flash' if "Slow" in model_choice else 'gemini-3.1-flash-lite'
    
    st.write("---")
    
    # 3. Structural Operational Actions
    st.subheader("Danger!!")
    col1, col2 = st.columns(2)
    
    with col1:
        # Initialize the confirmation flag if it doesn't exist yet
        if "confirm_reset" not in st.session_state:
            st.session_state.confirm_reset = False

        # If the user hasn't clicked reset yet, show the main button
        if not st.session_state.confirm_reset:
            if st.button("Reset All Session Progress", help="Clear all saved progress and reset to defaults", use_container_width=True):
                st.session_state.confirm_reset = True
                st.rerun()
        
        # If clicked, reveal the confirmation warning window
        else:
            st.warning("⚠️ Are you absolutely sure? This will permanently delete your progress file and cannot be undone.")
            
            sub_col1, sub_col2 = st.columns(2)
            with sub_col1:
                if st.button("Yes, Clear Everything", type="primary", use_container_width=True):
                    user_specific_progress = f"{active_user}_progress.json"
                    if os.path.exists(user_specific_progress):
                        os.remove(user_specific_progress)
                        
                    initialize_app(active_user, force_reset=True)
                    st.session_state.confirm_reset = False  # Reset flag state
                    st.success("Progress reset successfully!")
                    st.rerun()
                    
            with sub_col2:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.confirm_reset = False  # Dismiss confirmation
                    st.rerun()
    with col2:
        render_data_portability_interface()

# ==============================================================
# FLOATING ACTION BUTTON (NATIVE HTML SCROLL BACK TO TOP)
# ==============================================================
st.markdown(
    """
    <style>
    /* Fixed container positioned perfectly over the app frame canvas */
    .floating-container {
        position: fixed;
        bottom: 30px;
        right: 30px;
        z-index: 999999;
    }
    
    .scroll-arrow-link {
        background-color: #FF4B4B; /* Streamlit Signature Red */
        color: white !important;
        border-radius: 50%;
        width: 52px;
        height: 52px;
        font-size: 24px;
        font-weight: bold;
        display: flex;
        align-items: center;
        justify-content: center;
        text-decoration: none !important;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, background-color 0.2s ease;
    }
    
    .scroll-arrow-link:hover {
        background-color: #D33636;
        transform: scale(1.1);
    }
    </style>
    
    <div class="floating-container">
        <a href="#top-anchor" target="_self" class="scroll-arrow-link">▲</a>
    </div>
    """,
    unsafe_allow_html=True
)
# Create the navigation router
pg = st.navigation([
    st.Page(render_trainer_page, title="Exam Trainer", icon="📝"),
    st.Page(render_stats_page, title="Stats", icon="📊", url_path="stats"),
    st.Page(render_export_page, title="Export Questions", icon="📥", url_path="export"), 
    st.Page(render_settings_page, title="Settings", icon="⚙️", url_path="settings")
])
pg.run()