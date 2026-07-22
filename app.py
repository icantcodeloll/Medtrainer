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
import json
from google import genai
from google.genai import types
from progress_manager import save_progress, load_progress, update_player_elo, save_single_player_score, get_leaderboard_data, supabase
import shutil

# PDF library availability check
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.units import inch
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.enums import TA_LEFT
    from reportlab.pdfbase.pdfdoc import PDFArray
    from reportlab.pdfbase.pdfdoc import PDFDictionary
    from reportlab.pdfbase.pdfdoc import PDFName
    from reportlab.pdfbase.pdfdoc import PDFString
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# PDF parsing library availability check
try:
    from PyPDF2 import PdfReader
    PDF_PARSING_AVAILABLE = True
except ImportError:
    PDF_PARSING_AVAILABLE = False

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
    except (FileNotFoundError, PermissionError, OSError, IOError):
        # Fails silently to prevent execution disruptions during local development environments
        pass

# Trigger asset pipeline validation prior to UI layout assembly
initialize_pwa_assets()

API_KEYS = [st.secrets["GENAI_KEY_1"], st.secrets["GENAI_KEY_2"]] #st.secrets["GENAI_KEY_3"]] # (Keep your full list here)
MAX_REQUESTS_PER_KEY_PER_MODEL = {
    'gemini-3.6-flash': 20,
    'gemini-3.5-flash-lite': 500
}  # Maximum requests per API key per model per day
CSV_FILE = "learning_objectives_informative_reports_y2s1.csv" 
NOTES_FILE = "lecture_notes_y2s1.csv"
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
EXAM_MODEL = 'gemini-3.5-flash-lite'
GRADER_MODEL = 'gemini-3.5-flash-lite'

# Constants
LEVEL_UP_THRESHOLD = 90
LEVEL_DOWN_THRESHOLD = 60
MAX_LEVEL = 50
MIN_LEVEL = 1

# Compiled regex patterns for performance
SENTENCE_SPLIT_PATTERN = re.compile(r'(?<=[.!?])\s+')
ANSWER_KEY_PATTERN = re.compile(r'[A-D]')
INTRO_CLEANUP_PATTERN = re.compile(r"^(Here are|Based on|Sure|I have generated).*?\n", re.IGNORECASE)
QUESTION_SPLIT_PATTERN = re.compile(r'\n(?=\d+\.\s)')
QUESTION_PROMPT_PATTERN = re.compile(r"(\d+\s*\.\s*.*?)(?=A\s*\.\s*)", re.DOTALL)
OPTION_A_PATTERN = re.compile(r"(A\s*\.\s*.*?)(?=[B-D]\s*\.\s*|$)", re.DOTALL)
OPTION_B_PATTERN = re.compile(r"(B\s*\.\s*.*?)(?=[A,C,D]\s*\.\s*|$)", re.DOTALL)
OPTION_C_PATTERN = re.compile(r"(C\s*\.\s*.*?)(?=[A,B,D]\s*\.\s*|$)", re.DOTALL)
OPTION_D_PATTERN = re.compile(r"(D\s*\.\s*.*?)(?=[A-C]\s*\.\s*|$)", re.DOTALL)
USERNAME_SANITIZE_PATTERN = re.compile(r'[^\w\s-]')
BR_CLEANUP_PATTERN = re.compile(r'<br\s*/?>')
EXAM_CLEANUP_PATTERN = re.compile(r'(D \. \s.*?) \n +(?=\d+ \. \s)')

# Helper Functions
def create_exam_backup(session_state) -> dict:
    """
    Create a backup dictionary of the current exam state.
    
    Args:
        session_state: Streamlit session state object
        
    Returns:
        dict: Backup dictionary containing exam data
    """
    if not session_state.get('current_exam'):
        return {}
    
    return {
        'current_exam': session_state.current_exam,
        'current_key': session_state.current_key,
        'user_selections': session_state.get('user_selections', {}),
        'exam_submitted': session_state.get('exam_submitted', False),
        'last_score': session_state.get('last_score', 0),
        'last_user_input': session_state.get('last_user_input', ""),
        'last_correct_key': session_state.get('last_correct_key', ""),
        'last_user_answers_list': session_state.get('last_user_answers_list', []),
        'current_categories': session_state.get('current_categories', []),
        'samples_df': session_state.get('samples_df', None)
    }

def restore_exam_from_backup(session_state, backup: dict) -> None:
    """
    Restore exam state from a backup dictionary.
    
    Args:
        session_state: Streamlit session state object
        backup: Backup dictionary containing exam data
    """
    if not backup:
        return
        
    session_state.current_exam = backup.get('current_exam')
    session_state.current_key = backup.get('current_key')
    session_state.user_selections = backup.get('user_selections', {})
    session_state.exam_submitted = backup.get('exam_submitted', False)
    session_state.last_score = backup.get('last_score', 0)
    session_state.last_user_input = backup.get('last_user_input', "")
    session_state.last_correct_key = backup.get('last_correct_key', "")
    session_state.last_user_answers_list = backup.get('last_user_answers_list', [])
    session_state.current_categories = backup.get('current_categories', [])
    session_state.samples_df = backup.get('samples_df', None)

def validate_username(username: str) -> str:
    """
    Validate and sanitize username input.
    
    Args:
        username: Raw username input
        
    Returns:
        str: Sanitized username
    """
    if not username:
        return "Default"
    
    # Remove any potentially harmful characters
    sanitized = username.strip()
    # Limit length to prevent abuse
    sanitized = sanitized[:50]
    # Remove special characters except alphanumeric, spaces, underscores, hyphens
    sanitized = USERNAME_SANITIZE_PATTERN.sub('', sanitized)
    
    return sanitized if sanitized else "Default"

def setup_user_profile() -> str:
    """
    Shared function to handle user profile setup across all pages.
    Displays username input, handles profile switching, and initializes app state.
    
    Returns:
        str: Active username
    """
    if 'username' not in st.session_state:
        st.session_state.username = "Default"
    
    
    new_user = st.sidebar.text_input("Enter your username:", st.session_state.username)
    if st.sidebar.button("Switch / Create Profile"):
        st.session_state.username = validate_username(new_user)
        
        # Wipe the screen clean so the new user's data can load
        keys_to_clear = ['current_level', 'num_questions', 'missed_questions', 'exam_history', 'current_exam', 'current_key', 'samples_df']
        for k in keys_to_clear:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()
    
    active_user = st.session_state.username
    
    
    st.sidebar.success(f"Logged in as: **{active_user}**")
    initialize_app(active_user)
    
    return active_user

# Global cleanup handler for tab close - defined at module level to avoid multiple registrations
_atexit_registered = False

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
                "exam_model": st.session_state.get("exam_model", 'gemini-3.5-flash-lite'),
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
    except (KeyError, AttributeError, TypeError, IOError) as e:
        # Silently pass to ensure the server thread terminates smoothly
        # Log error for debugging purposes
        print(f"Error in save_on_tab_close: {e}")

def register_cleanup_handler():
    """
    Register the cleanup handler once at module level to prevent multiple registrations.
    """
    global _atexit_registered
    if not _atexit_registered:
        atexit.register(save_on_tab_close)
        _atexit_registered = True

@st.cache_data
def load_csv_data(file_path: str) -> pd.DataFrame:
    """
    Load CSV data with caching to improve performance.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded CSV data
    """
    return pd.read_csv(file_path)


# ==========================================
# 0. MULTI-PAGE CONFIGURATION & NAVIGATION
# ==========================================
def initialize_app(active_user: str, force_reset: bool = False) -> None:
    """
    Handles all initial state configurations, progress restoration, 
    and systemic fallback settings in one central runtime hook.
    
    Args:
        active_user: Username for the current session
        force_reset: Whether to force reset all state to defaults
    """
    # Load saved progress dynamically from local disk storage
    if force_reset:
        loaded_progress = {}
    else:
        loaded_progress = load_progress(active_user) # Existing Line 4595

    # Core parameters mapping dictionary 
    defaults = {
        "current_level": loaded_progress.get("current_level", 1),
        "exam_model": loaded_progress.get("exam_model", 'gemini-3.5-flash-lite'),
        "num_questions": loaded_progress.get("num_questions", 5),
        "semester": loaded_progress.get("semester", "y2s2"),
        "missed_questions": loaded_progress.get("missed_questions", []),
        "exam_history": loaded_progress.get("exam_history", []),
        "current_exam": loaded_progress.get("current_exam", ""),
        "current_key": loaded_progress.get("current_key", []),
        "key_index": min(loaded_progress.get("key_index", 0), max(0, len(API_KEYS) - 1)) if len(API_KEYS) > 0 else 0,
        "api_request_counts": loaded_progress.get("api_request_counts", {i: {'gemini-3.6-flash': 0, 'gemini-3.5-flash-lite': 0} for i in range(len(API_KEYS))}),
        "current_categories": loaded_progress.get("current_categories", []),
        "previous_test_data": {},
        "use_search": False,
        "thinking_level": "MEDIUM",
        "temperature": 0.7,
        "top_p": 0.95,
        "exam_submitted": False,
        "last_score": 0,
        "user_selections": {},
        "last_user_answers_list": [],
        "show_settings": False,
        "current_page": "Exam Trainer",
        "leaderboard_opt_in": loaded_progress.get("leaderboard_opt_in", False),
        "uploaded_pdf_answers": None
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

def get_client() -> genai.Client:
    """Get the current Gemini AI client with API key rotation."""
    return genai.Client(api_key=API_KEYS[st.session_state.key_index])

def rotate_to_next_available_key(model: str):
    """Rotate to the next API key that hasn't exceeded the request limit for the specific model."""
    keys_checked = 0
    original_index = st.session_state.key_index
    max_requests = MAX_REQUESTS_PER_KEY_PER_MODEL.get(model, 500)
    
    while keys_checked < len(API_KEYS):
        current_index = st.session_state.key_index
        key_counts = st.session_state.api_request_counts.get(current_index, {'gemini-3.6-flash': 0, 'gemini-3.5-flash-lite': 0})
        request_count = key_counts.get(model, 0)
        
        if request_count < max_requests:
            return True  # Current key is still available for this model
        
        # Move to next key
        st.session_state.key_index = (st.session_state.key_index + 1) % len(API_KEYS)
        keys_checked += 1
        
        # If we've checked all keys and returned to original, all are exhausted
        if st.session_state.key_index == original_index and keys_checked > 0:
            return False
    
    return False

def call_gemini_with_rotation(prompt: str, model_to_use: str, use_search: bool = False) -> str | None:
    keys_tried = 0
    max_requests = MAX_REQUESTS_PER_KEY_PER_MODEL.get(model_to_use, 500)
    
    # Check if current key is exhausted before starting
    if not rotate_to_next_available_key(model_to_use):
        st.error(f"All API keys have reached their request limit for {model_to_use}. Please add more keys or wait for reset.")
        return None
    
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

    # Add temperature and top_p settings
    config_args["temperature"] = st.session_state.get("temperature", 0.7)
    config_args["top_p"] = st.session_state.get("top_p", 0.95)

    # Pack arguments into the structural API configuration object
    generation_config = types.GenerateContentConfig(**config_args)

    # --- REST OF YOUR CONTINUOUS API LOOP ---
    while keys_tried < len(API_KEYS):
        # Check if current key is exhausted for this model before making request
        key_counts = st.session_state.api_request_counts.get(st.session_state.key_index, {'gemini-3.6-flash': 0, 'gemini-3.5-flash-lite': 0})
        current_count = key_counts.get(model_to_use, 0)
        if current_count >= max_requests:
            if not rotate_to_next_available_key(model_to_use):
                st.error(f"All API keys have reached their request limit for {model_to_use}. Please add more keys or wait for reset.")
                return None
            keys_tried += 1
            continue
        
        try:
            client = get_client()
            response = client.models.generate_content(
                model=model_to_use,
                contents=prompt,
                config=generation_config
            )
            
            # Increment request counter for successful request (model-specific)
            if st.session_state.key_index not in st.session_state.api_request_counts:
                st.session_state.api_request_counts[st.session_state.key_index] = {'gemini-3.6-flash': 0, 'gemini-3.5-flash-lite': 0}
            st.session_state.api_request_counts[st.session_state.key_index][model_to_use] = \
                st.session_state.api_request_counts[st.session_state.key_index].get(model_to_use, 0) + 1
            
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

def get_blind_exam(topics_list: list[str], level: int, num_questions: int) -> str | None:
    """
    Generate a blind exam using AI based on provided topics and difficulty level.
    
    Args:
        topics_list: List of topic strings to base questions on
        level: Difficulty level (1-50)
        num_questions: Number of questions to generate
        
    Returns:
        Generated exam text or None if generation fails
    """
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

    CRITICAL RULES FOR APPLICATION:
    1. Do NOT use the exact phrasing, sentences, or structured examples found in the source text. 
    2. Apply "Conceptual Paraphrasing": Extract the core medical mechanism or fact, and invent a completely new patient case study or scenario to test that concept.
    3. If the source text mentions a specific drug or symptom explicitly in an example, write your question using an entirely different clinical presentation that tests the exact same underlying physiology.

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

def get_ai_grading(exam_text: str, user_answers: str, correct_key: str, score: int) -> str | None:
    """
    Generate AI grading feedback for exam answers.
    
    Args:
        exam_text: The exam questions text
        user_answers: The student's answers
        correct_key: The correct answer key
        score: The student's score
        
    Returns:
        AI-generated grading feedback or None if generation fails
    """
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

def create_exam_pdf(exam_text: str, answer_key: list, user_answers: list = None, score: int = None, max_score: int = None, metadata: dict = None) -> bytes | None:
    """
    Generate a PDF containing the exam questions with interactive radio buttons, answer key, and optionally user selections and filters.
    
    Args:
        exam_text: The exam questions text
        answer_key: List of correct answers
        user_answers: List of user's answers (optional)
        score: User's score (optional)
        max_score: Maximum possible score (optional)
        metadata: Dictionary containing exam metadata (optional)
        
    Returns:
        PDF bytes or None if PDF library is not available
    """
    if not PDF_AVAILABLE:
        return None

    # Create a temporary file for the PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp_path = tmp.name
    
    try:
        # Create PDF with canvas
        c = canvas.Canvas(tmp_path, pagesize=letter)
        width, height = letter
        margin = 72
        y_position = height - margin
        
        # Title
        c.setFont("Helvetica-Bold", 16)
        if score is not None and max_score is not None:
            title = f"Practice Exam Results - Score: {score}/{max_score}"
        else:
            title = "Practice Exam"
        c.drawCentredString(width / 2, y_position, title)
        y_position -= 30
        
        # Metadata
        if metadata:
            c.setFont("Helvetica-Oblique", 9)
            melbourne_time = datetime.datetime.now(ZoneInfo("Australia/Melbourne")).strftime('%Y-%m-%d %H:%M:%S')
            meta_text = f"Level: {metadata.get('level', 'N/A')} | Subject: {metadata.get('subject', 'All')} | Exam Filter: {metadata.get('exam', 'All')} | System Filter: {metadata.get('system', 'All')}"
            time_text = f"Generated on (Melbourne Time): {melbourne_time}"
            c.drawCentredString(width / 2, y_position, meta_text)
            y_position -= 15
            c.drawCentredString(width / 2, y_position, time_text)
            y_position -= 25
        
        # Parse questions
        individual_questions = QUESTION_SPLIT_PATTERN.split(exam_text.strip())
        
        c.setFont("Helvetica", 10)
        
        for q_idx, q_text in enumerate(individual_questions):
            # Check if we need a new page
            if y_position < 100:
                c.showPage()
                y_position = height - margin
            
            # Question number
            c.setFont("Helvetica-Bold", 12)
            c.drawString(margin, y_position, f"Question {q_idx + 1}")
            y_position -= 20
            
            # Extract question text
            prompt_match = QUESTION_PROMPT_PATTERN.search(q_text)
            q_prompt = prompt_match.group(1).strip() if prompt_match else q_text
            
            # Draw question text (wrapped)
            c.setFont("Helvetica", 10)
            lines = []
            for line in q_prompt.split('\n'):
                words = line.split()
                current_line = ""
                for word in words:
                    if c.stringWidth(current_line + " " + word, "Helvetica", 10) < width - 2 * margin:
                        current_line += " " + word if current_line else word
                    else:
                        if current_line:
                            lines.append(current_line)
                        current_line = word
                if current_line:
                    lines.append(current_line)
            
            for line in lines:
                if y_position < margin + 20:
                    c.showPage()
                    y_position = height - margin
                c.drawString(margin, y_position, line)
                y_position -= 12
            
            y_position -= 10
            
            # Extract options
            opt_A = OPTION_A_PATTERN.search(q_text)
            opt_B = OPTION_B_PATTERN.search(q_text)
            opt_C = OPTION_C_PATTERN.search(q_text)
            opt_D = OPTION_D_PATTERN.search(q_text)
            
            options = [
                ("A", opt_A.group(1).strip()[2:].strip() if opt_A else "Option A"),
                ("B", opt_B.group(1).strip()[2:].strip() if opt_B else "Option B"),
                ("C", opt_C.group(1).strip()[2:].strip() if opt_C else "Option C"),
                ("D", opt_D.group(1).strip()[2:].strip() if opt_D else "Option D")
            ]
            
            # Draw options with interactive radio buttons
            for opt_idx, (opt_letter, opt_text) in enumerate(options):
                if y_position < margin + 20:
                    c.showPage()
                    y_position = height - margin
                
                # Create radio button form field
                radio_name = f"q{q_idx+1}"
                c.acroForm.radio(
                    name=radio_name,
                    value=opt_letter,
                    selected=False,
                    x=margin + 10,
                    y=y_position - 10,
                    buttonStyle='circle',
                    size=12,
                    borderStyle='solid',
                    borderWidth=1
                )
                
                # Option text
                opt_label = f"{opt_letter.lower()}. {opt_text}"
                c.drawString(margin + 30, y_position - 6, opt_label)
                y_position -= 20
            
            y_position -= 15
        
        # Add Answer Key page
        c.showPage()
        y_position = height - margin
        
        c.setFont("Helvetica-Bold", 16)
        c.drawCentredString(width / 2, y_position, "Exam Summary")
        y_position -= 30
        
        c.setFont("Helvetica", 10)
        for i, ans in enumerate(answer_key):
            if y_position < margin + 20:
                c.showPage()
                y_position = height - margin
            
            text = f"Question {i+1}: Correct Key: {ans}"
            if user_answers and i < len(user_answers):
                u_ans = user_answers[i] if user_answers[i] else "No Answer"
                match_text = " (CORRECT)" if u_ans == ans else " (INCORRECT)"
                text += f" | Your Answer: {u_ans}{match_text}"
            
            c.drawString(margin, y_position, text)
            y_position -= 15
        
        c.save()
        
        # Read the PDF bytes
        with open(tmp_path, "rb") as f:
            pdf_bytes = f.read()
        
        return pdf_bytes
        
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def lock_submit() -> None:
    """Lock the submit button to prevent multiple submissions."""
    st.session_state.is_submitting = True

def extract_answers_from_pdf(pdf_bytes: bytes) -> list[str] | None:
    """
    Extract user answers from a filled PDF exam form.
    
    Args:
        pdf_bytes: PDF file bytes
        
    Returns:
        List of user answers (A, B, C, D) or None if parsing fails
    """
    if not PDF_PARSING_AVAILABLE:
        return None
    
    try:
        # Create a PDF reader object
        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
        
        # Extract form fields from the PDF
        user_answers = []
        
        # Iterate through all pages
        for page in pdf_reader.pages:
            # Check if the page has form fields
            if '/Annots' in page:
                annotations = page['/Annots']
                if annotations:
                    for annotation in annotations:
                        annotation_obj = annotation.get_object()
                        if '/T' in annotation_obj and '/V' in annotation_obj:
                            # Get the field name and value
                            field_name = annotation_obj['/T']
                            field_value = annotation_obj['/V']
                            
                            # Extract question number from field name (e.g., "q1" -> 1)
                            if isinstance(field_name, str) and field_name.startswith('q'):
                                try:
                                    q_num = int(field_name[1:])
                                    # Ensure we have enough slots in the list
                                    while len(user_answers) < q_num:
                                        user_answers.append(None)
                                    # Store the answer value (should be A, B, C, or D)
                                    user_answers[q_num - 1] = str(field_value).upper()
                                except (ValueError, IndexError):
                                    continue
        
        return user_answers if user_answers else None
        
    except Exception as e:
        st.error(f"Error parsing PDF: {e}")
        return None

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

    # Generate timestamp once for all downloads
    timestamp_melb = datetime.datetime.now(ZoneInfo("Australia/Melbourne")).strftime('%Y%m%d_%H%M%S')

    # 2. FETCH ALL USER PROGRESS DATA FROM SUPABASE
    try:
        from progress_manager import supabase
        response = supabase.table("user_progress").select("*").execute()
        
        if response.data:
            # Create an in-memory ZIP package
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for user_record in response.data:
                    username = user_record.get("username", "unknown")
                    progress_data = user_record.get("progress_data", {})
                    
                    # Create a JSON file for each user
                    json_content = json.dumps(progress_data, indent=2)
                    zip_file.writestr(f"{username}_progress.json", json_content)
            
            zip_buffer.seek(0)
            
            st.sidebar.download_button(
                label="Download All User Data (.zip)",
                data=zip_buffer,
                file_name=f"backup_user_profiles_{timestamp_melb}.zip",
                mime="application/zip",
                key="download_all_data_zip"
            )
        else:
            st.sidebar.info("No user progress data found in Supabase.")
    except Exception as e:
        st.sidebar.error(f"Error fetching data from Supabase: {str(e)}")

    # 3. RESTORE AND UNPACK USER ARCHIVES TO SUPABASE
    uploaded_zip = st.sidebar.file_uploader(
        "Restore / Migrate Profiles (.zip)", 
        type=["zip"], 
        key="upload_migration_zip"
    )

    if uploaded_zip is not None:
        if st.sidebar.button("Confirm Overwrite & Restore Data"):
            try:
                restored_count = 0
                
                with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
                    file_list = zip_ref.namelist()
                    
                    for file_name in file_list:
                        if file_name.endswith('.json'):
                            # Extract username from filename (remove _progress.json suffix)
                            username = file_name.replace('_progress.json', '').replace('.json', '')
                            
                            # Read JSON content
                            json_content = zip_ref.read(file_name)
                            progress_data = json.loads(json_content)
                            
                            # Upsert to Supabase
                            row_payload = {
                                "username": username,
                                "progress_data": progress_data
                            }
                            supabase.table("user_progress").upsert(row_payload).execute()
                            restored_count += 1
                
                st.sidebar.success(f"Successfully restored {restored_count} user profile(s) to Supabase!")
                st.sidebar.info("Please refresh or interact with the app to load profiles.")
                st.balloons()
            except Exception as e:
                st.sidebar.error(f"Migration processing error: {str(e)}")
                
    # -------------------------------------------------------------
    # 4. DOWNLOAD USER CONTRIBUTIONS
    # -------------------------------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.markdown("**User Contributions**")
    
    contrib_dir = "contributions"
    # Check if the folder exists and actually has files in it
    if os.path.exists(contrib_dir) and os.listdir(contrib_dir):
        contrib_zip_buffer = io.BytesIO()
        with zipfile.ZipFile(contrib_zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for file_name in os.listdir(contrib_dir):
                file_path = os.path.join(contrib_dir, file_name)
                if os.path.isfile(file_path):
                    zip_file.write(file_path, file_name)
        
        contrib_zip_buffer.seek(0)
        
        st.sidebar.download_button(
            label="Download All Contributed Notes (.zip)",
            data=contrib_zip_buffer,
            file_name=f"contributed_notes_{timestamp_melb}.zip",
            mime="application/zip",
            key="download_contrib_zip"
        )
    else:
        st.sidebar.info("No user contributions available yet.")

# Define the view functions
def render_trainer_page():
    global CSV_FILE
    # ==========================================
    # 0. INITIALIZATION ENGINE
    # ==========================================

    # Register the cleanup handler once
    register_cleanup_handler()

    # ==========================================
    # 1. SETUP & CONFIGURATION
    # ==========================================
    st.set_page_config(page_title="Trainer", page_icon="🩺", layout="wide")

    # ==========================================
    # 1A. PROFILE MANAGEMENT
    # ==========================================
    active_user = setup_user_profile()

    # Load saved progress on startup
    loaded_progress = load_progress(active_user)
    # ==========================================
    # GLOBAL MASTER TEMPLATE CONFIGURATION
    # ==========================================


    if not os.path.exists(CSV_FILE):
        st.error(f"Fatal Error: Master template file '{CSV_FILE}' not found!")
        st.stop()




    # ==========================================
    # 2. CORE LOGIC FUNCTIONS
    # ==========================================
    # Dynamically pick the target CSV depending on the selectbox state
    if st.session_state.get("semester") == "Y1S1":
        CSV_FILE = "learning_objectives_informative_reports_y1s1.csv"
        NOTES_FILE = "lecture_notes_y1s1.csv"
    elif st.session_state.get("semester") == "Y1S2":
        CSV_FILE = "learning_objectives_informative_reports_y1s2.csv"
        NOTES_FILE = "lecture_notes_y1s2.csv"
    elif st.session_state.get("semester") == "Y2S1":
        CSV_FILE = "learning_objectives_informative_reports_y2s1.csv"
        NOTES_FILE = "lecture_notes_y2s1.csv"
    elif st.session_state.get("semester") == "Y2S2":
        CSV_FILE = "learning_objectives_informative_reports_y2s2.csv"
        NOTES_FILE = "lecture_notes_y2s2.csv"
    else:
        CSV_FILE = "learning_objectives_informative_reports_y2s2.csv"
        NOTES_FILE = "lecture_notes_y2s2.csv"

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
        num_actual_questions = len(st.session_state.current_key)
        
        # Prevent division by zero
        if num_actual_questions > 0:
            percentage = (st.session_state.last_score / num_actual_questions * 100)
            delta_text = f"{percentage:.1f}% Correct"
            delta_color = "normal" if percentage >= 70 else "inverse"
        else:
            delta_text = "0% Correct"
            delta_color = "inverse"
            
        # Display score and leveling notification in callout boxes
        st.metric(
            label="Exam Performance", 
            value=f"{st.session_state.last_score} / {num_actual_questions}",
            delta=delta_text,
            delta_color=delta_color
        )
        
        if st.session_state.get('level_message'):
            st.info(st.session_state.level_message)
        st.write("")
        
        if st.session_state.get('immediate_wrong_breakdown'):
            st.markdown("### Answer Breakdown")
            st.markdown(st.session_state.immediate_wrong_breakdown)
        
        st.write("---")
    
    if generate_clicked:
        st.session_state.is_submitting = False
        # --- NEW: BACKUP THE CURRENT EXAM BEFORE OVERWRITING ---
        st.session_state.previous_test_data = create_exam_backup(st.session_state)
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
            df_main = load_csv_data(CSV_FILE)
            df_notes = load_csv_data(NOTES_FILE)
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
        
            # --- SMART SAMPLING (CATEGORY WEIGHTED) ---
            # Map the base category weights from your blueprint
            df['sampling_weight'] = df['category'].map(EXAM_WEIGHTS).fillna(0.05)

            # Sample using the category weights
            try:
                # We use replace=False so we don't duplicate questions in the same exam
                st.session_state.samples_df = df.sample(min(n, len(df)), weights='sampling_weight', replace=False)
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
                if not text or not isinstance(text, str):
                    return ""
                
                sentences = SENTENCE_SPLIT_PATTERN.split(text.strip())
                if len(sentences) <= 1:
                    rotated_text = text
                else:
                    start_idx = random.randint(0, len(sentences) - 1)
                    rotated_sentences = sentences[start_idx:] + sentences[:start_idx]
                    rotated_text = " ".join(rotated_sentences)
                
                # --- NEW: Word limit enforcer ---
                words = rotated_text.split()
                if len(words) > 500:
                    return " ".join(words[:500]) + "..." # Truncate and add ellipsis
                    
                return rotated_text

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

                if raw_response and "[KEY:" in raw_response:
                    # Use a split that keeps the questions separate from the key
                    text, key_part = raw_response.split("[KEY:")

                    # CLEANING: Remove the key section from the visible text
                    st.session_state.current_exam = text.strip()
                    st.session_state.current_key = ANSWER_KEY_PATTERN.findall(key_part)
                    
                    # --- PERSISTENT SAVE AT MOMENT OF GENERATION ---
                    try:
                        save_progress(st.session_state, active_user)
                    except Exception as e:
                        st.error(f"Failed to save progress to Supabase: {e}")
                    # -----------------------------------------------

                    st.rerun()
                else:
                    st.error(f"Failed to generate a perfectly formatted exam. Please click generate again.")
        except Exception as e:
            st.error(f"File Error: Ensure {CSV_FILE} and {NOTES_FILE} are in the folder. ({e})")

    st.sidebar.header("Stats & Controls")

    # Move Active Level metric here
    st.sidebar.metric("Active Level", f"{st.session_state.current_level}/50")

    df_sidebar = load_csv_data(CSV_FILE)

    # Create two tabs inside the sidebar
    filter_tab1, filter_tab2 = st.sidebar.tabs(["Exam Filter", "Lecture Filter"])

    # --- TAB 1: BLUEPRINT FILTERS (Original Logic) ---
    with filter_tab1:
        with st.form("exam_filter_form"):
            # --- Subject filter Checkboxes ---
            st.markdown("**Subjects:**")
            categories = sorted(df_sidebar['category'].fillna("Uncategorized").astype(str).unique().tolist())
            subject_filter = []
            for cat in categories:
                if st.checkbox(cat, value=True, key=f"focus_{cat}"):
                    subject_filter.append(cat)
            
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
            
            st.form_submit_button("Apply Filters")

    # --- TAB 2: LECTURE FILTERS (New Integrated Feature) ---
    with filter_tab2:
        with st.form("lecture_filter_form"):
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
            
            st.form_submit_button("Apply Lecture Filter")

    # PDF Upload Section for Grading
    st.sidebar.subheader("Submit Filled PDF")
    uploaded_pdf = st.sidebar.file_uploader(
        "Upload filled exam PDF",
        type="pdf",
        help="Upload a PDF exported from the Export Questions page with your answers filled in"
    )
    
    if uploaded_pdf is not None:
        if st.sidebar.button("Grade Uploaded PDF", use_container_width=True):
            if not PDF_PARSING_AVAILABLE:
                st.sidebar.error("PDF parsing library not available. Please install PyPDF2.")
            else:
                with st.sidebar.spinner("Parsing PDF..."):
                    pdf_bytes = uploaded_pdf.read()
                    user_answers = extract_answers_from_pdf(pdf_bytes)
                    
                    if user_answers is None:
                        st.sidebar.error("Could not extract answers from PDF.")
                    else:
                        st.session_state.uploaded_pdf_answers = user_answers
                        st.sidebar.success(f"Extracted {len(user_answers)} answers. Click 'Submit for Grading' in main view.")

    # --- NEW: RESTORE BACKUP BUTTON ---
    if st.session_state.get('previous_test_data'):
        if st.sidebar.button("Load Previous Exam", help="Accidentally clicked generate? Restore the last exam.", use_container_width=True):
            
            # Take a snapshot of the active exam before swapping, so you can toggle back and forth!
            current_backup = create_exam_backup(st.session_state)
            
            # Load the backup into the live view
            restore_exam_from_backup(st.session_state, st.session_state.previous_test_data)
            
            # Make the old current exam the new backup
            st.session_state.previous_test_data = current_backup if current_backup else {}
                
            st.rerun()
    
    # Generate New Exam button in sidebar
    st.sidebar.write("---")
    generate_clicked = st.sidebar.button(
        "Generate New Exam", 
        type="primary", 
        use_container_width=True,
        help="Click here to compile a fresh customized exam based on your filter selections."
    )
    # ----------------------------------



    # Display the Exam
    if st.session_state.current_exam:

        

        # 1. CLEANING: Remove introductory fluff and trailing keys
        clean_text = st.session_state.current_exam.strip()
        # Remove common AI intros like "Here are your questions..."
        clean_text = INTRO_CLEANUP_PATTERN.sub("", clean_text)

        # 2. SPLITTING: Look for "1. ", "2. ", etc. at the START of a line only
        # This prevents it from splitting on "1." inside a sentence
        raw_questions = QUESTION_SPLIT_PATTERN.split(clean_text)
        
        # Remove any empty strings resulting from the split
        individual_questions = [q.strip() for q in raw_questions if q.strip()]



        if 'user_selections' not in st.session_state:
            st.session_state.user_selections = {}

        # Increase radio option text size for better readability
        st.markdown("""
            <style>
            div[data-testid="stRadio"] > div > div > label > div {
                font-size: 24px !important;
            }
            </style>
        """, unsafe_allow_html=True)

        # --- 2. RENDER THE QUESTIONS DYNAMICALLY (INSIDE FORM TO PREVENT RELOADS) ---
        with st.form(key="exam_form"):
            for i, q_text in enumerate(individual_questions):
                st.subheader(f"**Question {i+1}**")
                
                # Extract clinical question text body before the option choices begin
                prompt_match = QUESTION_PROMPT_PATTERN.search(q_text)
                q_prompt = prompt_match.group(1).strip() if prompt_match else q_text
                
                # Keep the question at its native, original markdown text size
                st.markdown(q_prompt.replace("\n", "<br>"), unsafe_allow_html=True)
                st.write("") 

                # Clean option boundaries handling spacing nuances from API outputs
                opt_A = OPTION_A_PATTERN.search(q_text)
                opt_B = OPTION_B_PATTERN.search(q_text)
                opt_C = OPTION_C_PATTERN.search(q_text)
                opt_D = OPTION_D_PATTERN.search(q_text)

                options_dict = {
                    "A": opt_A.group(1).strip()[2:].strip() if opt_A else "Option A",
                    "B": opt_B.group(1).strip()[2:].strip() if opt_B else "Option B",
                    "C": opt_C.group(1).strip()[2:].strip() if opt_C else "Option C",
                    "D": opt_D.group(1).strip()[2:].strip() if opt_D else "Option D"
                }

                current_selection = st.session_state.user_selections.get(i, None)

                # Safely convert keys to list and track indexing position mapping
                choice_keys = list(options_dict.keys())
                default_index = choice_keys.index(current_selection) if current_selection in choice_keys else None

                # Render the clean native layout with left-aligned circular radio dots
                st.radio(
                    label=f"Options for Question {i+1}",
                    options=choice_keys,
                    index=default_index,
                    format_func=lambda x: f"{x.lower()}. {options_dict[x]}",
                    key=f"radio_q_{i}",
                    disabled=st.session_state.get('exam_submitted', False),
                    label_visibility="collapsed"
                )

                # --- NEW: INJECT PER-QUESTION AI FEEDBACK RIGHT HERE ---
                if st.session_state.get('exam_submitted') and st.session_state.get('ai_feedback_clean'):
                    # Look for the section matching "### Question X" or "### Question [X]"
                    feedback_str = st.session_state.ai_feedback_clean
                    pattern = rf"### Question \s*\[?{i+1}\]?.*?(?=### Question \s*\[?{i+2}\]?|---|$)"
                    match = re.search(pattern, feedback_str, re.DOTALL | re.IGNORECASE)

                    if match:
                        correct_ans = st.session_state.current_key[i]
                        user_ans = st.session_state.user_selections.get(i, "No Answer")
                        st.error(f"**Your Answer:** {user_ans} | **Correct Answer:** {correct_ans}")
                        # Remove the "### Question X" heading and answer line from the feedback
                        feedback_text = match.group(0).strip()
                        feedback_text = re.sub(r'^### Question \s*\[?\d+\]?.*?\n', '', feedback_text, flags=re.IGNORECASE)
                        feedback_text = re.sub(r'^\*\*Your Answer:.*?\n', '', feedback_text, flags=re.IGNORECASE)
                        st.markdown(feedback_text.strip())
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
                st.write("Click the button on the right to scroll up >>>")

            # Standalone execution grading submission action button (normalized look)
            submitted = st.form_submit_button("Submit for Grading", disabled=st.session_state.get('is_submitting', False) or st.session_state.get('exam_submitted', False))
        if submitted:
            # Check if we have uploaded PDF answers to use instead of form selections
            if 'uploaded_pdf_answers' in st.session_state and st.session_state.uploaded_pdf_answers:
                # Use answers extracted from PDF
                num_actual_questions = len(individual_questions)
                pdf_answers = st.session_state.uploaded_pdf_answers
                
                # Ensure we have the right number of answers
                user_answers = pdf_answers[:num_actual_questions] if len(pdf_answers) >= num_actual_questions else pdf_answers + [None] * (num_actual_questions - len(pdf_answers))
                
                # Update user_selections to match PDF answers
                for idx, ans in enumerate(user_answers):
                    st.session_state.user_selections[idx] = ans
                
                user_input = "\n".join([f"Q{idx+1}: {ans if ans else 'No Answer'}" for idx, ans in enumerate(user_answers)])
                
                # Clear the uploaded PDF answers after use
                st.session_state.uploaded_pdf_answers = None
            else:
                # Capture selections from form radio buttons (original logic)
                num_actual_questions = len(individual_questions)
                for idx in range(num_actual_questions):
                    st.session_state.user_selections[idx] = st.session_state.get(f"radio_q_{idx}")
                
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
                    "semester": st.session_state.get("semester", "Y2S2"),
                    "category": st.session_state.current_categories[i] if i < len(st.session_state.current_categories) else "General",
                })
                
                if u_ans == correct:
                    score += 1
                else:
                    clean_q_snippet = BR_CLEANUP_PATTERN.sub(' ', individual_questions[i].split('\n')[0][:120])
                    incorrect_summary_markdown += f"**Question {i+1}:** *{clean_q_snippet}...*\n"
                    incorrect_summary_markdown += f"&nbsp;&nbsp;&nbsp;&nbsp;• **Your Answer:** `{u_ans}` | **Correct Answer:** `{correct}`\n\n"
                    
                    # 2. ONLY add to the missed questions bank if it was wrong
                    st.session_state.missed_questions.append({
                        "question": individual_questions[i].strip(),
                        "correct": correct,
                        "yours": u_ans,
                        "semester": st.session_state.get("semester", "Y2S2"),
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
                save_progress(st.session_state, st.session_state.get("username", "Default"))
            except Exception as e:
                st.error(f"Failed to save progress after grading: {e}")

            percentage_correct = (score / num_actual_questions) * 100
            if (num_actual_questions - score) <= 1 or percentage_correct >= LEVEL_UP_THRESHOLD:
                next_level = min(MAX_LEVEL, st.session_state.current_level + 1)
                if next_level > st.session_state.current_level:
                    st.session_state.level_message = f"**Excellent performance ({percentage_correct:.0f}%)! You have leveled up to Level {next_level}!**"
                else:
                    st.session_state.level_message = f"**Fantastic score ({percentage_correct:.0f}%)! You are at the maximum mastery level (Level {MAX_LEVEL})!**"
                st.session_state.current_level = next_level
            elif percentage_correct <= LEVEL_DOWN_THRESHOLD:
                next_level = max(MIN_LEVEL, st.session_state.current_level - 1)
                if next_level < st.session_state.current_level:
                    st.session_state.level_message = f"**Score was {percentage_correct:.0f}%. The system adjusted your difficulty down to Level {next_level} to rebuild foundations.**"
                else:
                    st.session_state.level_message = f"**Score was {percentage_correct:.0f}%. You are at Level {MIN_LEVEL}. Keep practicing to build confidence!**"
                st.session_state.current_level = next_level
            else:
                st.session_state.level_message = f"**Solid effort ({percentage_correct:.0f}%)! Remaining at Level {st.session_state.current_level} to lock in consistency.**"
            
            st.rerun()
            
            

    # Missed Questions Bank in Sidebar
    if st.session_state.missed_questions:
        # Filter to only include mistakes from the active semester
        current_semester = st.session_state.get("semester", "Y2S2")
        semester_filtered_mistakes = [
            item for item in st.session_state.missed_questions
            if isinstance(item, dict) and item.get('semester') == current_semester
        ]
        
        # 2. YOUR ORIGINAL SIDEBAR EXPORT (Preserved)
        st.sidebar.subheader(f"Missed Questions ({len(semester_filtered_mistakes)})")
        # Prepare the missed questions text content dynamically in memory
        melbourne_mistakes_time = datetime.datetime.now(ZoneInfo("Australia/Melbourne")).strftime('%Y-%m-%d %H:%M')
        export_text = f"=== WEB SESSION (Melbourne Time): {melbourne_mistakes_time} ===\n"
        export_text += f"=== SEMESTER: {current_semester} ===\n"

        for item in semester_filtered_mistakes:
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
    active_user = setup_user_profile()

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
        current_sem = st.session_state.get("semester", "Y2S2").upper()
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
    if PDF_AVAILABLE:
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
    clean_exam_text = EXAM_CLEANUP_PATTERN.sub(r'\1 \n ', raw_exam_text)
    
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

def render_game_page():
    """Timed quiz game page - answer as many questions as possible in 1 minute."""
    register_cleanup_handler()
    active_user = setup_user_profile()
    
    st.title("🎮 Speed Quiz Challenge")
    st.write("---")
    
    # Initialize game state
    if 'game_questions' not in st.session_state:
        st.session_state.game_questions = []
    if 'game_answers' not in st.session_state:
        st.session_state.game_answers = []
    if 'game_current_index' not in st.session_state:
        st.session_state.game_current_index = 0
    if 'game_score' not in st.session_state:
        st.session_state.game_score = 0
    if 'game_start_time' not in st.session_state:
        st.session_state.game_start_time = None
    if 'game_active' not in st.session_state:
        st.session_state.game_active = False
    if 'game_completed' not in st.session_state:
        st.session_state.game_completed = False
    if 'game_score_saved' not in st.session_state:
        st.session_state.game_score_saved = False
    if 'game_settings' not in st.session_state:
        st.session_state.game_settings = {
            'difficulty': st.session_state.current_level,
            'num_questions': 20,
            'categories': []
        }
    
    # Load CSV data for filtering
    df_main = load_csv_data(CSV_FILE)
    df_notes = load_csv_data(NOTES_FILE)
    df = pd.merge(df_main, df_notes, on=JOIN_COLUMN, how='left')
    if "semester" in df.columns:
        df = df[df['semester'] == st.session_state.semester]
    
    # Get available categories
    available_categories = df['subject'].unique().tolist() if 'subject' in df.columns else []
    
    # Settings phase
    if not st.session_state.game_active and not st.session_state.game_completed:
        st.subheader("⚙️ Game Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.game_settings['difficulty'] = st.slider(
                "Difficulty Level",
                1, 50,
                st.session_state.game_settings['difficulty'],
                help="Higher difficulty = more complex questions"
            )
            st.session_state.game_settings['num_questions'] = st.slider(
                "Number of Questions to Generate",
                10, 50,
                st.session_state.game_settings['num_questions'],
                help="Questions will be pre-generated before the game starts"
            )
        
        with col2:
            st.session_state.game_settings['categories'] = st.multiselect(
                "Categories (leave empty for all)",
                available_categories,
                default=st.session_state.game_settings['categories'],
                help="Filter questions by specific subjects"
            )
        
        
        if st.button("🚀 Start Game", type="primary", use_container_width=True):
            # Generate questions before starting
            with st.spinner("Generating questions... This may take a moment."):
                # Filter data based on settings
                filtered_df = df.copy()
                if st.session_state.game_settings['categories']:
                    filtered_df = filtered_df[filtered_df['subject'].isin(st.session_state.game_settings['categories'])]
                
                if filtered_df.empty:
                    st.error("No content available for selected filters. Please adjust your settings.")
                    return
                
                # Sample content for question generation
                num_to_sample = min(st.session_state.game_settings['num_questions'], len(filtered_df))
                samples_df = filtered_df.sample(n=num_to_sample, replace=False)
                topics_list = samples_df['content'].dropna().head(st.session_state.game_settings['num_questions']).tolist()
                
                # Generate questions
                exam_text = get_blind_exam(
                    topics_list,
                    st.session_state.game_settings['difficulty'],
                    st.session_state.game_settings['num_questions']
                )
                
                if exam_text:
                    # Parse the generated exam
                    st.session_state.game_questions = parse_exam_text(exam_text)
                    st.session_state.game_answers = extract_answer_key(exam_text)
                    
                    if not st.session_state.game_questions or not st.session_state.game_answers:
                        st.error("Failed to parse generated questions. Please try again.")
                        return
                    
                    st.session_state.game_current_index = 0
                    st.session_state.game_score = 0
                    st.session_state.game_user_answers = []
                    st.session_state.game_active = True
                    st.session_state.game_start_time = time.time()
                    st.rerun()
                else:
                    st.error("Failed to generate questions. Please try again.")
    
    # Active game phase
    elif st.session_state.game_active:
        # Calculate remaining time
        elapsed = time.time() - st.session_state.game_start_time
        remaining = max(0, 60 - elapsed)
        
        # Display timer with JavaScript for smooth updates
        timer_html = f"""
        <div style="text-align: center; padding: 20px;">
            <div style="font-size: 24px; font-weight: bold; color: {'#ff6b6b' if remaining <= 10 else '#4CAF50'};">
                ⏱️ Time Remaining: <span id="timer">{remaining:.1f}</span>s
            </div>
        </div>
        <script>
            var startTime = {time.time()};
            var timeLimit = 60;
            var timerElement = document.getElementById('timer');
            
            function updateTimer() {{
                var elapsed = (Date.now() / 1000) - startTime;
                var remaining = Math.max(0, timeLimit - elapsed);
                timerElement.textContent = remaining.toFixed(1);
                
                if (remaining <= 0) {{
                    timerElement.style.color = '#ff6b6b';
                }}
                
                if (remaining > 0) {{
                    requestAnimationFrame(updateTimer);
                }}
            }}
            
            updateTimer();
        </script>
        """
        components.html(timer_html, height=80)
        
        # Check if time is up
        if remaining <= 0:
            st.session_state.game_active = False
            st.session_state.game_completed = True
            st.rerun()
            return
        
        # Display current question
        if st.session_state.game_current_index < len(st.session_state.game_questions):
            current_q = st.session_state.game_questions[st.session_state.game_current_index]
            
            with st.form(f"game_question_form_{st.session_state.game_current_index}"):
                st.markdown(f"### Question {st.session_state.game_current_index + 1}")
                st.markdown(current_q['question'])
                
                # Display options (matching trainer page layout)
                options_dict = current_q['options']
                choice_keys = list(options_dict.keys())
                user_answer = st.radio(
                    "Select your answer:",
                    options=choice_keys,
                    format_func=lambda x: f"{x.lower()}. {options_dict[x]}",
                    key=f"game_q_{st.session_state.game_current_index}",
                    label_visibility="collapsed"
                )
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    submitted = st.form_submit_button("Submit Answer", type="primary", use_container_width=True)
                
                if submitted:
                    # Check if user selected an answer
                    if user_answer is None:
                        st.warning("Please select an answer before submitting.")
                        st.rerun()
                        return
                    
                    # Record answer
                    correct_answer = st.session_state.game_answers[st.session_state.game_current_index]
                    is_correct = user_answer == correct_answer
                    
                    if is_correct:
                        st.session_state.game_score += 1
                    
                    st.session_state.game_user_answers.append({
                        'question_index': st.session_state.game_current_index,
                        'user_answer': user_answer,
                        'correct_answer': correct_answer,
                        'is_correct': is_correct
                    })
                    
                    # Move to next question
                    st.session_state.game_current_index += 1
                    st.rerun()
        else:
            # All questions answered
            st.session_state.game_active = False
            st.session_state.game_completed = True
            st.rerun()
    
    # Results phase
    elif st.session_state.game_completed:
        st.subheader("🏆 Game Results")
        
        total_answered = len(st.session_state.game_user_answers)
        total_questions = len(st.session_state.game_questions)
        accuracy = (st.session_state.game_score / total_answered * 100) if total_answered > 0 else 0
        time_taken = 60.0  # Fixed 60 seconds
        
        # Save score to leaderboard (only once and only if opted in)
        if 'game_score_saved' not in st.session_state or not st.session_state.game_score_saved:
            if st.session_state.leaderboard_opt_in:
                save_single_player_score(
                    active_user,
                    st.session_state.game_score,
                    total_answered,
                    accuracy,
                    st.session_state.game_settings['difficulty'],
                    time_taken,
                    st.session_state.game_settings['categories']
                )
            st.session_state.game_score_saved = True
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Correct Answers", st.session_state.game_score)
        with col2:
            st.metric("Questions Answered", f"{total_answered}/{total_questions}")
        with col3:
            st.metric("Accuracy", f"{accuracy:.1f}%")
        
        st.write("---")
        
        # Show answer breakdown
        if st.session_state.game_user_answers:
            st.subheader("Answer Breakdown")
            for answer in st.session_state.game_user_answers:
                q_idx = answer['question_index']
                q_data = st.session_state.game_questions[q_idx]
                
                if answer['is_correct']:
                    st.success(f"Q{q_idx + 1}: ✅ Correct ({answer['user_answer']})")
                else:
                    st.error(f"Q{q_idx + 1}: ❌ Wrong (You: {answer['user_answer']} | Correct: {answer['correct_answer']})")
                
                with st.expander(f"View Question {q_idx + 1}"):
                    st.markdown(q_data['question'])
        
        st.write("---")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Play Again", type="primary", use_container_width=True):
                # Reset game state
                st.session_state.game_questions = []
                st.session_state.game_answers = []
                st.session_state.game_current_index = 0
                st.session_state.game_score = 0
                st.session_state.game_start_time = None
                st.session_state.game_active = False
                st.session_state.game_completed = False
                st.session_state.game_user_answers = []
                st.session_state.game_score_saved = False
                st.rerun()
        
        with col2:
            if st.button("🏠 Back to Settings", use_container_width=True):
                st.session_state.game_questions = []
                st.session_state.game_answers = []
                st.session_state.game_current_index = 0
                st.session_state.game_score = 0
                st.session_state.game_start_time = None
                st.session_state.game_active = False
                st.session_state.game_completed = False
                st.session_state.game_user_answers = []
                st.session_state.game_score_saved = False
                st.rerun()

def render_multiplayer_page():
    """1v1 multiplayer quiz game - GeoGuessr style."""
    register_cleanup_handler()
    active_user = setup_user_profile()
    
    st.title("⚔️ 1v1 Multiplayer Challenge")
    st.write("---")
    
    # Initialize multiplayer state
    if 'mp_room_code' not in st.session_state:
        st.session_state.mp_room_code = None
    if 'mp_is_host' not in st.session_state:
        st.session_state.mp_is_host = False
    if 'mp_questions' not in st.session_state:
        st.session_state.mp_questions = []
    if 'mp_answers' not in st.session_state:
        st.session_state.mp_answers = []
    if 'mp_current_index' not in st.session_state:
        st.session_state.mp_current_index = 0
    if 'mp_score' not in st.session_state:
        st.session_state.mp_score = 0
    if 'mp_user_answers' not in st.session_state:
        st.session_state.mp_user_answers = []
    if 'mp_game_active' not in st.session_state:
        st.session_state.mp_game_active = False
    if 'mp_game_completed' not in st.session_state:
        st.session_state.mp_game_completed = False
    if 'mp_elo_updated' not in st.session_state:
        st.session_state.mp_elo_updated = False
    if 'mp_opponent_progress' not in st.session_state:
        st.session_state.mp_opponent_progress = {'current_index': 0, 'score': 0}
    if 'mp_game_start_time' not in st.session_state:
        st.session_state.mp_game_start_time = None
    
    # Load CSV data
    df_main = load_csv_data(CSV_FILE)
    df_notes = load_csv_data(NOTES_FILE)
    df = pd.merge(df_main, df_notes, on=JOIN_COLUMN, how='left')
    if "semester" in df.columns:
        df = df[df['semester'] == st.session_state.semester]
    
    available_categories = df['category'].unique().tolist() if 'category' in df.columns else []
    
    # LOBBY PHASE
    if not st.session_state.mp_room_code:
        st.subheader("🎯 Game Lobby")
        
        tab1, tab2 = st.tabs(["Create Room", "Join Room"])
        
        with tab1:
            st.write("Create a new game room and share the code with a friend.")
            
            col1, col2 = st.columns(2)
            with col1:
                difficulty = st.slider("Difficulty Level", 1, 50, st.session_state.current_level)
                num_questions = st.slider("Number of Questions", 5, 30, 10)
            with col2:
                all_subjects = st.checkbox("All Subjects", value=True, help="Include questions from all subjects")
                if not all_subjects:
                    categories = st.multiselect("Select Subjects", available_categories, help="Choose specific subjects to focus on")
                else:
                    categories = []
            
            if st.button("🏠 Create Room", type="primary", use_container_width=True):
                # Generate room code
                import random
                import string
                room_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
                
                # Generate questions
                with st.spinner("Generating questions..."):
                    filtered_df = df.copy()
                    if categories:
                        filtered_df = filtered_df[filtered_df['category'].isin(categories)]
                    
                    if filtered_df.empty:
                        st.error("No content available for selected filters.")
                        return
                    
                    num_to_sample = min(num_questions, len(filtered_df))
                    samples_df = filtered_df.sample(n=num_to_sample, replace=False)
                    topics_list = samples_df['content'].dropna().head(num_questions).tolist()
                    
                    exam_text = get_blind_exam(topics_list, difficulty, num_questions)
                    
                    if exam_text:
                        questions = parse_exam_text(exam_text)
                        answers = extract_answer_key(exam_text)
                        
                        if questions and answers:
                            # Create room in Supabase
                            room_data = {
                                'room_code': room_code,
                                'host': active_user,
                                'player2': None,
                                'status': 'waiting',
                                'questions': json.dumps(questions),
                                'answers': json.dumps(answers),
                                'host_progress': json.dumps({'current_index': 0, 'score': 0}),
                                'player2_progress': json.dumps({'current_index': 0, 'score': 0}),
                                'host_answers': json.dumps([]),
                                'player2_answers': json.dumps([]),
                                'created_at': datetime.datetime.now().isoformat()
                            }
                            
                            try:
                                supabase.table('multiplayer_rooms').insert(room_data).execute()
                                st.session_state.mp_room_code = room_code
                                st.session_state.mp_is_host = True
                                st.session_state.mp_questions = questions
                                st.session_state.mp_answers = answers
                                st.success(f"Room created! Share this code with your friend: **{room_code}**")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to create room: {e}")
                        else:
                            st.error("Failed to parse questions.")
                    else:
                        st.error("Failed to generate questions.")
        
        with tab2:
            st.write("Join an existing game room using the code provided by your friend.")
            
            room_code_input = st.text_input("Enter Room Code", max_chars=6, help="6-character code").upper()
            
            if st.button("🚀 Join Room", type="primary", use_container_width=True):
                if len(room_code_input) != 6:
                    st.error("Invalid room code. Must be 6 characters.")
                    return
                
                # Check if room exists
                try:
                    response = supabase.table('multiplayer_rooms').select('*').eq('room_code', room_code_input).execute()
                    
                    if not response.data:
                        st.error("Room not found. Check the code and try again.")
                        return
                    
                    room = response.data[0]
                    
                    if room['status'] != 'waiting':
                        st.error("This room is already full or the game has started.")
                        return
                    
                    if room['host'] == active_user:
                        st.error("You cannot join your own room.")
                        return
                    
                    # Join the room
                    supabase.table('multiplayer_rooms').update({
                        'player2': active_user,
                        'status': 'ready'
                    }).eq('room_code', room_code_input).execute()
                    
                    st.session_state.mp_room_code = room_code_input
                    st.session_state.mp_is_host = False
                    st.session_state.mp_questions = json.loads(room['questions'])
                    st.session_state.mp_answers = json.loads(room['answers'])
                    st.success("Joined room successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Failed to join room: {e}")
    
    # WAITING FOR OPPONENT (HOST ONLY)
    elif st.session_state.mp_room_code and not st.session_state.mp_game_active and not st.session_state.mp_game_completed:
        if st.session_state.mp_is_host:
            st.subheader(f"🏠 Room: {st.session_state.mp_room_code}")
            st.info("Waiting for opponent to join...")
            
            # Poll for opponent
            try:
                response = supabase.table('multiplayer_rooms').select('*').eq('room_code', st.session_state.mp_room_code).execute()
                if response.data:
                    room = response.data[0]
                    if room['player2'] and room['status'] == 'ready':
                        st.success(f"**{room['player2']}** has joined! Starting game...")
                        st.session_state.mp_game_active = True
                        st.session_state.mp_game_start_time = time.time()
                        st.rerun()
                    else:
                        st.write(f"Share this code with a friend: **{st.session_state.mp_room_code}**")
            except Exception as e:
                st.error(f"Error checking room status: {e}")
            
            if st.button("Cancel", use_container_width=True):
                # Delete room
                try:
                    supabase.table('multiplayer_rooms').delete().eq('room_code', st.session_state.mp_room_code).execute()
                except:
                    pass
                st.session_state.mp_room_code = None
                st.session_state.mp_questions = []
                st.session_state.mp_answers = []
                st.rerun()
        
        else:
            # Player 2 waiting for host to start
            st.subheader(f"🏠 Room: {st.session_state.mp_room_code}")
            st.info("Waiting for host to start the game...")
            
            try:
                response = supabase.table('multiplayer_rooms').select('*').eq('room_code', st.session_state.mp_room_code).execute()
                if response.data:
                    room = response.data[0]
                    if room['status'] == 'active':
                        st.session_state.mp_game_active = True
                        st.session_state.mp_game_start_time = time.time()
                        st.rerun()
            except Exception as e:
                st.error(f"Error checking room status: {e}")
    
    # ACTIVE GAME PHASE
    elif st.session_state.mp_game_active and not st.session_state.mp_game_completed:
        # Poll opponent progress
        try:
            response = supabase.table('multiplayer_rooms').select('*').eq('room_code', st.session_state.mp_room_code).execute()
            if response.data:
                room = response.data[0]
                if st.session_state.mp_is_host:
                    opponent_progress = json.loads(room['player2_progress'])
                else:
                    opponent_progress = json.loads(room['host_progress'])
                st.session_state.mp_opponent_progress = opponent_progress
        except:
            pass
        
        # Live scoreboard
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Your Score", st.session_state.mp_score)
        with col2:
            st.metric("Your Progress", f"{st.session_state.mp_current_index}/{len(st.session_state.mp_questions)}")
        with col3:
            opponent_name = "Opponent"
            st.metric(f"{opponent_name} Score", st.session_state.mp_opponent_progress['score'])
        
        st.write("---")
        
        # Display current question
        if st.session_state.mp_current_index < len(st.session_state.mp_questions):
            current_q = st.session_state.mp_questions[st.session_state.mp_current_index]
            
            with st.form(f"mp_question_form_{st.session_state.mp_current_index}"):
                st.markdown(f"### Question {st.session_state.mp_current_index + 1}/{len(st.session_state.mp_questions)}")
                st.markdown(current_q['question'])
                
                user_answer = st.radio(
                    "Select your answer:",
                    options=['A', 'B', 'C', 'D'],
                    key=f"mp_q_{st.session_state.mp_current_index}",
                    horizontal=True
                )
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    submitted = st.form_submit_button("Submit Answer", type="primary", use_container_width=True)
                
                if submitted:
                    # Check if user selected an answer
                    if user_answer is None:
                        st.warning("Please select an answer before submitting.")
                        st.rerun()
                        return
                    
                    correct_answer = st.session_state.mp_answers[st.session_state.mp_current_index]
                    is_correct = user_answer == correct_answer
                    
                    if is_correct:
                        st.session_state.mp_score += 1
                    
                    st.session_state.mp_user_answers.append({
                        'question_index': st.session_state.mp_current_index,
                        'user_answer': user_answer,
                        'correct_answer': correct_answer,
                        'is_correct': is_correct
                    })
                    
                    # Update progress in Supabase
                    progress = {
                        'current_index': st.session_state.mp_current_index + 1,
                        'score': st.session_state.mp_score
                    }
                    
                    try:
                        if st.session_state.mp_is_host:
                            supabase.table('multiplayer_rooms').update({
                                'host_progress': json.dumps(progress),
                                'host_answers': json.dumps(st.session_state.mp_user_answers),
                                'status': 'active'
                            }).eq('room_code', st.session_state.mp_room_code).execute()
                        else:
                            supabase.table('multiplayer_rooms').update({
                                'player2_progress': json.dumps(progress),
                                'player2_answers': json.dumps(st.session_state.mp_user_answers),
                                'status': 'active'
                            }).eq('room_code', st.session_state.mp_room_code).execute()
                    except:
                        pass
                    
                    st.session_state.mp_current_index += 1
                    st.rerun()
        else:
            # All questions answered
            st.session_state.mp_game_active = False
            st.session_state.mp_game_completed = True
            st.rerun()
    
    # RESULTS PHASE
    elif st.session_state.mp_game_completed:
        st.subheader("🏆 Game Results")
        
        # Fetch final opponent data
        try:
            response = supabase.table('multiplayer_rooms').select('*').eq('room_code', st.session_state.mp_room_code).execute()
            if response.data:
                room = response.data[0]
                if st.session_state.mp_is_host:
                    opponent_answers = json.loads(room['player2_answers'])
                    opponent_progress = json.loads(room['player2_progress'])
                    opponent_name = room['player2']
                else:
                    opponent_answers = json.loads(room['host_answers'])
                    opponent_progress = json.loads(room['host_progress'])
                    opponent_name = room['host']
                
                opponent_score = opponent_progress['score']
            else:
                opponent_answers = []
                opponent_score = 0
                opponent_name = "Opponent"
        except:
            opponent_answers = []
            opponent_score = 0
            opponent_name = "Opponent"
        
        # Determine result and update ELO
        if st.session_state.mp_score > opponent_score:
            result = "win"
            result_display = "🎉 You Win!"
        elif opponent_score > st.session_state.mp_score:
            result = "loss"
            result_display = "😢 You Lose"
        else:
            result = "tie"
            result_display = "🤝 It's a Tie!"
        
        # Update ELO ratings (only once per game completion and only if opted in)
        if 'mp_elo_updated' not in st.session_state or not st.session_state.mp_elo_updated:
            if opponent_name and opponent_name != "Opponent" and st.session_state.leaderboard_opt_in:
                update_player_elo(active_user, opponent_name, result)
                st.session_state.mp_elo_updated = True
        
        # Display comparison
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Your Score", st.session_state.mp_score, 
                      delta="WINNER!" if result == "win" else "")
        with col2:
            st.metric(f"{opponent_name}'s Score", opponent_score,
                      delta="WINNER!" if result == "loss" else "")
        with col3:
            if result == "win":
                st.success(result_display)
            elif result == "loss":
                st.error(result_display)
            else:
                st.info(result_display)
        
        st.write("---")
        
        # Answer comparison
        st.subheader("Answer Comparison")
        for i, q in enumerate(st.session_state.mp_questions):
            your_answer = st.session_state.mp_user_answers[i] if i < len(st.session_state.mp_user_answers) else None
            opp_answer = opponent_answers[i] if i < len(opponent_answers) else None
            correct = st.session_state.mp_answers[i]
            
            your_status = "✅" if your_answer and your_answer['is_correct'] else "❌"
            opp_status = "✅" if opp_answer and opp_answer['is_correct'] else "❌"
            
            your_ans = your_answer['user_answer'] if your_answer else "-"
            opp_ans = opp_answer['user_answer'] if opp_answer else "-"
            
            st.markdown(f"**Q{i+1}**: Correct: {correct} | You: {your_ans} {your_status} | {opponent_name}: {opp_ans} {opp_status}")
            
            with st.expander(f"View Question {i+1}"):
                st.markdown(q['question'])
        
        st.write("---")
        
        if st.button("🏠 Back to Lobby", use_container_width=True):
            # Clean up room
            try:
                supabase.table('multiplayer_rooms').delete().eq('room_code', st.session_state.mp_room_code).execute()
            except:
                pass
            
            st.session_state.mp_room_code = None
            st.session_state.mp_is_host = False
            st.session_state.mp_questions = []
            st.session_state.mp_answers = []
            st.session_state.mp_current_index = 0
            st.session_state.mp_score = 0
            st.session_state.mp_user_answers = []
            st.session_state.mp_game_active = False
            st.session_state.mp_game_completed = False
            st.session_state.mp_opponent_progress = {'current_index': 0, 'score': 0}
            st.session_state.mp_elo_updated = False
            st.session_state.mp_game_start_time = None
            st.rerun()

def render_leaderboard_page():
    """Leaderboard page showing single player and multiplayer rankings."""
    register_cleanup_handler()
    active_user = setup_user_profile()
    
    st.title("🏆 Leaderboard")
    st.write("---")
    
    # Fetch leaderboard data
    leaderboard_data = get_leaderboard_data()
    elo_leaderboard = leaderboard_data.get('elo_leaderboard', [])
    single_leaderboard = leaderboard_data.get('single_player_leaderboard', [])
    accuracy_leaderboard = leaderboard_data.get('accuracy_leaderboard', [])
    questions_leaderboard = leaderboard_data.get('questions_leaderboard', [])
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Total Questions Completed", "🎯 Overall Accuracy", "⚔️ 1v1 ELO Rankings", "🎮 Single Player Best Scores"])
    
    with tab1:
        st.subheader("Total Questions Completed - All Time")
        
        if not questions_leaderboard:
            st.info("No question data yet. Play the Speed Quiz to get on the leaderboard!")
        else:
            # Display questions leaderboard table
            for idx, entry in enumerate(questions_leaderboard, 1):
                with st.container():
                    col1, col2, col3 = st.columns([1, 2, 2])
                    
                    # Rank badge
                    if idx == 1:
                        rank_badge = "🥇"
                    elif idx == 2:
                        rank_badge = "🥈"
                    elif idx == 3:
                        rank_badge = "🥉"
                    else:
                        rank_badge = f"#{idx}"
                    
                    col1.markdown(f"### {rank_badge}")
                    col2.markdown(f"**{entry['username']}**")
                    col3.metric("Total Questions", entry['total_questions'])
                    
                    st.divider()
        
        # Show user's total questions
        if questions_leaderboard:
            user_questions = next((e for e in questions_leaderboard if e['username'] == active_user), None)
            if user_questions:
                st.info(f"📊 Your total questions completed: **{user_questions['total_questions']}**")
            else:
                st.info("You don't have question data yet. Play the Speed Quiz to get ranked!")
    
    with tab2:
        st.subheader("Overall Accuracy - All Time")
        
        if not accuracy_leaderboard:
            st.info("No accuracy data yet. Play the Speed Quiz to get on the leaderboard!")
        else:
            # Display accuracy leaderboard table
            for idx, entry in enumerate(accuracy_leaderboard, 1):
                with st.container():
                    col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 2])
                    
                    # Rank badge
                    if idx == 1:
                        rank_badge = "🥇"
                    elif idx == 2:
                        rank_badge = "🥈"
                    elif idx == 3:
                        rank_badge = "🥉"
                    else:
                        rank_badge = f"#{idx}"
                    
                    col1.markdown(f"### {rank_badge}")
                    col2.markdown(f"**{entry['username']}**")
                    col3.metric("Overall Accuracy", f"{entry['overall_accuracy']:.1f}%")
                    col4.metric("Correct", entry['total_correct'])
                    col5.metric("Total Questions", entry['total_questions'])
                    
                    st.divider()
        
        # Show user's overall accuracy
        if accuracy_leaderboard:
            user_accuracy = next((e for e in accuracy_leaderboard if e['username'] == active_user), None)
            if user_accuracy:
                st.info(f"🎯 Your overall accuracy: **{user_accuracy['overall_accuracy']:.1f}%** ({user_accuracy['total_correct']}/{user_accuracy['total_questions']} correct)")
            else:
                st.info("You don't have accuracy data yet. Play the Speed Quiz to get ranked!")

    with tab3:
        st.subheader("1v1 Multiplayer - ELO Rankings")
        
        if not elo_leaderboard:
            st.info("No ELO ratings yet. Play 1v1 Multiplayer to get ranked!")
        else:
            # Display ELO leaderboard table
            for idx, entry in enumerate(elo_leaderboard, 1):
                with st.container():
                    col1, col2, col3, col4, col5, col6 = st.columns([1, 2, 2, 2, 2, 2])
                    
                    # Rank badge
                    if idx == 1:
                        rank_badge = "🥇"
                    elif idx == 2:
                        rank_badge = "🥈"
                    elif idx == 3:
                        rank_badge = "🥉"
                    else:
                        rank_badge = f"#{idx}"
                    
                    col1.markdown(f"### {rank_badge}")
                    col2.markdown(f"**{entry['username']}**")
                    # Use Glicko rating with fallback to old elo_rating for backward compatibility
                    rating = entry.get('rating', entry.get('elo_rating', 1500))
                    rd = entry.get('rd', 350)
                    col3.metric("Rating", f"{rating:.0f} ±{rd:.0f}")
                    col4.metric("W/L/T", f"{entry['games_won']}/{entry['games_lost']}/{entry['games_tied']}")
                    col5.metric("Games", entry['games_played'])
                    
                    # Win rate calculation
                    if entry['games_played'] > 0:
                        win_rate = (entry['games_won'] / entry['games_played']) * 100
                        col6.metric("Win Rate", f"{win_rate:.1f}%")
                    else:
                        col6.metric("Win Rate", "0%")
                    
                    st.divider()
        
        # Show user's Glicko rating
        if elo_leaderboard:
            user_elo = next((e for e in elo_leaderboard if e['username'] == active_user), None)
            if user_elo:
                # Use Glicko rating with fallback to old elo_rating for backward compatibility
                rating = user_elo.get('rating', user_elo.get('elo_rating', 1500))
                rd = user_elo.get('rd', 350)
                st.info(f"⚔️ Your Glicko rating: **{rating:.0f} ±{rd:.0f}** (Games: {user_elo['games_played']}, W/L/T: {user_elo['games_won']}/{user_elo['games_lost']}/{user_elo['games_tied']})")
            else:
                st.info("You don't have a Glicko rating yet. Play 1v1 Multiplayer to get ranked!")

    with tab4:
        st.subheader("Single Player Speed Quiz - Top Scores")
        
        if not single_leaderboard:
            st.info("No scores recorded yet. Play the Speed Quiz to get on the leaderboard!")
        else:
            # Display leaderboard table
            for idx, entry in enumerate(single_leaderboard, 1):
                with st.container():
                    col1, col2, col3, col4, col5, col6 = st.columns([1, 2, 2, 2, 2, 3])
                    
                    # Rank badge
                    if idx == 1:
                        rank_badge = "🥇"
                    elif idx == 2:
                        rank_badge = "🥈"
                    elif idx == 3:
                        rank_badge = "🥉"
                    else:
                        rank_badge = f"#{idx}"
                    
                    col1.markdown(f"### {rank_badge}")
                    col2.markdown(f"**{entry['username']}**")
                    col3.metric("Score", entry['score'])
                    col4.metric("Accuracy", f"{entry['accuracy']:.1f}%")
                    col5.metric("Difficulty", entry['difficulty'])
                    col6.caption(f"{entry['created_at'][:10]}")
                    
                    if entry.get('categories'):
                        categories_str = ', '.join(entry['categories']) if isinstance(entry['categories'], list) else str(entry['categories'])
                        col6.caption(f"Categories: {categories_str}")
                    
                    st.divider()
        
        # Show user's best score
        if single_leaderboard:
            user_scores = [s for s in single_leaderboard if s['username'] == active_user]
            if user_scores:
                user_best = max(user_scores, key=lambda x: x['score'])
                st.info(f"🎯 Your best score: **{user_best['score']}** (Accuracy: {user_best['accuracy']:.1f}%)")

def parse_exam_text(exam_text: str) -> list:
    """Parse exam text into structured question format."""
    questions = []
    individual_questions = QUESTION_SPLIT_PATTERN.split(exam_text.strip())
    
    for q_text in individual_questions:
        if not q_text.strip():
            continue
        
        # Extract question text
        prompt_match = QUESTION_PROMPT_PATTERN.search(q_text)
        question_text = prompt_match.group(1).strip() if prompt_match else q_text
        
        # Extract options
        opt_A = OPTION_A_PATTERN.search(q_text)
        opt_B = OPTION_B_PATTERN.search(q_text)
        opt_C = OPTION_C_PATTERN.search(q_text)
        opt_D = OPTION_D_PATTERN.search(q_text)
        
        options = {
            'A': opt_A.group(1).strip()[2:].strip() if opt_A else "",
            'B': opt_B.group(1).strip()[2:].strip() if opt_B else "",
            'C': opt_C.group(1).strip()[2:].strip() if opt_C else "",
            'D': opt_D.group(1).strip()[2:].strip() if opt_D else ""
        }
        
        questions.append({
            'question': question_text,
            'options': options
        })
    
    return questions

def extract_answer_key(exam_text: str) -> list:
    """Extract answer key from exam text."""
    # Look for [KEY: A, B, C, ...] pattern at the end
    key_match = re.search(r'\[KEY:\s*([^\]]+)\]', exam_text)
    if key_match:
        key_string = key_match.group(1).strip()
        # Parse the key string
        answers = [ans.strip() for ans in key_string.split(',')]
        return answers
    
    # Fallback: try to extract from the last line
    lines = exam_text.strip().split('\n')
    if lines:
        last_line = lines[-1].strip()
        if 'KEY:' in last_line:
            key_part = last_line.split('KEY:')[1].strip()
            answers = [ans.strip() for ans in key_part.split(',')]
            return answers
    
    return []

def render_settings_page():
    st.title("⚙️ Settings")
    st.write("---")
    
    active_user = setup_user_profile()
    
    # 1. Main Configuration Sliders
    st.subheader("Manually adjust")
    st.session_state.current_level = st.slider("Starting Level", 1, 50, st.session_state.current_level)
    st.session_state.num_questions = st.slider("Number of Questions", 1, 50, st.session_state.num_questions)
    
    # Define the options exactly as you want them
    semester_options = ["Y1S1", "Y1S2", "Y2S1", "Y2S2"]

    # Look up what is currently saved in session state to determine the starting index (default to 0 if not found)
    current_semester = st.session_state.get("semester", "Y2S2")
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
    
    st.session_state.temperature = st.slider(
        "Question Creativity 1",
        min_value=0.0,
        max_value=2.0,
        value=st.session_state.temperature,
        step=0.1,
        help="Controls randomness: Lower (0.0) = more deterministic, Higher (2.0) = more creative/random"
    )
    
    st.session_state.top_p = st.slider(
        "Question Creativity 2",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.top_p,
        step=0.05,
        help="Nucleus sampling: Lower (0.0) = more focused, Higher (1.0) = more diverse"
    )
    
    st.write("---")
    
    # Leaderboard opt-in setting
    st.subheader("Leaderboard Privacy")
    st.session_state.leaderboard_opt_in = st.checkbox(
        "Show my scores on the leaderboard",
        value=st.session_state.leaderboard_opt_in,
        help="When enabled, your scores will appear on the public leaderboard. When disabled, your scores are private."
    )
    
    st.write("---")
    

    if st.button("Undo last submission"):
        if 'pre_submit_backup' in st.session_state and st.session_state['pre_submit_backup']:
            # Overwrite the database with the pre-submission backup
            save_progress(st.session_state['pre_submit_backup'], st.session_state.username)
            
            st.success("Previous save restored successfully! Please refresh the page.")
            # Clear the backup so it can't be spammed
            del st.session_state['pre_submit_backup'] 
        else:
            st.info("No previous submission backup found to restore.")

    st.write("---")
    
    # API Key Usage Display
    st.subheader("API Key Usage")
    
    for i, model_counts in st.session_state.api_request_counts.items():
        is_current = i == st.session_state.key_index
        status = "🟢 Active" if is_current else "⚪ Available"
        
        st.markdown(f"**Key {i + 1}** {status}")
        
        # Show usage for each model
        col1, col2 = st.columns(2)
        
        with col1:
            flash36_count = model_counts.get('gemini-3.6-flash', 0)
            flash36_max = MAX_REQUESTS_PER_KEY_PER_MODEL['gemini-3.6-flash']
            flash36_percent = min((flash36_count / flash36_max) * 100, 100)
            
            st.metric(
                "3.6 Flash",
                f"{flash36_count}/{flash36_max}",
                delta_color="normal" if flash36_count < flash36_max else "inverse"
            )
            st.progress(flash36_percent / 100)
        
        with col2:
            flash35lite_count = model_counts.get('gemini-3.5-flash-lite', 0)
            flash35lite_max = MAX_REQUESTS_PER_KEY_PER_MODEL['gemini-3.5-flash-lite']
            flash35lite_percent = min((flash35lite_count / flash35lite_max) * 100, 100)
            
            st.metric(
                "3.5 Flash Lite",
                f"{flash35lite_count}/{flash35lite_max}",
                delta_color="normal" if flash35lite_count < flash35lite_max else "inverse"
            )
            st.progress(flash35lite_percent / 100)
        
        st.divider()
    
    st.write("---")
    # 2. Speed Switch
    st.subheader("Model Selection")
    model_choice = st.radio(
        label="",
        options=["3.5 flash lite", "3.6 flash"],
        index=1 if st.session_state.get('exam_model', 'gemini-3.6-flash') == 'gemini-3.6-flash' else 0,
        horizontal=True,
        help="Fast uses Flash-Lite. Slow & Smart uses Flash."
    )
    st.session_state.exam_model = 'gemini-3.6-flash' if "Slow" in model_choice else 'gemini-3.5-flash-lite'
    
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
                    # Clear progress from Supabase instead of local file
                    try:
                        from progress_manager import supabase
                        supabase.table("user_progress").delete().eq("username", active_user).execute()
                    except Exception as e:
                        st.error(f"Failed to clear progress from Supabase: {e}")
                        
                    initialize_app(active_user, force_reset=True)
                    st.session_state.confirm_reset = False  # Reset flag state
                    st.success("Progress reset successfully!")
                    st.rerun()
                    
            with sub_col2:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.confirm_reset = False  # Dismiss confirmation
                    st.rerun()
    with col2:
        st.markdown("### Contribute")
        uploaded_note = st.file_uploader("Upload Notes to Contribute", type=["txt", "csv", "pdf"])

        if uploaded_note is not None:
            # Save it to a designated directory or process it into your database
            save_path = os.path.join("contributions", uploaded_note.name)
            os.makedirs("contributions", exist_ok=True)
            
            with open(save_path, "wb") as f:
                f.write(uploaded_note.getbuffer())
            
            st.success("Thank you for contributing! Your notes have been submitted.")
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
    st.Page(render_game_page, title="Speed Quiz", icon="🎮", url_path="game"),
    # st.Page(render_multiplayer_page, title="1v1 Multiplayer", icon="⚔️", url_path="multiplayer"),
    st.Page(render_leaderboard_page, title="Leaderboard", icon="🏆", url_path="leaderboard"),
    st.Page(render_settings_page, title="Settings", icon="⚙️", url_path="settings")
])
pg.run()