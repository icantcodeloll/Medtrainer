from shiny import App, ui, render, reactive, reactive_calc
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

# Configuration - These would normally come from secrets
# For Shiny, we'll need to handle secrets differently
API_KEYS = []  # User needs to configure these
MAX_REQUESTS_PER_KEY_PER_MODEL = {
    'gemini-3.6-flash': 20,
    'gemini-3.5-flash-lite': 500
}
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
EXAM_MODEL = 'gemini-3.5-flash-lite'
GRADER_MODEL = 'gemini-3.5-flash-lite'

# Constants
LEVEL_UP_THRESHOLD = 90
LEVEL_DOWN_THRESHOLD = 60
MAX_LEVEL = 50
MIN_LEVEL = 1

# Compiled regex patterns
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
def create_exam_backup(state_dict) -> dict:
    """Create a backup dictionary of the current exam state."""
    if not state_dict.get('current_exam'):
        return {}
    
    return {
        'current_exam': state_dict['current_exam'],
        'current_key': state_dict['current_key'],
        'user_selections': state_dict.get('user_selections', {}),
        'exam_submitted': state_dict.get('exam_submitted', False),
        'last_score': state_dict.get('last_score', 0),
        'last_user_input': state_dict.get('last_user_input', ""),
        'last_correct_key': state_dict.get('last_correct_key', ""),
        'last_user_answers_list': state_dict.get('last_user_answers_list', []),
        'current_categories': state_dict.get('current_categories', []),
        'samples_df': state_dict.get('samples_df', None)
    }

def restore_exam_from_backup(state_dict, backup: dict):
    """Restore exam state from a backup dictionary."""
    if not backup:
        return
        
    state_dict['current_exam'] = backup.get('current_exam')
    state_dict['current_key'] = backup.get('current_key')
    state_dict['user_selections'] = backup.get('user_selections', {})
    state_dict['exam_submitted'] = backup.get('exam_submitted', False)
    state_dict['last_score'] = backup.get('last_score', 0)
    state_dict['last_user_input'] = backup.get('last_user_input', "")
    state_dict['last_correct_key'] = backup.get('last_correct_key', "")
    state_dict['last_user_answers_list'] = backup.get('last_user_answers_list', [])
    state_dict['current_categories'] = backup.get('current_categories', [])
    state_dict['samples_df'] = backup.get('samples_df', None)

def validate_username(username: str) -> str:
    """Validate and sanitize username input."""
    if not username:
        return "Default"
    
    sanitized = username.strip()
    sanitized = sanitized[:50]
    sanitized = USERNAME_SANITIZE_PATTERN.sub('', sanitized)
    
    return sanitized if sanitized else "Default"

def load_csv_data(file_path: str) -> pd.DataFrame:
    """Load CSV data."""
    return pd.read_csv(file_path)

def get_client(api_key: str) -> genai.Client:
    """Get the current Gemini AI client."""
    return genai.Client(api_key=api_key)

def rotate_to_next_available_key(key_index: int, api_request_counts: dict, model: str, api_keys: list) -> tuple:
    """Rotate to the next API key that hasn't exceeded the request limit."""
    keys_checked = 0
    original_index = key_index
    max_requests = MAX_REQUESTS_PER_KEY_PER_MODEL.get(model, 500)
    
    while keys_checked < len(api_keys):
        current_index = key_index
        key_counts = api_request_counts.get(current_index, {'gemini-3.6-flash': 0, 'gemini-3.5-flash-lite': 0})
        request_count = key_counts.get(model, 0)
        
        if request_count < max_requests:
            return key_index, True  # Current key is still available
        
        key_index = (key_index + 1) % len(api_keys)
        keys_checked += 1
        
        if key_index == original_index and keys_checked > 0:
            return key_index, False
    
    return key_index, False

def call_gemini_with_rotation(prompt: str, model_to_use: str, 
                             api_keys: list, key_index: int, api_request_counts: dict,
                             use_search: bool = False, thinking_level: str = "MEDIUM", 
                             temperature: float = 0.7, top_p: float = 0.95) -> tuple:
    """Call Gemini API with key rotation."""
    keys_tried = 0
    max_requests = MAX_REQUESTS_PER_KEY_PER_MODEL.get(model_to_use, 500)
    
    current_key_index, available = rotate_to_next_available_key(key_index, api_request_counts, model_to_use, api_keys)
    
    if not available:
        return None, current_key_index, api_request_counts, f"All API keys have reached their request limit for {model_to_use}"
    
    tools = []
    if use_search and "3.6-flash" in model_to_use.lower():
        tools = [types.Tool(google_search=types.GoogleSearch())]
    
    config_args = {}
    if tools:
        config_args["tools"] = tools
        
    if "3.5-flash-lite" in model_to_use.lower() or "3.6-flash" in model_to_use.lower():
        config_args["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)
    
    config_args["temperature"] = temperature
    config_args["top_p"] = top_p
    
    generation_config = types.GenerateContentConfig(**config_args)
    
    while keys_tried < len(api_keys):
        key_counts = api_request_counts.get(current_key_index, {'gemini-3.6-flash': 0, 'gemini-3.5-flash-lite': 0})
        current_count = key_counts.get(model_to_use, 0)
        
        if current_count >= max_requests:
            current_key_index, available = rotate_to_next_available_key(current_key_index, api_request_counts, model_to_use, api_keys)
            if not available:
                return None, current_key_index, api_request_counts, f"All API keys have reached their request limit for {model_to_use}"
            keys_tried += 1
            continue
        
        try:
            client = get_client(api_keys[current_key_index])
            response = client.models.generate_content(
                model=model_to_use,
                contents=prompt,
                config=generation_config
            )
            
            if current_key_index not in api_request_counts:
                api_request_counts[current_key_index] = {'gemini-3.6-flash': 0, 'gemini-3.5-flash-lite': 0}
            api_request_counts[current_key_index][model_to_use] = \
                api_request_counts[current_key_index].get(model_to_use, 0) + 1
            
            return response.text, current_key_index, api_request_counts, None
        except Exception as e:
            if "429" in str(e):
                keys_tried += 1
                if keys_tried >= len(api_keys):
                    return None, current_key_index, api_request_counts, "Reduce the question count."
                current_key_index = (current_key_index + 1) % len(api_keys)
                time.sleep(1)
            elif "503" in str(e):
                time.sleep(5)
            else:
                return None, current_key_index, api_request_counts, f"Error during generation: {e}"
    
    return None, current_key_index, api_request_counts, "Failed to generate response"

def get_blind_exam(topics_list: list, level: int, num_questions: int, exam_model: str,
                  api_keys: list, key_index: int, api_request_counts: dict,
                  use_search: bool = False, thinking_level: str = "MEDIUM", 
                  temperature: float = 0.7, top_p: float = 0.95) -> tuple:
    """Generate a blind exam using AI."""
    combined_content = "\n\n".join([f"Source {i+1}: {t}" for i, t in enumerate(topics_list)])

    # Difficulty calibration
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
    else:
        difficulty_desc = "supreme counterintuition - advanced mastery of medical paradoxes and exceptions"
        complexity_guide = "multi-system integration where standard rules don't apply, latest research breakthroughs that overturn conventional wisdom, complex clinical reasoning requiring recognition of exceptions, niche subspecialty knowledge where intuitive answers are wrong, molecular-level pathophysiology that defies simple explanations, emerging treatment protocols with paradoxical mechanisms, rare disease patterns that mimic opposite conditions, advanced diagnostic challenges where the obvious answer is incorrect, cutting-edge research that contradicts established dogma"

    options_pool = ['A', 'B', 'C', 'D'] * ((num_questions // 4) + 1)
    dynamic_keys = random.sample(options_pool, num_questions)
    formatted_key_string = ", ".join(dynamic_keys)

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

    exam_text, new_key_index, new_api_counts, error = call_gemini_with_rotation(
        prompt, exam_model, api_keys, key_index, api_request_counts,
        use_search, thinking_level, temperature, top_p
    )
    
    return exam_text, new_key_index, new_api_counts, error

def get_ai_grading(exam_text: str, user_answers: str, correct_key: str, score: int,
                   api_keys: list, key_index: int, api_request_counts: dict,
                   use_search: bool = False) -> tuple:
    """Generate AI grading feedback for exam answers."""
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
    
    return call_gemini_with_rotation(prompt, GRADER_MODEL, use_search, api_keys, key_index, api_request_counts)

def create_exam_pdf(exam_text: str, answer_key: list, user_answers: list = None, 
                   score: int = None, max_score: int = None, metadata: dict = None) -> bytes | None:
    """Generate a PDF containing the exam questions."""
    if not PDF_AVAILABLE:
        return None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp_path = tmp.name
    
    try:
        c = canvas.Canvas(tmp_path, pagesize=letter)
        width, height = letter
        margin = 72
        y_position = height - margin
        
        c.setFont("Helvetica-Bold", 16)
        if score is not None and max_score is not None:
            title = f"Practice Exam Results - Score: {score}/{max_score}"
        else:
            title = "Practice Exam"
        c.drawCentredString(width / 2, y_position, title)
        y_position -= 30
        
        if metadata:
            c.setFont("Helvetica-Oblique", 9)
            melbourne_time = datetime.datetime.now(ZoneInfo("Australia/Melbourne")).strftime('%Y-%m-%d %H:%M:%S')
            meta_text = f"Level: {metadata.get('level', 'N/A')} | Subject: {metadata.get('subject', 'All')} | Exam Filter: {metadata.get('exam', 'All')} | System Filter: {metadata.get('system', 'All')}"
            time_text = f"Generated on (Melbourne Time): {melbourne_time}"
            c.drawCentredString(width / 2, y_position, meta_text)
            y_position -= 15
            c.drawCentredString(width / 2, y_position, time_text)
            y_position -= 25
        
        individual_questions = QUESTION_SPLIT_PATTERN.split(exam_text.strip())
        
        c.setFont("Helvetica", 10)
        
        for q_idx, q_text in enumerate(individual_questions):
            if y_position < 100:
                c.showPage()
                y_position = height - margin
            
            c.setFont("Helvetica-Bold", 12)
            c.drawString(margin, y_position, f"Question {q_idx + 1}")
            y_position -= 20
            
            prompt_match = QUESTION_PROMPT_PATTERN.search(q_text)
            q_prompt = prompt_match.group(1).strip() if prompt_match else q_text
            
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
            
            for opt_idx, (opt_letter, opt_text) in enumerate(options):
                if y_position < margin + 20:
                    c.showPage()
                    y_position = height - margin
                
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
                
                opt_label = f"{opt_letter.lower()}. {opt_text}"
                c.drawString(margin + 30, y_position - 6, opt_label)
                y_position -= 20
            
            y_position -= 15
        
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
        
        with open(tmp_path, "rb") as f:
            pdf_bytes = f.read()
        
        return pdf_bytes
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def extract_answers_from_pdf(pdf_bytes: bytes) -> list | None:
    """Extract user answers from a filled PDF exam form."""
    if not PDF_PARSING_AVAILABLE:
        return None
    
    try:
        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
        user_answers = []
        
        for page in pdf_reader.pages:
            if '/Annots' in page:
                annotations = page['/Annots']
                if annotations:
                    for annotation in annotations:
                        annotation_obj = annotation.get_object()
                        if '/T' in annotation_obj and '/V' in annotation_obj:
                            field_name = annotation_obj['/T']
                            field_value = annotation_obj['/V']
                            
                            if isinstance(field_name, str) and field_name.startswith('q'):
                                try:
                                    q_num = int(field_name[1:])
                                    while len(user_answers) < q_num:
                                        user_answers.append(None)
                                    user_answers[q_num - 1] = str(field_value).upper()
                                except (ValueError, IndexError):
                                    continue
        
        return user_answers if user_answers else None
        
    except Exception as e:
        return None

def parse_exam_text(exam_text: str) -> list:
    """Parse exam text into structured question format."""
    questions = []
    individual_questions = QUESTION_SPLIT_PATTERN.split(exam_text.strip())
    
    for q_text in individual_questions:
        if not q_text.strip():
            continue
        
        prompt_match = QUESTION_PROMPT_PATTERN.search(q_text)
        question_text = prompt_match.group(1).strip() if prompt_match else q_text
        
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
    key_match = re.search(r'\[KEY:\s*([^\]]+)\]', exam_text)
    if key_match:
        key_string = key_match.group(1).strip()
        answers = [ans.strip() for ans in key_string.split(',')]
        return answers
    
    lines = exam_text.strip().split('\n')
    if lines:
        last_line = lines[-1].strip()
        if 'KEY:' in last_line:
            key_part = last_line.split('KEY:')[1].strip()
            answers = [ans.strip() for ans in key_part.split(',')]
            return answers
    
    return []

# Shiny App Definition
app_ui = ui.page_fluid(
    ui.head_content(
        ui.tags.style("""
            .question-container { margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }
            .option-label { margin-left: 10px; }
            .metric-card { padding: 15px; border: 1px solid #ddd; border-radius: 8px; text-align: center; }
            .sidebar-section { margin-bottom: 20px; padding: 15px; background: #f8f9fa; border-radius: 8px; }
        """)
    ),
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_text("username", "Enter your username:", value="Default"),
            ui.input_action_button("switch_profile", "Switch / Create Profile"),
            ui.output_text("user_display"),
            ui.hr(),
            ui.input_slider("current_level", "Active Level", 1, 50, 1),
            ui.output_text("level_display"),
            ui.hr(),
            ui.navset_card_underline(
                ui.nav_panel("Exam Filter",
                    ui.input_checkbox_group("subject_filter", "Subjects:", choices=[]),
                    ui.input_checkbox_group("exam_filter", "Exam:", choices=[]),
                    ui.input_checkbox_group("system_filter", "Systems:", choices=[]),
                    ui.input_action_button("apply_filters", "Apply Filters")
                ),
                ui.nav_panel("Lecture Filter",
                    ui.input_select("lecture_filter", "Select Lecture:", choices=[], multiple=True),
                    ui.input_action_button("apply_lecture_filter", "Apply Lecture Filter")
                )
            ),
            ui.hr(),
            ui.input_file("uploaded_pdf", "Upload filled exam PDF", accept=[".pdf"]),
            ui.input_action_button("grade_pdf", "Grade Uploaded PDF"),
            ui.output_text("pdf_status"),
            ui.hr(),
            ui.input_action_button("load_previous", "Load Previous Exam"),
            ui.input_action_button("generate_exam", "Generate New Exam"),
        ),
        ui.navset_tab(
            ui.nav_panel("Exam Trainer",
                ui.output_ui("trainer_ui")
            ),
            ui.nav_panel("Stats",
                ui.output_ui("stats_ui")
            ),
            ui.nav_panel("Export Questions",
                ui.output_ui("export_ui")
            ),
            ui.nav_panel("Speed Quiz",
                ui.output_ui("game_ui")
            ),
            # ui.nav_panel("1v1 Multiplayer",
            #     ui.output_ui("multiplayer_ui")
            # ),
            ui.nav_panel("Leaderboard",
                ui.output_ui("leaderboard_ui")
            ),
            ui.nav_panel("Settings",
                ui.output_ui("settings_ui")
            )
        )
    )
)

def server(input, output, session):
    # Reactive state management
    state = reactive.Value({
        "username": "Default",
        "current_level": 1,
        "num_questions": 5,
        "semester": "Y2S2",
        "missed_questions": [],
        "exam_history": [],
        "current_exam": "",
        "current_key": [],
        "key_index": 0,
        "api_request_counts": {},
        "current_categories": [],
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
        "leaderboard_opt_in": False,
        "uploaded_pdf_answers": None,
        "exam_model": "gemini-3.5-flash-lite",
        "samples_df": pd.DataFrame(),
        "ai_feedback_clean": "",
        "level_message": "",
        "immediate_wrong_breakdown": ""
    })
    
    # API keys configuration (user needs to set these)
    api_keys = reactive.Value([])
    api_keys_input = reactive.Value("")
    
    @reactive.calc
    def get_state():
        return state()
    
    @output
    @render.text
    def user_display():
        current_state = get_state()
        return f"Logged in as: {current_state['username']}"
    
    @output
    @render.text
    def level_display():
        current_state = get_state()
        return f"Level: {current_state['current_level']}/50"
    
    @reactive.effect
    def _():
        if input.switch_profile():
            new_username = validate_username(input.username())
            current_state = get_state()
            current_state["username"] = new_username
            # Clear session data
            keys_to_clear = ['current_level', 'num_questions', 'missed_questions', 'exam_history', 'current_exam', 'current_key', 'samples_df']
            for k in keys_to_clear:
                if k in current_state:
                    current_state[k] = current_state.get(k, 1 if k == 'current_level' else 5 if k == 'num_questions' else [])
            state.set(current_state)
    
    @reactive.effect
    def _():
        if input.generate_exam():
            current_state = get_state()
            
            # Backup current exam
            current_state["previous_test_data"] = create_exam_backup(current_state)
            
            # Reset exam state
            current_state["exam_submitted"] = False
            current_state["last_score"] = 0
            current_state["user_selections"] = {}
            current_state["last_user_answers_list"] = []
            current_state["samples_df"] = pd.DataFrame()
            current_state["ai_feedback_clean"] = ""
            current_state["level_message"] = ""
            current_state["immediate_wrong_breakdown"] = ""
            
            n = current_state["num_questions"]
            
            try:
                df_main = load_csv_data(CSV_FILE)
                df_notes = load_csv_data(NOTES_FILE)
                df = pd.merge(df_main, df_notes, on=JOIN_COLUMN, how='left')
                
                if "semester" in df.columns:
                    df = df[df['semester'] == current_state["semester"]]
                
                # Apply filters
                subject_filter = input.subject_filter()
                exam_filter = input.exam_filter()
                system_filter = input.system_filter()
                
                if subject_filter:
                    df = df[df['category'].isin(subject_filter)]
                if exam_filter and 'exam' in df.columns:
                    df = df[df['exam'].isin(exam_filter)]
                if system_filter and 'system' in df.columns:
                    df = df[df['system'].isin(system_filter)]
                
                if df.empty:
                    ui.notification_show("No questions found matching your filter criteria.", duration=5)
                    return
                
                if 'include' in df.columns:
                    df = df[df['include'].astype(str).str.lower().str.strip() == 'y']
                    if df.empty:
                        ui.notification_show("No active objectives found.", duration=5)
                        return
                
                # Smart sampling
                df['sampling_weight'] = df['category'].map(EXAM_WEIGHTS).fillna(0.05)
                try:
                    samples_df = df.sample(min(n, len(df)), weights='sampling_weight', replace=False)
                except ValueError:
                    samples_df = df.sample(min(n, len(df)))
                
                current_state["samples_df"] = samples_df
                
                if 'category' in samples_df.columns:
                    current_state["current_categories"] = samples_df['category'].fillna('General').tolist()
                else:
                    current_state["current_categories"] = ['General'] * n
                
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
                    
                    words = rotated_text.split()
                    if len(words) > 500:
                        return " ".join(words[:500]) + "..."
                        
                    return rotated_text
                
                def combine_row_text(row):
                    explanation = str(row.get('explanation', '')).strip()
                    content = str(row.get('content', '')).strip()
                    flashcards = str(row.get('flashcards', '')).strip()
                    
                    valid_segments = [seg for seg in [explanation, content, flashcards] if seg]
                    return " ".join(valid_segments)
                
                combined_raw_text = samples_df.apply(combine_row_text, axis=1)
                samples = combined_raw_text.apply(randomize_paragraph_start).tolist()
                
                ui.notification_show(f"Generating {n} questions at Level {current_state['current_level']}...", duration=10)
                
                exam_text, new_key_index, new_api_counts, error = get_blind_exam(
                    samples, current_state["current_level"], n, current_state["exam_model"],
                    current_state["use_search"], current_state["thinking_level"],
                    current_state["temperature"], current_state["top_p"],
                    api_keys(), current_state["key_index"], current_state["api_request_counts"]
                )
                
                if error:
                    ui.notification_show(f"Error: {error}", duration=10, type="error")
                    return
                
                if exam_text and "[KEY:" in exam_text:
                    text, key_part = exam_text.split("[KEY:")
                    current_state["current_exam"] = text.strip()
                    current_state["current_key"] = ANSWER_KEY_PATTERN.findall(key_part)
                    current_state["key_index"] = new_key_index
                    current_state["api_request_counts"] = new_api_counts
                    
                    # Save progress
                    try:
                        save_progress(current_state, current_state["username"])
                    except Exception as e:
                        ui.notification_show(f"Failed to save progress: {e}", duration=5, type="error")
                    
                    state.set(current_state)
                else:
                    ui.notification_show("Failed to generate exam. Please try again.", duration=5, type="error")
                    
            except Exception as e:
                ui.notification_show(f"Error: {e}", duration=10, type="error")
    
    @output
    @render.ui
    def trainer_ui():
        current_state = get_state()
        
        if not current_state["current_exam"]:
            return ui.TagList(
                ui.h2("Trainer"),
                ui.p("Click 'Generate New Exam' in the sidebar to start.")
            )
        
        clean_text = current_state["current_exam"].strip()
        clean_text = INTRO_CLEANUP_PATTERN.sub("", clean_text)
        raw_questions = QUESTION_SPLIT_PATTERN.split(clean_text)
        individual_questions = [q.strip() for q in raw_questions if q.strip()]
        
        question_elements = []
        
        for i, q_text in enumerate(individual_questions):
            prompt_match = QUESTION_PROMPT_PATTERN.search(q_text)
            q_prompt = prompt_match.group(1).strip() if prompt_match else q_text
            
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
            
            question_elements.append(
                ui.div(
                    ui.h3(f"Question {i+1}"),
                    ui.p(q_prompt.replace("\n", "<br>")),
                    ui.input_radio(
                        f"q_{i}",
                        None,
                        choices={"A": f"a. {options_dict['A']}", "B": f"b. {options_dict['B']}", 
                                "C": f"c. {options_dict['C']}", "D": f"d. {options_dict['D']}"},
                        selected=current_state["user_selections"].get(i)
                    ),
                    class_="question-container"
                )
            )
        
        return ui.TagList(
            ui.h2("Trainer"),
            ui.input_action_button("submit_exam", "Submit for Grading"),
            ui.div(*question_elements)
        )
    
    @reactive.effect
    def _():
        if input.submit_exam():
            current_state = get_state()
            
            clean_text = current_state["current_exam"].strip()
            clean_text = INTRO_CLEANUP_PATTERN.sub("", clean_text)
            raw_questions = QUESTION_SPLIT_PATTERN.split(clean_text)
            individual_questions = [q.strip() for q in raw_questions if q.strip()]
            
            # Capture user selections
            for i in range(len(individual_questions)):
                selection = input.get(f"q_{i}")
                if selection:
                    current_state["user_selections"][i] = selection.upper()
            
            num_actual_questions = len(individual_questions)
            user_answers = [current_state["user_selections"].get(idx, None) for idx in range(num_actual_questions)]
            user_input = "\n".join([f"Q{idx+1}: {ans if ans else 'No Answer'}" for idx, ans in enumerate(user_answers)])
            
            correct_key = current_state["current_key"][:num_actual_questions]
            correct_key_formatted = "\n".join([f"Q{idx+1}: {ans}" for idx, ans in enumerate(correct_key)])
            
            current_state["last_user_input"] = user_input
            current_state["last_correct_key"] = correct_key_formatted
            current_state["last_user_answers_list"] = user_answers
            
            if len(user_answers) != len(correct_key):
                ui.notification_show(f"Mismatch: {len(correct_key)} questions vs {len(user_answers)} answers", duration=5, type="error")
                return
            
            score = 0
            for i, u_ans in enumerate(user_answers):
                correct = correct_key[i] if i < len(correct_key) else None
                
                current_state["exam_history"].append({
                    "question": individual_questions[i].strip(),
                    "correct": correct,
                    "yours": u_ans if u_ans else "No Answer",
                    "semester": current_state["semester"],
                    "category": current_state["current_categories"][i] if i < len(current_state["current_categories"]) else "General",
                })
                
                if u_ans == correct:
                    score += 1
                else:
                    current_state["missed_questions"].append({
                        "question": individual_questions[i].strip(),
                        "correct": correct,
                        "yours": u_ans if u_ans else "No Answer",
                        "semester": current_state["semester"],
                        "category": current_state["current_categories"][i] if i < len(current_state["current_categories"]) else "General",
                    })
            
            current_state["exam_submitted"] = True
            current_state["last_score"] = score
            
            # Get AI feedback
            feedback, _, _, error = get_ai_grading(
                current_state["current_exam"], user_input, correct_key_formatted, score,
                api_keys(), current_state["key_index"], current_state["api_request_counts"],
                current_state["use_search"]
            )
            
            current_state["ai_feedback_clean"] = feedback if feedback else f"Error generating explanation: {error}"
            
            # Level adjustment
            percentage_correct = (score / num_actual_questions) * 100
            if (num_actual_questions - score) <= 1 or percentage_correct >= LEVEL_UP_THRESHOLD:
                next_level = min(MAX_LEVEL, current_state["current_level"] + 1)
                if next_level > current_state["current_level"]:
                    current_state["level_message"] = f"Excellent performance ({percentage_correct:.0f}%)! Leveled up to Level {next_level}!"
                else:
                    current_state["level_message"] = f"Fantastic score ({percentage_correct:.0f}%)! At maximum mastery level (Level {MAX_LEVEL})!"
                current_state["current_level"] = next_level
            elif percentage_correct <= LEVEL_DOWN_THRESHOLD:
                next_level = max(MIN_LEVEL, current_state["current_level"] - 1)
                if next_level < current_state["current_level"]:
                    current_state["level_message"] = f"Score was {percentage_correct:.0f}%. Difficulty adjusted down to Level {next_level}."
                else:
                    current_state["level_message"] = f"Score was {percentage_correct:.0f}%. At Level {MIN_LEVEL}. Keep practicing!"
                current_state["current_level"] = next_level
            else:
                current_state["level_message"] = f"Solid effort ({percentage_correct:.0f}%)! Remaining at Level {current_state['current_level']}."
            
            state.set(current_state)
    
    @output
    @render.ui
    def stats_ui():
        current_state = get_state()
        history_data = current_state["exam_history"]
        
        if not history_data:
            return ui.TagList(
                ui.h2("Stats"),
                ui.p("No exam submissions recorded yet.")
            )
        
        df_history = pd.DataFrame(history_data)
        current_sem = current_state["semester"].upper()
        if "semester" in df_history.columns:
            df_history = df_history[df_history['semester'].str.upper() == current_sem]
        
        if df_history.empty:
            return ui.TagList(
                ui.h2("Stats"),
                ui.p(f"No history records found for semester: {current_sem}")
            )
        
        total_completed = len(df_history)
        df_history['is_correct'] = df_history['yours'] == df_history['correct']
        total_correct = df_history['is_correct'].sum()
        overall_accuracy = (total_correct / total_completed) * 100 if total_completed > 0 else 0
        
        subject_stats = df_history.groupby('category').agg(
            Total=('is_correct', 'count'),
            Correct=('is_correct', 'sum')
        ).reset_index()
        
        subject_stats['Accuracy (%)'] = (subject_stats['Correct'] / subject_stats['Total']) * 100
        subject_stats = subject_stats.sort_values(by='Accuracy (%)', ascending=False)
        
        return ui.TagList(
            ui.h2("Stats"),
            ui.div(
                ui.div(ui.p(f"Total Questions Done: {total_completed} Qs"), class_="metric-card"),
                ui.div(ui.p(f"Overall Accuracy: {overall_accuracy:.1f}%"), class_="metric-card"),
                ui.div(ui.p(f"Correct Answers: {total_correct}"), class_="metric-card"),
                ui.div(ui.p(f"Current Level: Lvl {current_state['current_level']} / 50"), class_="metric-card"),
                style="display: flex; gap: 10px; margin-bottom: 20px;"
            ),
            ui.h3("Performance Breakdown by Medical Specialty"),
            ui.data_frame(subject_stats)
        )
    
    @output
    @render.ui
    def export_ui():
        current_state = get_state()
        
        if not current_state["current_exam"]:
            return ui.TagList(
                ui.h2("Export Questions"),
                ui.p("No active exam found. Generate an exam first.")
            )
        
        return ui.TagList(
            ui.h2("Export Questions"),
            ui.p("Download your current exam in your preferred format."),
            ui.download_button("download_pdf", "Download Exam as PDF", 
                               content=lambda: create_exam_pdf(current_state["current_exam"], current_state["current_key"])),
            ui.download_button("download_txt", "Download Exam as TXT",
                               content=lambda: current_state["current_exam"])
        )
    
    @output
    @render.ui
    def game_ui():
        return ui.TagList(
            ui.h2("Speed Quiz Challenge"),
            ui.p("Timed quiz game - coming soon!")
        )
    
    @output
    @render.ui
    def multiplayer_ui():
        return ui.TagList(
            ui.h2("1v1 Multiplayer Challenge"),
            ui.p("Multiplayer game - coming soon!")
        )
    
    @output
    @render.ui
    def leaderboard_ui():
        leaderboard_data = get_leaderboard_data()
        elo_leaderboard = leaderboard_data.get('elo_leaderboard', [])
        single_leaderboard = leaderboard_data.get('single_player_leaderboard', [])
        
        return ui.TagList(
            ui.h2("Leaderboard"),
            ui.h3("1v1 ELO Rankings"),
            ui.p("ELO rankings - coming soon!"),
            ui.h3("Single Player Best Scores"),
            ui.p("Single player scores - coming soon!")
        )
    
    @output
    @render.ui
    def settings_ui():
        current_state = get_state()
        
        return ui.TagList(
            ui.h2("Settings"),
            ui.input_slider("settings_level", "Starting Level", 1, 50, current_state["current_level"]),
            ui.input_slider("settings_questions", "Number of Questions", 1, 50, current_state["num_questions"]),
            ui.input_select("settings_semester", "Active Semester", 
                          choices=["Y1S1", "Y1S2", "Y2S1", "Y2S2"],
                          selected=current_state["semester"]),
            ui.input_select("settings_thinking", "Gemini Thinking Level",
                          choices=["MINIMAL", "LOW", "MEDIUM", "HIGH"],
                          selected=current_state["thinking_level"]),
            ui.input_slider("settings_temp", "Question Creativity 1", 0.0, 2.0, current_state["temperature"], step=0.1),
            ui.input_slider("settings_top_p", "Question Creativity 2", 0.0, 1.0, current_state["top_p"], step=0.05),
            ui.input_checkbox("settings_leaderboard", "Show my scores on leaderboard", 
                            value=current_state["leaderboard_opt_in"]),
            ui.hr(),
            ui.h3("API Configuration"),
            ui.input_text_area("api_keys_input", "Enter API Keys (one per line)", 
                              value="\n".join(api_keys()) if api_keys() else ""),
            ui.input_action_button("save_api_keys", "Save API Keys"),
            ui.p("Note: You need to configure your Google Gemini API keys here for the app to work."),
            ui.hr(),
            ui.h3("Danger Zone"),
            ui.input_action_button("reset_progress", "Reset All Session Progress", class_="btn-danger")
        )
    
    @reactive.effect
    def _():
        if input.save_api_keys():
            keys_text = input.api_keys_input()
            keys = [k.strip() for k in keys_text.split('\n') if k.strip()]
            api_keys.set(keys)
            ui.notification_show(f"Saved {len(keys)} API keys", duration=3)
    
    @reactive.effect
    def _():
        if input.reset_progress():
            current_state = get_state()
            try:
                supabase.table("user_progress").delete().eq("username", current_state["username"]).execute()
                # Reset state
                current_state["current_level"] = 1
                current_state["num_questions"] = 5
                current_state["missed_questions"] = []
                current_state["exam_history"] = []
                current_state["current_exam"] = ""
                current_state["current_key"] = []
                state.set(current_state)
                ui.notification_show("Progress reset successfully!", duration=5)
            except Exception as e:
                ui.notification_show(f"Failed to reset progress: {e}", duration=5, type="error")

app = App(app_ui, server)
