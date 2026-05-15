import streamlit as st
import pandas as pd
from google import genai
from google.genai import types 
import time
import re
import os
import atexit
from progress_manager import save_progress, load_progress, restore_progress
from typing import List, Tuple, Dict, Any, Optional
import json

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="Trainer", page_icon="🩺", layout="wide")

# Safari compatibility fixes
st.markdown("""
<style>
.stButton > button {
    width: 100%;
}
.stSelectbox > div > div > select {
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

API_KEYS = [st.secrets["GENAI_KEY_1"], st.secrets["GENAI_KEY_2"], st.secrets["GENAI_KEY_3"], st.secrets["GENAI_KEY_4"], st.secrets["GENAI_KEY_5"], st.secrets["GENAI_KEY_6"], st.secrets["GENAI_KEY_7"], st.secrets["GENAI_KEY_8"], st.secrets["GENAI_KEY_9"], st.secrets["GENAI_KEY_10"], st.secrets["GENAI_KEY_11"], st.secrets["GENAI_KEY_12"], st.secrets["GENAI_KEY_13"], st.secrets["GENAI_KEY_14"], st.secrets["GENAI_KEY_15"], st.secrets["GENAI_KEY_16"], st.secrets["GENAI_KEY_17"], st.secrets["GENAI_KEY_18"], st.secrets["GENAI_KEY_19"], st.secrets["GENAI_KEY_20"]]
CSV_FILE = "learning_objectives_informative_reports.csv"
NOTES_FILE = "lecture_notes.csv"
JOIN_COLUMN = "lecture_id"

# Models
EXAM_MODEL = 'gemini-2.5-flash-lite'
GRADER_MODEL = 'gemini-2.5-flash'

mastery_mode = "off"

if mastery_mode == "on":
    mastery_change = 1
else:
    mastery_change = 0

# Load saved progress on startup
loaded_progress = load_progress()

# Initialize Session States (with progress restoration)
if 'current_level' not in st.session_state:
    st.session_state.current_level = loaded_progress.get("current_level", 1)
if 'num_questions' not in st.session_state:
    st.session_state.num_questions = loaded_progress.get("num_questions", 5) 
if 'missed_questions' not in st.session_state:
    st.session_state.missed_questions = loaded_progress.get("missed_questions", [])
if 'current_exam' not in st.session_state:
    st.session_state.current_exam = loaded_progress.get("current_exam", None)
if 'current_key' not in st.session_state:
    st.session_state.current_key = loaded_progress.get("current_key", [])
if 'key_index' not in st.session_state:
    st.session_state.key_index = loaded_progress.get("key_index", 0)
if 'current_categories' not in st.session_state:
    st.session_state.current_categories = loaded_progress.get("current_categories", [])
if 'samples_df' not in st.session_state:
    st.session_state.samples_df = loaded_progress.get("samples_df", None)

# Register auto-save on app shutdown
atexit.register(save_progress, st.session_state)

if st.sidebar.button("Save Progress", help="Manually save your current progress"):
    if save_progress(st.session_state):
        st.sidebar.success("Progress saved successfully!")
    else:
        st.sidebar.error("Failed to save progress")

def get_client():
    return genai.Client(api_key=API_KEYS[st.session_state.key_index])

def call_gemini_with_rotation(prompt, model_to_use, use_search=False, timeout_per_question=3):
    keys_tried = 0
    tools = []
    if use_search:
        tools = [types.Tool(google_search=types.GoogleSearch())]
    
    while keys_tried < len(API_KEYS):
        try:
            client = get_client()
            response = client.models.generate_content(
                model=model_to_use, 
                contents=prompt,
                config=types.GenerateContentConfig(tools=tools) if tools else None
            )
            return response.text
        except Exception as e:
            if "429" in str(e):
                keys_tried += 1
                if keys_tried >= len(API_KEYS):
                    st.error("All API keys exhausted.")
                    return None
                st.session_state.key_index = (st.session_state.key_index + 1) % len(API_KEYS)
                time.sleep(1)
            elif "503" in str(e):
                time.sleep(5)
            else:
                st.error(f"Error: {e}")
                return None

# ==========================================
# 2. CORE LOGIC FUNCTIONS
# ==========================================
def get_ai_grading(exam_text, user_answers, correct_key, score):
    prompt = f"""
    Here is the input:
    EXAM QUESTIONS: {exam_text}
    CORRECT KEY: {correct_key}
    STUDENT ANSWERS: {user_answers}
    SCORE: {score}

    You are a medical instructor. Grade the student's performance.
    ### GRADING PROTOCOL:
    1. Compare the student's answer for each index (1 through 10) against the correct key.
    2. USE GOOGLE SEARCH to check if the correct key is correct, 
    3. If they match, it is correct. If they differ, it is incorrect.
    4. Provide a brief explanation for any incorrect answers.
    """
    return call_gemini_with_rotation(prompt, GRADER_MODEL, use_search=True)

def validate_questions_double_check(
    exam_text: str, 
    model: Any = EXAM_MODEL, 
    use_search: bool = True
) -> Tuple[bool, str, List[str]]:
    
    prompt = f"""
    You are a medical education specialist. CRITICALLY analyze these exam questions.
    
    CONTEXT: Exam generated at difficulty level {st.session_state.current_level}
    
    EXAM TO REVIEW:
    {exam_text}
    
    Your analysis MUST check:
    
    1. MEDICAL ACCURACY & SAFETY:
       - All facts must be medically accurate, current (2024)
       - Treatments align with current guidelines
       
    2. QUALITY CHECKS:
       - Questions: One clear best answer
       - Distractors: Plausible but clearly inferior
       - Difficulty matches level {st.session_state.current_level}
       
    3. EXAM INTEGRITY:
       - All questions answerable from provided content
       - Key answers are actually correct
       
    If VALID, respond: "VALID"
    If INVALID, format each issue as:
    - [SEVERITY: LOW/MED/HIGH] Description of issue
    
    Your analysis:
    """
    try:
        response = call_gemini_with_rotation(prompt, model=GRADER_MODEL, use_search=use_search)
        if "VALID" in response.upper():
            return True, "All questions passed quality check ✅", []
        else:
            issues = [line.strip('-* ') for line in response.split('\n') if line.strip()]
            return False, "Issues found during validation", issues
    except Exception as e:
        return False, f"Validation error: {str(e)}", ["Validation failed"]

def extract_questions_from_exam(exam_text: str) -> List[str]:
    """Extract individual questions from raw exam text for verification"""
    if "[KEY:" in exam_text:
        exam_text = exam_text.split("[KEY:")[0]
        
    clean_text = re.sub(r"^(Here are|Based on|Sure|I have generated).*?\n", "", exam_text.strip(), flags=re.IGNORECASE)
    raw_questions = re.split(r'\n(?=\d+\.\s)', clean_text)
    return [q.strip() for q in raw_questions if q.strip()]

def validate_medical_facts_via_search(questions: List[str]) -> Tuple[bool, str]:
    verification_prompt = f"""
    You are a medical fact-checker. Verify these medical claims:
    {chr(10).join(f'{i+1}. {q[:150]}...' for i,q in enumerate(questions))}
    
    For EACH statement above: 
    1. Is it MEDICALLY ACCURATE by current standards?
    2. Is it RECENT (within 5 years)?
    3. Does it align with standard medical textbooks?
    
    If ANY claim is false or questionable, FLAG it.
    RESPOND with "ALL VALID" or list INVALID claims.
    """
    
    response = call_gemini_with_rotation(
        verification_prompt, 
        model=EXAM_MODEL, 
        use_search=True,
        timeout_per_question=5
    )
    
    return "ALL VALID" in response.upper(), response

def calculate_exam_quality(exam_text: str) -> float:
    key_letters = [k for k in exam_text if k in "ABCD"]
    counts = {letter: key_letters.count(letter) for letter in "ABCD"}
    if not key_letters:
        return 0.0
    balance_score = 1 - (max(counts.values()) - min(counts.values())) / len(key_letters)
    
    if "?" not in exam_text or "A." not in exam_text:
        return 0.3
    
    return (balance_score * 0.3 + 0.7)

def generate_and_validate_exam(samples, level, n):
    attempts = 0
    max_attempts = 3
    best_exam = None
    
    while attempts < max_attempts:
        # 1. Generate exam
        raw_exam = get_blind_exam(samples, level, n)
        
        # 2. First-pass technical validation
        if not raw_exam or "[KEY:" not in raw_exam or len(raw_exam.strip()) < 100:  
            attempts += 1
            continue
            
        best_exam = raw_exam  # Keep structurally valid exam as fallback
        
        # 3. Content validation
        is_valid, msg, issues = validate_questions_double_check(raw_exam)
        
        # 4. Quality metrics
        quality_score = calculate_exam_quality(raw_exam)
        
        # 5. Accept or retry logic
        if is_valid and quality_score > 0.7:  
            # Double-check factual accuracy via search
            questions = extract_questions_from_exam(raw_exam)
            all_valid, response_details = validate_medical_facts_via_search(questions)
            
            if all_valid:
                return raw_exam  # Passed all checks perfectly
            else:
                attempts += 1
                continue
        else:
            attempts += 1
            continue
            
    # If perfect validation failed, return the best formatted one we have
    return best_exam if best_exam else raw_exam

def validate_exam_format(exam_text, expected_questions):
    if not exam_text or not exam_text.strip():
        return False, "Empty response"
    
    if not re.match(r'^\s*1\.\s', exam_text.strip()):
        return False, "Does not start with '1. '"
    
    question_pattern = r'^\s*\d+\.\s'
    questions = re.findall(question_pattern, exam_text, re.MULTILINE)
    if len(questions) != expected_questions:
        return False, f"Expected {expected_questions} questions, found {len(questions)}"
    
    key_pattern = r'$$KEY:\s*[A-D,\s]+$$$'
    if not re.search(key_pattern, exam_text.strip()):
        return False, "Missing or malformed answer key"
    
    lines = exam_text.split('\n')
    current_question = 0
    option_count = 0
    
    for line in lines:
        line = line.strip()
        if re.match(r'^\d+\.\s', line):
            current_question += 1
            option_count = 0
        elif re.match(r'^[A-D]\.\s', line):
            option_count += 1
    
    if current_question != expected_questions:
        return False, f"Question count mismatch: {current_question} vs {expected_questions}"
    
    return True, "Valid format"

def get_blind_exam(topics_list, level, num_questions):
    combined_content = "\n\n".join([f"Source {i+1}: {t}" for i, t in enumerate(topics_list)])
    
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
        complexity_guide = "multi-system integration where standard rules don't apply, latest research breakthroughs that overturn conventional wisdom, complex clinical reasoning requiring recognition of exceptions, niche subspecialty knowledge where intuitive answers are wrong, molecular-level pathophysiology that defies simple explanations, emerging treatment protocols with paradoxical mechanisms, rare disease patterns that mimic opposite conditions, advanced diagnostic challenges where the obvious answer is incorrect"
    
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
    6. The VERY LAST line must be: [KEY: A, B, C, D, A, B, C, D, A, B]

    STUDY MATERIAL:
    {combined_content}

    REMEMBER: Start with '1. ' immediately. No introduction. End with [KEY: format].
    """
    
    exam_text = call_gemini_with_rotation(prompt, EXAM_MODEL, use_search=True, timeout_per_question=3)
    return exam_text

def get_deep_explanation(question_text):
    prompt = f"""
    Provide a comprehensive medical deep-dive for this question:
    "{question_text}"
    
    Include:
    1. **Pathophysiology**: Explain the underlying mechanism.
    2. **Gold Standard**: Why the correct answer is the preferred clinical choice.
    3. **Clinical Pearl**: A high-yield tip for board exams (e.g., 'Always look for X in a patient with Y').
    """
    return call_gemini_with_rotation(prompt, EXAM_MODEL, use_search=True)

# ==========================================
# 3. WEB INTERFACE
# ==========================================
st.title("Trainer")
st.sidebar.header("Stats & Controls")

df_sidebar = pd.read_csv(CSV_FILE)
categories = df_sidebar['category'].fillna("Uncategorized").astype(str).unique().tolist()
all_categories = ["All Topics"] + sorted(categories)
focus_mode = st.sidebar.selectbox("Focus Mode:", all_categories)

if 'exam' in df_sidebar.columns:
    exams = df_sidebar['exam'].fillna("Uncategorized").astype(str).unique().tolist()
    all_exams = ["All Exams"] + sorted(exams)
    exam_filter = st.sidebar.selectbox("Exam Filter:", all_exams)
else:
    exam_filter = "All Exams"

if st.sidebar.button("Generate New Exam"):
    st.session_state.exam_submitted = False  
    st.session_state.last_score = 0
    st.session_state.user_selections = {}  
    n = st.session_state.num_questions
    try:
        df_main = pd.read_csv(CSV_FILE)
        df_notes = pd.read_csv(NOTES_FILE)
        df = pd.merge(df_main, df_notes, on=JOIN_COLUMN, how='left')
                
        if focus_mode != "All Topics":
            df = df[df['category'] == focus_mode]
            if df.empty:
                st.error(f"No questions found for {focus_mode}. Check your CSV.")
                st.stop()

        if exam_filter != "All Exams" and 'exam' in df.columns:
            df = df[df['exam'] == exam_filter]
            if df.empty:
                st.error(f"No questions found for {exam_filter}. Check your CSV.")
                st.stop()

        if 'include' in df.columns:
            df = df[df['include'].astype(str).str.lower().str.strip() == 'y']
            
            if df.empty:
                st.error("No active objectives found. Mark some as 'y' in your CSV.")
                st.stop()
    
        if 'mastery_score' in df.columns:
            df['mastery_score'] = pd.to_numeric(df['mastery_score'], errors='coerce').fillna(1).astype(int)
            weak_pool = df[df['mastery_score'] <= 3]
            if len(weak_pool) >= n:
                st.session_state.samples_df = weak_pool.sample(n)
            else:
                st.session_state.samples_df = df.sample(n)
            st.sidebar.info("Smart Sampling: Prioritizing weak areas.")
        else:
            st.session_state.samples_df = df.sample(min(n, len(df)))
        
        samples_df = st.session_state.samples_df

        if 'category' in samples_df.columns:
            st.session_state.current_categories = samples_df['category'].fillna('General').tolist()
        else:
            st.session_state.current_categories = ['General'] * n

        samples = (samples_df['explanation'] + "\n[Notes: " + samples_df['content'].fillna('') + "]").tolist()
        
        # 🟢 UPDATED: This now passes through the fact-verification pipeline!
        with st.spinner(f"Generating & Fact-Checking {n} questions at Level {st.session_state.current_level}..."):
            raw_response = generate_and_validate_exam(samples, st.session_state.current_level, n)
            
            if raw_response and "[KEY:" in raw_response:
                text, key_part = raw_response.split("[KEY:")
                st.session_state.current_exam = text.strip() 
                st.session_state.current_key = re.findall(r'[A-D]', key_part)
                st.rerun()
            else:
                st.error("Failed to generate a valid exam. The AI response was incomplete.")
    except Exception as e:
        st.error(f"File Error: Ensure {CSV_FILE} and {NOTES_FILE} are in the folder. ({e})")

st.sidebar.metric("Active Level", f"{st.session_state.current_level}/50")

if st.sidebar.button("Settings", use_container_width=True):
    if 'show_settings' not in st.session_state:
        st.session_state.show_settings = False
    st.session_state.show_settings = not st.session_state.show_settings

if st.session_state.get('show_settings', False):
    st.session_state.current_level = st.sidebar.slider("Starting Level", 1, 50, st.session_state.current_level)
    st.session_state.num_questions = st.sidebar.slider("Number of Questions", 3, 20, st.session_state.num_questions)
    
    st.sidebar.markdown("---")
    if st.sidebar.button("Reset Progress", help="Clear all saved progress and reset to defaults"):
        if os.path.exists("user_progress.json"):
            os.remove("user_progress.json")
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
    st.sidebar.subheader("Knowledge Bank")
    if 'mastery_score' in df_sidebar.columns:
        df_sidebar['mastery_score'] = pd.to_numeric(df_sidebar['mastery_score'], errors='coerce').fillna(1).astype(int)
        total_objs = len(df_sidebar)
        mastered = len(df_sidebar[df_sidebar['mastery_score'] == 5])
        progress_val = mastered / total_objs if total_objs > 0 else 0
        
        st.sidebar.write(f"Mastered: {mastered} / {total_objs}")
        st.sidebar.progress(progress_val)
        st.sidebar.caption(f"{progress_val*100:.1f}% of curriculum at Mastery Level 5")

# Display the Exam
if st.session_state.current_exam:
    st.info("Select the best answer for each clinical scenario below.")
    
    clean_text = st.session_state.current_exam.strip()
    clean_text = re.sub(r"^(Here are|Based on|Sure|I have generated).*?\n", "", clean_text, flags=re.IGNORECASE)

    raw_questions = re.split(r'\n(?=\d+\.\s)', clean_text)
    individual_questions = [q.strip() for q in raw_questions if q.strip()]

    if 'user_selections' not in st.session_state:
        st.session_state.user_selections = {}

    with st.form("grading_form"):
        for i, q_text in enumerate(individual_questions):
            st.subheader(f"Question {i+1}")
            
            formatted_q = re.sub(r"(\d+\.\s[^A-D]*?)(?=A\.)", r"\1<br>", q_text)
            formatted_q = re.sub(r"([A-D]\.\s[^A-D]*?)(?=[A-D]\.|$)", r"\1<br>", formatted_q)
            formatted_q = formatted_q.replace("\n", "<br>")
            formatted_q = re.sub(r"(<br>){2,}", "<br>", formatted_q)
            
            st.markdown(formatted_q, unsafe_allow_html=True)
            
            exam_content = st.session_state.get('current_exam', '')
            exam_content_hash = hash(exam_content) if exam_content else 0
            st.session_state.user_selections[i] = st.radio(
                label=f"Select answer for Question {i+1}", 
                options=["A", "B", "C", "D"],
                key=f"q_radio_{i}_{abs(exam_content_hash) % 1000}",
                horizontal=True,
                index=None,
                label_visibility="collapsed" 
            )
            st.write("---")
        
        submitted = st.form_submit_button("Submit for Grading")
        
    if submitted:
            num_actual_questions = len(raw_questions)
            user_answers = [st.session_state.user_selections[i] for i in range(num_actual_questions)]
            user_input = "\n".join([f"Q{i+1}: {ans if ans else 'No Answer'}" for i, ans in enumerate(user_answers)])
            
            correct_key = st.session_state.current_key[:num_actual_questions]
            correct_key_formatted = "\n".join([f"Q{i+1}: {ans}" for i, ans in enumerate(correct_key)])

            st.session_state.last_user_input = user_input
            st.session_state.last_correct_key = correct_key_formatted
            
            if len(user_answers) != len(correct_key):
                st.error(f"Mismatch: The exam has {len(correct_key)} questions, but you entered {len(user_answers)} answers. Please fix your input.")
                st.stop() 
            score = 0
            
            df_main = pd.read_csv(CSV_FILE)
            
            raw_chunks = re.split(r'\n(?=\d+\.)', st.session_state.current_exam.strip())
            individual_questions = [q for q in raw_chunks if re.match(r'^\d+\.', q.strip())]     

            for i, (u_ans, correct) in enumerate(zip(user_answers, correct_key)):
                original_idx = st.session_state.samples_df.index[i]
                
                if u_ans == correct:
                    score += 1
                    if 'mastery_score' in df_main.columns:
                        df_main['mastery_score'] = pd.to_numeric(df_main['mastery_score'], errors='coerce').fillna(1).astype(int)
                        df_main.at[original_idx, 'mastery_score'] = min(5, df_main.at[original_idx, 'mastery_score']) + mastery_change
                else:
                    if 'mastery_score' in df_main.columns:
                        df_main['mastery_score'] = pd.to_numeric(df_main['mastery_score'], errors='coerce').fillna(1).astype(int)
                        df_main.at[original_idx, 'mastery_score'] = max(1, df_main.at[original_idx, 'mastery_score']) - mastery_change
                    
                    if i < len(individual_questions):
                        st.session_state.missed_questions.append({
                            "question": individual_questions[i].strip(),
                            "correct": correct,
                            "yours": u_ans,
                            "category": st.session_state.current_categories[i] if i < len(st.session_state.current_categories) else "General"
                        })
            
            df_main.to_csv(CSV_FILE, index=False)
            
            st.session_state.exam_submitted = True
            st.session_state.last_score = score
            st.session_state.last_user_input = user_input

            num_actual_questions = len(user_answers)
            percentage_correct = (score / num_actual_questions) * 100
            questions_wrong = num_actual_questions - score
            
            if questions_wrong <= 1 or percentage_correct >= 80:
                st.session_state.current_level = min(50, st.session_state.current_level + 1)
                st.success(f"Level Up! Now at Level {st.session_state.current_level}")
            elif percentage_correct < 50:
                st.session_state.current_level = max(1, st.session_state.current_level - 1)
                st.warning(f"Level Down. Now at Level {st.session_state.current_level}")
            else:
                st.info(f"Score: {score}/{st.session_state.num_questions} ({percentage_correct:.0f}%) - Level maintained")

    if st.session_state.get('exam_submitted'):
        st.subheader(f"Results: {st.session_state.last_score}/{st.session_state.num_questions}")
        
        with st.spinner("Instructor is searching for the latest feedback..."):
            feedback = get_ai_grading(
                st.session_state.current_exam, 
                st.session_state.last_user_input, 
                st.session_state.last_correct_key,
                st.session_state.last_score
            )
            st.markdown(feedback)

        st.write("---")

if st.session_state.missed_questions:
    st.write("---")
    st.header("Weakness Heatmap")
    m_df = pd.DataFrame(st.session_state.missed_questions)
    if 'category' in m_df.columns:
        st.bar_chart(m_df['category'].value_counts(), color="#ff4b4b")

    st.sidebar.markdown("---")
    st.sidebar.subheader(f"Missed Questions ({len(st.session_state.missed_questions)})")
    if st.sidebar.button("Export Mistakes to .txt"):
        with open("missed_questions.txt", "a") as f:
            f.write(f"\n\n=== WEB SESSION: {time.strftime('%Y-%m-%d %H:%M')} ===\n")
            for item in st.session_state.missed_questions:
                cat = item.get('category', 'General')
                f.write(f"\n[{cat}] {item['question']}\n[CORRECT: {item['correct']} | YOURS: {item['yours']}]\n")
        st.sidebar.success("Saved to missed_questions.txt")