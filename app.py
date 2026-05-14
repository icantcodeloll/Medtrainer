import streamlit as st
import pandas as pd
from google import genai
from google.genai import types 
import time
import re
import os
import atexit
from progress_manager import save_progress, load_progress, restore_progress
import concurrent.futures

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
# Note: Google Search works best with Flash/Pro (Lite may have tool limitations)
EXAM_MODEL = 'gemini-2.5-flash'
GRADER_MODEL = 'gemini-2.5-flash'

#Models that work: gemini-2.5-flash, gemini-2.5-flash-lite

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

def call_gemini_with_rotation(prompt, model_to_use, use_search=False, starting_key_idx=0):
    keys_tried = 0
    current_idx = starting_key_idx
    
    # Configure Google Search Tool if requested
    tools = []
    if use_search:
        tools = [types.Tool(google_search=types.GoogleSearch())]
    
    while keys_tried < len(API_KEYS):
        try:
            client = genai.Client(api_key=API_KEYS[current_idx])
            response = client.models.generate_content(
                model=model_to_use, 
                contents=prompt,
                config=types.GenerateContentConfig(tools=tools) if tools else None
            )
            return response.text
        except Exception as e:
            # Instead of giving up on unknown errors, log it to your terminal and try the next key
            print(f"Thread error with key {current_idx}: {e}") 
            keys_tried += 1
            if keys_tried >= len(API_KEYS):
                return None
            
            # Move to the next key and wait a second before retrying
            current_idx = (current_idx + 1) % len(API_KEYS)
            time.sleep(2) 

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
    
    # Using search during grading ensures explanations match current guidelines
    return call_gemini_with_rotation(prompt, GRADER_MODEL, use_search=True)

def validate_exam_format(exam_text, expected_questions):
    """Validate that the AI response follows the correct format"""
    if not exam_text or not exam_text.strip():
        return False, "Empty response"
    
    # Check if starts with question number
    if not re.match(r'^\s*1\.\s', exam_text.strip()):
        return False, "Does not start with '1. '"
    
    # Check for correct number of questions
    question_pattern = r'^\s*\d+\.\s'
    questions = re.findall(question_pattern, exam_text, re.MULTILINE)
    if len(questions) != expected_questions:
        return False, f"Expected {expected_questions} questions, found {len(questions)}"
    
    # Check for answer key format
    key_pattern = r'\[KEY:\s*[A-D,\s]+\]$'
    if not re.search(key_pattern, exam_text.strip()):
        return False, "Missing or malformed answer key"
    
    # Check each question has A, B, C, D options
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


def generate_single_question(topic_content, level, q_number, api_key_idx):
    """Generates a SINGLE question for a specific thread."""
    # Difficulty calibration (Keep your existing difficulty logic here)
    if level <= 5:
        difficulty_desc = "intuitive basics - straightforward medical concepts"
        complexity_guide = "focus on intuitive anatomy, obvious physiology"
    elif level <= 15:
        difficulty_desc = "logical progression - clinical applications"
        complexity_guide = "include common diseases with predictable presentations"
    elif level <= 25:
        difficulty_desc = "complex but predictable - applied knowledge"
        complexity_guide = "complex clinical cases with clear patterns"
    elif level <= 35:
        difficulty_desc = "challenging patterns - specialized knowledge"
        complexity_guide = "specialty-specific conditions with some counterintuitive elements"
    elif level <= 45:
        difficulty_desc = "counterintuitive expert - defies common assumptions"
        complexity_guide = "subspecialty expertise where textbook knowledge fails"
    else:
        difficulty_desc = "supreme counterintuition - advanced mastery"
        complexity_guide = "multi-system integration where standard rules don't apply"

    prompt = f"""
    You are a medical board examiner. 
    TASK: Generate EXACTLY 1 Multiple Choice Question based on the study material below.
    DIFFICULTY LEVEL: {level}/50.
    DIFFICULTY DESCRIPTION: {difficulty_desc}.
    COMPLEXITY GUIDANCE: {complexity_guide}.

    CRITICAL FORMATTING REQUIREMENTS:
    1. START IMMEDIATELY with '{q_number}. ' followed by the question text. NO preamble.
    2. ABSOLUTELY NO introductory text, explanations, or meta-commentary.
    3. The question MUST follow this EXACT format:
       "{q_number}. [Question text]
       A. [Option A]
       B. [Option B] 
       C. [Option C]
       D. [Option D]"
    4. The VERY LAST line must be: [KEY: X] (Replace X with the correct letter A, B, C, or D).
    5. All options must be plausible distractors. Use Google Search to verify accuracy.

    STUDY MATERIAL:
    {topic_content}
    """
    
    # Stagger thread starts slightly to prevent hitting the 429 rate limit in the exact same millisecond
    time.sleep((q_number % 10) * 1.0)
    return call_gemini_with_rotation(prompt, EXAM_MODEL, use_search=True, starting_key_idx=api_key_idx)


def get_blind_exam(topics_list, level, num_questions):
    """Manages parallel generation of all questions simultaneously."""
    questions_data = {}
    
    def fetch_question(i, topic):
        # Evenly distribute the 10 API keys across the threads to avoid rate limits
        key_idx = i % len(API_KEYS)
        res = generate_single_question(topic, level, i + 1, key_idx)
        return i, res

    # Run all requests simultaneously using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_questions) as executor:
        futures = [executor.submit(fetch_question, i, topic) for i, topic in enumerate(topics_list)]
        for future in concurrent.futures.as_completed(futures):
            i, res = future.result()
            questions_data[i] = res

    # ---------------------------------------------------------
    # STITCHING IT BACK TOGETHER 
    # ---------------------------------------------------------
    # Now we recombine them into the exact format your UI expects
    # so that the rest of your Streamlit app requires 0 changes!
    final_exam_text = ""
    combined_keys = []
    
    for i in range(num_questions):
        raw_q = questions_data.get(i, "")
        if raw_q and "[KEY:" in raw_q:
            # Separate the question from the individual key
            q_text, k_part = raw_q.split("[KEY:")
            final_exam_text += q_text.strip() + "\n\n"
            
            # Extract just the letter (A, B, C, or D)
            k_match = re.search(r'[A-D]', k_part)
            combined_keys.append(k_match.group() if k_match else "A")
        else:
            # Fallback in case a specific thread fails
            final_exam_text += f"{i+1}. [Error generating question. Please select A to continue]\nA. \nB. \nC. \nD.\n\n"
            combined_keys.append("A")
            
    # Append the combined key format your app looks for at the very end
    final_exam_text += f"\n[KEY: {', '.join(combined_keys)}]"
    
    return final_exam_text

# Replace get_mnemonic with this:
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

# --- ADD THIS: Focus Mode Dropdown ---
df_sidebar = pd.read_csv(CSV_FILE)
categories = df_sidebar['category'].fillna("Uncategorized").astype(str).unique().tolist()
all_categories = ["All Topics"] + sorted(categories)
focus_mode = st.sidebar.selectbox("Focus Mode:", all_categories)

# --- ADD THIS: Exam Filter Dropdown ---
if 'exam' in df_sidebar.columns:
    exams = df_sidebar['exam'].fillna("Uncategorized").astype(str).unique().tolist()
    all_exams = ["All Exams"] + sorted(exams)
    exam_filter = st.sidebar.selectbox("Exam Filter:", all_exams)
else:
    exam_filter = "All Exams"

if st.sidebar.button("Generate New Exam"):
    st.session_state.exam_submitted = False  # Add this line
    st.session_state.last_score = 0
    st.session_state.user_selections = {}  # Clear previous selections
    n = st.session_state.num_questions
    try:
        df_main = pd.read_csv(CSV_FILE)
        df_notes = pd.read_csv(NOTES_FILE)
        df = pd.merge(df_main, df_notes, on=JOIN_COLUMN, how='left')
                
        # --- ADD THIS: Filter by category ---
        if focus_mode != "All Topics":
            df = df[df['category'] == focus_mode]
            if df.empty:
                st.error(f"No questions found for {focus_mode}. Check your CSV.")
                st.stop()

        # --- ADD THIS: Filter by exam ---
        if exam_filter != "All Exams" and 'exam' in df.columns:
            df = df[df['exam'] == exam_filter]
            if df.empty:
                st.error(f"No questions found for {exam_filter}. Check your CSV.")
                st.stop()

        # --- TOGGLE LOGIC ---
        if 'include' in df.columns:
            df = df[df['include'].astype(str).str.lower().str.strip() == 'y']
            
            if df.empty:
                st.error("No active objectives found. Mark some as 'y' in your CSV.")
                st.stop()
    

        # --- SMART SAMPLING (SRS) ---
        if 'mastery_score' in df.columns:
            # Convert mastery_score to integers for comparison
            df['mastery_score'] = pd.to_numeric(df['mastery_score'], errors='coerce').fillna(1).astype(int)
            # Prioritize items with mastery <= 3
            weak_pool = df[df['mastery_score'] <= 3]
            if len(weak_pool) >= n:
                st.session_state.samples_df = weak_pool.sample(n)
            else:
                st.session_state.samples_df = df.sample(n)
            st.sidebar.info("Smart Sampling: Prioritizing weak areas.")
        else:
            st.session_state.samples_df = df.sample(min(n, len(df)))
        
        # Use the session state version for the rest of the generation
        samples_df = st.session_state.samples_df
        # ----------------------------

        if 'category' in samples_df.columns:
            st.session_state.current_categories = samples_df['category'].fillna('General').tolist()
        else:
            st.session_state.current_categories = ['General'] * n

        samples = (samples_df['explanation'] + "\n[Notes: " + samples_df['content'].fillna('') + "]").tolist()
        
        with st.spinner(f"Generating {n} questions at Level {st.session_state.current_level}..."):
            raw_response = get_blind_exam(samples, st.session_state.current_level, n)
            if raw_response and "[KEY:" in raw_response:
                # Use a split that keeps the questions separate from the key
                text, key_part = raw_response.split("[KEY:")
                
                # CLEANING: Remove the key section from the visible text 
                # so it doesn't show up in the last radio button question
                st.session_state.current_exam = text.strip() 
                
                st.session_state.current_key = re.findall(r'[A-D]', key_part)
                st.rerun()
            else:
                st.error("Failed to generate a valid exam. The AI response was incomplete.")
    except Exception as e:
        st.error(f"File Error: Ensure {CSV_FILE} and {NOTES_FILE} are in the folder. ({e})")

# Move Active Level metric here
st.sidebar.metric("Active Level", f"{st.session_state.current_level}/50")

# Settings button for sliders
if st.sidebar.button("Settings", use_container_width=True):
    if 'show_settings' not in st.session_state:
        st.session_state.show_settings = False
    st.session_state.show_settings = not st.session_state.show_settings

# Show sliders only when settings is expanded
if st.session_state.get('show_settings', False):
    st.session_state.current_level = st.sidebar.slider("Starting Level", 1, 50, st.session_state.current_level)
    st.session_state.num_questions = st.sidebar.slider("Number of Questions", 3, 20, st.session_state.num_questions)
    
    st.sidebar.markdown("---")
    if st.sidebar.button("Reset Progress", help="Clear all saved progress and reset to defaults"):
        if os.path.exists("user_progress.json"):
            os.remove("user_progress.json")
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
    st.sidebar.subheader("Knowledge Bank")
    # Use the df_sidebar we loaded earlier
    if 'mastery_score' in df_sidebar.columns:
        # Convert mastery_score to integers for comparison
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

            # Save this formatted version to session state
            st.session_state.last_user_input = user_input
            st.session_state.last_correct_key = correct_key_formatted
            
            if len(user_answers) != len(correct_key):
                st.error(f"Mismatch: The exam has {len(correct_key)} questions, but you entered {len(user_answers)} answers. Please fix your input.")
                st.stop() # Stops the code before it hits the loop and crashes
            score = 0
            
            # --- ADDED: Load the CSV to update scores ---
            df_main = pd.read_csv(CSV_FILE)
            
            raw_chunks = re.split(r'\n(?=\d+\.)', st.session_state.current_exam.strip())
            individual_questions = [q for q in raw_chunks if re.match(r'^\d+\.', q.strip())]     

            for i, (u_ans, correct) in enumerate(zip(user_answers, correct_key)):
                # --- ADDED: Link the question back to the original CSV index ---
                # We use st.session_state.samples_df to avoid the NameError
                original_idx = st.session_state.samples_df.index[i]
                
                if u_ans == correct:
                    score += 1
                    # --- ADDED: Increase Mastery Score (Max 5) ---
                    if 'mastery_score' in df_main.columns:
                        # Convert mastery_score to integers for comparison
                        df_main['mastery_score'] = pd.to_numeric(df_main['mastery_score'], errors='coerce').fillna(1).astype(int)
                        df_main.at[original_idx, 'mastery_score'] = min(5, df_main.at[original_idx, 'mastery_score']) + mastery_change
                else:
                    # --- ADDED: Decrease Mastery Score (Min 1) ---
                    if 'mastery_score' in df_main.columns:
                        # Convert mastery_score to integers for comparison
                        df_main['mastery_score'] = pd.to_numeric(df_main['mastery_score'], errors='coerce').fillna(1).astype(int)
                        df_main.at[original_idx, 'mastery_score'] = max(1, df_main.at[original_idx, 'mastery_score']) - mastery_change
                    
                    if i < len(individual_questions):
                        st.session_state.missed_questions.append({
                            "question": individual_questions[i].strip(),
                            "correct": correct,
                            "yours": u_ans,
                            "category": st.session_state.current_categories[i] if i < len(st.session_state.current_categories) else "General"
                        })
            
            # --- ADDED: Save the changes back to your file ---
            df_main.to_csv(CSV_FILE, index=False)
            
            # Save state so the feedback stays visible after submission
            st.session_state.exam_submitted = True
            st.session_state.last_score = score
            st.session_state.last_user_input = user_input

            # Update Level based on performance
            num_actual_questions = len(user_answers)
            percentage_correct = (score / num_actual_questions) * 100
            questions_wrong = num_actual_questions - score
            
            # Level up if: only 1 question wrong OR 80%+ correct (whichever is lower threshold)
            if questions_wrong <= 1 or percentage_correct >= 80:
                st.session_state.current_level = min(50, st.session_state.current_level + 1)
                st.success(f"Level Up! Now at Level {st.session_state.current_level}")
            # Level down if: less than half correct
            elif percentage_correct < 50:
                st.session_state.current_level = max(1, st.session_state.current_level - 1)
                st.warning(f"Level Down. Now at Level {st.session_state.current_level}")
            else:
                st.info(f"Score: {score}/{st.session_state.num_questions} ({percentage_correct:.0f}%) - Level maintained")

# 2. THE FEEDBACK (Outside the form)
    if st.session_state.get('exam_submitted'):
        st.subheader(f"Results: {st.session_state.last_score}/{st.session_state.num_questions}")
        
        with st.spinner("Instructor is searching for the latest feedback..."):
            # This keeps your original answer comparison and AI explanation
            feedback = get_ai_grading(
                st.session_state.current_exam, 
                st.session_state.last_user_input, 
                st.session_state.last_correct_key,
                st.session_state.last_score
            )
            st.markdown(feedback)

        st.write("---")
        

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


