from tkinter.constants import N
import streamlit as st
import pandas as pd
from google import genai
from google.genai import types 
import time
import re

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="Trainer", page_icon="🩺", layout="wide")

API_KEYS = [st.secrets["GENAI_KEY_1"], st.secrets["GENAI_KEY_2"], st.secrets["GENAI_KEY_3"], st.secrets["GENAI_KEY_4"], st.secrets["GENAI_KEY_5"], st.secrets["GENAI_KEY_6"], st.secrets["GENAI_KEY_7"], st.secrets["GENAI_KEY_8"], st.secrets["GENAI_KEY_9"], st.secrets["GENAI_KEY_10"]]
CSV_FILE = "learning_objectives_informative_reports.csv"
NOTES_FILE = "lecture_notes.csv"
JOIN_COLUMN = "lecture_id"

# Models
# Note: Google Search works best with Flash/Pro (Lite may have tool limitations)
EXAM_MODEL = 'gemini-2.5-flash'
GRADER_MODEL = 'gemini-2.5-flash-lite'

#Models that work: gemini-2.5-flash, gemini-2.5-flash-lite

mastery_mode = "off"

if mastery_mode == "on":
    mastery_change = 1
else:
    mastery_change = 0


# Initialize Session States
if 'current_level' not in st.session_state:
    st.session_state.current_level = 10
if 'num_questions' not in st.session_state:
    st.session_state.num_questions = 10 
if 'missed_questions' not in st.session_state:
    st.session_state.missed_questions = []
if 'current_exam' not in st.session_state:
    st.session_state.current_exam = None
if 'current_key' not in st.session_state:
    st.session_state.current_key = []
if 'key_index' not in st.session_state:
    st.session_state.key_index = 0
if 'current_categories' not in st.session_state:
    st.session_state.current_categories = []
if 'samples_df' not in st.session_state:
    st.session_state.samples_df = None

def get_client():
    return genai.Client(api_key=API_KEYS[st.session_state.key_index])

def call_gemini_with_rotation(prompt, model_to_use, use_search=False):
    keys_tried = 0
    
    # Configure Google Search Tool if requested
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
                    st.error("🛑 All API keys exhausted.")
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
def get_ai_grading(exam_text, user_answers, correct_key):
    prompt = f"""
    Here is the input:
    EXAM QUESTIONS: {exam_text}
    CORRECT KEY: {correct_key}
    STUDENT ANSWERS: {user_answers}

    You are a medical instructor. Grade the student's performance.
    ### GRADING PROTOCOL:
    1. Compare the student's answer for each index (1 through 10) against the correct key.
    2. USE GOOGLE SEARCH to check if the correct key is correct, 
    3. If they match, it is correct. If they differ, it is incorrect.
    4. Provide a brief explanation for any incorrect answers.
    """
    
    # Using search during grading ensures explanations match current guidelines
    return call_gemini_with_rotation(prompt, GRADER_MODEL, use_search=True)

def get_blind_exam(topics_list, level, num_questions):
    combined_content = "\n\n".join([f"Source {i+1}: {t}" for i, t in enumerate(topics_list)])
    prompt = f"""
    You are a medical board examiner. 
    TASK: Generate EXACTLY {num_questions} Multiple Choice Questions (1 per snippet provided below).
    DIFFICULTY LEVEL: {level}/50.
    

    INSTRUCTIONS:
    1. START IMMEDIATELY with '1. [Question Text]'. 
    2. Do NOT include any introductory text, pleasantries, or descriptions of the task.
    3. Every question MUST start with its number and a period (e.g., '1.', '2.').
    4. Use the STUDY MATERIAL provided below as the base.
    5. USE GOOGLE SEARCH to supplement these questions with external medical knowledge, 
       latest clinical guidelines (e.g., NICE, Sepsis-3, GOLD), and realistic clinical presentations.
    6. Ensure questions are complex enough for Level {level}.
    7. **STRICT ANSWER DISTRIBUTION**: You must ensure a mathematically balanced distribution of correct answers across the 10 questions.
       - Each letter (A, B, C, D) should be the correct answer approximately 2-3 times. 
       - Do NOT favor any specific letter. For every question, ensure there is exactly a 25% chance of the answer being A.
    8. Provide 4 options (A, B, C, D).    
    9. Provide the key in this format at the VERY end: [KEY: A, B, C, D, A, B, C, D, A, B]
    

    STUDY MATERIAL:
    {combined_content}
    """
    return call_gemini_with_rotation(prompt, EXAM_MODEL, use_search=True)

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
st.title("🩺 Trainer")
st.sidebar.header("Stats & Controls")
st.session_state.current_level = st.sidebar.slider("Starting Level", 1, 50, st.session_state.current_level)
st.session_state.num_questions = st.sidebar.slider("Number of Questions", 1, 20, st.session_state.num_questions)

st.sidebar.metric("Active Level", f"{st.session_state.current_level}/50")

# --- ADD THIS: Focus Mode Dropdown ---
df_sidebar = pd.read_csv(CSV_FILE)
categories = df_sidebar['category'].fillna("Uncategorized").astype(str).unique().tolist()
all_categories = ["All Topics"] + sorted(categories)
focus_mode = st.sidebar.selectbox("🎯 Focus Mode:", all_categories)

if st.sidebar.button("🔄 Generate New Exam"):
    st.session_state.exam_submitted = False  # Add this line
    st.session_state.last_score = 0
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


        # --- TOGGLE LOGIC ---
        if 'include' in df.columns:
            df = df[df['include'].astype(str).str.lower().str.strip() == 'y']
            
            if df.empty:
                st.error("❌ No active objectives found. Mark some as 'y' in your CSV.")
                st.stop()
    

        # --- SMART SAMPLING (SRS) ---
        if 'mastery_score' in df.columns:
            # Prioritize items with mastery <= 3
            weak_pool = df[df['mastery_score'] <= 3]
            if len(weak_pool) >= n:
                st.session_state.samples_df = weak_pool.sample(n)
            else:
                st.session_state.samples_df = df.sample(n)
            st.sidebar.info("🎯 Smart Sampling: Prioritizing weak areas.")
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
            formatted_q = re.sub(r"\s+([A-D]\.)", r"\n\1", q_text)
            # Use markdown so the bolding and newlines from the AI render correctly
            st.markdown(q_text)
            
            # This makes the radio buttons cleaner
            st.session_state.user_selections[i] = st.radio(
                label=f"Select answer for Question {i+1}", # Provide a real label
                options=["A", "B", "C", "D"],
                key=f"q_radio_{i}",
                horizontal=True,
                index=None,
                label_visibility="collapsed" # This hides the label visually
            )
            st.write("---")
        
        submitted = st.form_submit_button("Submit for AI Grading")
    if submitted:
            # Convert dictionary to a sorted list of answers
            user_answers = [st.session_state.user_selections[i] for i in range(len(raw_questions))]
            user_input = "\n".join([f"Q{i+1}: {ans if ans else 'No Answer'}" for i, ans in enumerate(user_answers)])
            correct_key = "\n".join([f"Q{i+1}: {ans}" for i, ans in enumerate(st.session_state.current_key)])

            # Save this formatted version to session state
            st.session_state.last_user_input = user_input
            st.session_state.last_correct_key = correct_key
            correct_key = st.session_state.current_key
            
            if len(user_answers) != len(correct_key):
                st.error(f"⚠️ Mismatch: The exam has {len(correct_key)} questions, but you entered {len(user_answers)} answers. Please fix your input.")
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
                        df_main.at[original_idx, 'mastery_score'] = min(5, df_main.at[original_idx, 'mastery_score']) + mastery_change
                else:
                    # --- ADDED: Decrease Mastery Score (Min 1) ---
                    if 'mastery_score' in df_main.columns:
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

            # Update Level
            if score >= 9: st.session_state.current_level = min(50, st.session_state.current_level + 1)
            elif score <= 4: st.session_state.current_level = max(1, st.session_state.current_level - 1)

# 2. THE FEEDBACK (Outside the form)
    if st.session_state.get('exam_submitted'):
        st.subheader(f"Results: {st.session_state.last_score}/10")
        
        with st.spinner("Instructor is searching for the latest feedback..."):
            # This keeps your original answer comparison and AI explanation
            feedback = get_ai_grading(
                st.session_state.current_exam, 
                st.session_state.last_user_input, 
                st.session_state.last_correct_key
            )
            st.markdown(feedback)

        st.write("---")
        
        # 3. INTERACTIVE TOOLS (Upgraded with Selection Dropdown)
        if st.session_state.missed_questions:
            st.subheader("🛠️ Remediation Tools")
            
            # Dropdown to select WHICH missed question to analyze
            # We show the first 60 characters of the question text in the list
            # Clean the display list for the dropdown
            mistake_options = [m['question'][:80].replace('\n', ' ') + "..." for m in st.session_state.missed_questions]
            selected_index = st.selectbox(
                "Pick a question to focus on:", 
                range(len(mistake_options)), 
                format_func=lambda x: f"Question {x + 1}: {mistake_options[x]}"
            )
            
            
            # The data for the specific question chosen from the dropdown
            target_mistake = st.session_state.missed_questions[selected_index]

            col1, col2 = st.columns(2)
            
            with col1:
                # Change the subheader and button text here:
                st.subheader("📖 Deep Dive")
                if st.button("Explain More (Deep Dive)"):
                    with st.spinner("Generating detailed clinical explanation..."):
                        # Make sure to call the NEW function name here:
                        explanation = get_deep_explanation(target_mistake['question'])
                        st.info(explanation)

            with col2:
                st.subheader("🔍 Logic Check")
                # unique key is needed for text_input inside interactive areas
                user_logic = st.text_input("Why did you pick that answer?", key=f"logic_input_{selected_index}")
                if st.button("Analyze My Reasoning"):
                    if user_logic:
                        with st.spinner("Analyzing clinical logic..."):
                            # We send the specific question context and the user's logic
                            logic_prompt = f"""
                            Context: {target_mistake['question']}
                            Student Logic: {user_logic}
                            Correct Answer: {target_mistake['correct']}
                            Identify the clinical logic error and why their reasoning was incorrect.
                            """
                            analysis = call_gemini_with_rotation(logic_prompt, GRADER_MODEL)
                            st.info(analysis)
                    else:
                        st.warning("Please enter your reasoning first.")
        else:
            st.success("Perfect score! No remediation tools needed.")

# Missed Questions Bank in Sidebar
if st.session_state.missed_questions:
    # 1. THE HEATMAP (Visual)
    st.write("---")
    st.header("📊 Weakness Heatmap")
    m_df = pd.DataFrame(st.session_state.missed_questions)
    if 'category' in m_df.columns:
        st.bar_chart(m_df['category'].value_counts(), color="#ff4b4b")

    # 2. YOUR ORIGINAL SIDEBAR EXPORT (Preserved)
    st.sidebar.markdown("---")
    st.sidebar.subheader(f"Missed Questions ({len(st.session_state.missed_questions)})")
    if st.sidebar.button("💾 Export Mistakes to .txt"):
        with open("missed_questions.txt", "a") as f:
            f.write(f"\n\n=== WEB SESSION: {time.strftime('%Y-%m-%d %H:%M')} ===\n")
            for item in st.session_state.missed_questions:
                # Upgraded text format to include category in the file
                cat = item.get('category', 'General')
                f.write(f"\n[{cat}] {item['question']}\n[CORRECT: {item['correct']} | YOURS: {item['yours']}]\n")
        st.sidebar.success("Saved to missed_questions.txt")


# --- ADD THIS: Mastery Progress Tracker ---
st.sidebar.markdown("---")
st.sidebar.subheader("🎓 Knowledge Bank")
# Use the df_sidebar we loaded earlier
if 'mastery_score' in df_sidebar.columns:
    total_objs = len(df_sidebar)
    mastered = len(df_sidebar[df_sidebar['mastery_score'] == 5])
    progress_val = mastered / total_objs if total_objs > 0 else 0
    
    st.sidebar.write(f"Mastered: {mastered} / {total_objs}")
    st.sidebar.progress(progress_val)
    st.sidebar.caption(f"{progress_val*100:.1f}% of curriculum at Mastery Level 5")
