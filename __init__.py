import os
import csv
import json
import random
import urllib.request
import threading
from aqt import mw
from aqt.qt import *
from aqt.utils import showInfo, showWarning

# ----------------------------------------------------------------------
# DIRECT PATH TO YOUR DOCUMENTS FOLDER
# ----------------------------------------------------------------------
DATA_DIR = "/Users/AdrianL/Documents/Medtrainer"

# ----------------------------------------------------------------------
# HELPERS: DYNAMICALLY EXTRACT SUBJECTS & LECTURES FROM CSV DATA
# ----------------------------------------------------------------------
def get_target_csv_files(year_sem_code: str) -> list:
    if not os.path.exists(DATA_DIR):
        return []
    try:
        return [
            os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR)
            if f.endswith(".csv") and year_sem_code in f.lower()
        ]
    except Exception:
        return []

def extract_available_subjects(year_sem_code: str) -> list:
    """Scans CSV files for a block and returns unique subjects from subject columns ONLY."""
    subjects = set()
    subject_keywords = ["subject", "module", "category", "course name"]

    for fpath in get_target_csv_files(year_sem_code):
        try:
            with open(fpath, mode="r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    continue

                # Find specific subject column headers
                subject_cols = [
                    col for col in reader.fieldnames 
                    if any(kw in col.lower() for kw in subject_keywords)
                ]

                if not subject_cols:
                    continue

                for row in reader:
                    for col in subject_cols:
                        val = row.get(col, "")
                        if val and len(val.strip()) > 1:
                            subjects.add(val.strip().title())
        except Exception:
            continue

    return sorted(list(subjects))


def extract_available_lectures(year_sem_code: str, selected_subject: str) -> list:
    """Scans CSV files and returns lectures strictly belonging to the selected subject."""
    lectures = set()
    subject_keywords = ["subject", "module", "discipline", "course name"]
    lecture_keywords = ["lecture", "title", "session", "topic"]

    for fpath in get_target_csv_files(year_sem_code):
        try:
            with open(fpath, mode="r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    continue

                # Find exact subject and lecture columns
                subject_cols = [col for col in reader.fieldnames if any(kw in col.lower() for kw in subject_keywords)]
                lecture_cols = [col for col in reader.fieldnames if any(kw in col.lower() for kw in lecture_keywords)]

                for row in reader:
                    # If filtering by subject, check ONLY the subject column(s)
                    if selected_subject and selected_subject != "All Subjects":
                        row_subjects = [row.get(sc, "").strip().lower() for sc in subject_cols]
                        if not any(selected_subject.lower() in subj for subj in row_subjects):
                            continue  # Skip row if subject column doesn't match

                    # Extract lecture title from lecture column(s)
                    for col in (lecture_cols if lecture_cols else reader.fieldnames):
                        val = row.get(col, "")
                        if val and len(val.strip()) > 2:
                            lectures.add(val.strip().title())
        except Exception:
            continue

    return sorted(list(lectures))

# ----------------------------------------------------------------------
# 1. MULTI-CSV CONTEXT LOADER
# ----------------------------------------------------------------------
def get_medtrainer_context(year_sem_code: str, subject_filter: str, lecture_filter: str) -> str:
    matched_rows = []
    for fpath in get_target_csv_files(year_sem_code):
        try:
            with open(fpath, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row_text_all = " ".join([str(v) for v in row.values()]).lower()
                    
                    if subject_filter and subject_filter != "All Subjects" and subject_filter.lower() not in row_text_all:
                        continue
                    if lecture_filter and lecture_filter != "All Lectures" and lecture_filter.lower() not in row_text_all:
                        continue

                    row_str = " | ".join([f"{k}: {v}" for k, v in row.items() if v])
                    if row_str:
                        matched_rows.append(row_str)
        except Exception:
            continue

    random.shuffle(matched_rows)
    return "\n".join(matched_rows[:25])

# ----------------------------------------------------------------------
# 2. GEMINI API CALLER WITH STYLED HTML OUTPUT
# ----------------------------------------------------------------------
def generate_mcq_cards(
    api_key: str, 
    year_sem_code: str, 
    subject_filter: str, 
    lecture_filter: str, 
    num_questions: int, 
    level: int
) -> list:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent?key={api_key}"
    
    csv_context = get_medtrainer_context(year_sem_code, subject_filter, lecture_filter)
    
    # Split context into topics list
    topics_list = csv_context.split('\n') if csv_context else []

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

    combined_content = "\n\n".join([f"Source {i+1}: {t}" for i, t in enumerate(topics_list)])

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
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.7
        }
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    
    with urllib.request.urlopen(req) as response:
        res_data = json.loads(response.read().decode("utf-8"))
        exam_text = res_data["candidates"][0]["content"]["parts"][0]["text"]
        
        # Parse the exam text to extract questions and convert to card format
        questions = []
        lines = exam_text.strip().split('\n')
        current_question = None
        current_options = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line is a question number
            if line[0].isdigit() and line[1] == '.':
                # Save previous question if exists
                if current_question and current_options:
                    questions.append({
                        'question': current_question,
                        'options': current_options.copy()
                    })
                current_question = line[2:].strip()
                current_options = []
            # Check if line is an option
            elif line[0] in ['A', 'B', 'C', 'D'] and line[1] == '.':
                current_options.append(line[2:].strip())
        
        # Don't forget the last question
        if current_question and current_options:
            questions.append({
                'question': current_question,
                'options': current_options.copy()
            })
        
        # Convert to card format with HTML
        cards = []
        for i, q in enumerate(questions):
            correct_letter = dynamic_keys[i] if i < len(dynamic_keys) else 'A'
            correct_option = q['options'][ord(correct_letter) - ord('A')] if ord(correct_letter) - ord('A') < len(q['options']) else q['options'][0]
            
            front_html = f"""<div class="card-container">
         <div class="question-stem">{q['question']}</div>
         <div class="mcq-options">
           <div class="mcq-option"><b>A)</b> {q['options'][0] if len(q['options']) > 0 else ''}</div>
           <div class="mcq-option"><b>B)</b> {q['options'][1] if len(q['options']) > 1 else ''}</div>
           <div class="mcq-option"><b>C)</b> {q['options'][2] if len(q['options']) > 2 else ''}</div>
           <div class="mcq-option"><b>D)</b> {q['options'][3] if len(q['options']) > 3 else ''}</div>
         </div>
       </div>"""
            
            back_html = f"""<div class="answer-container">
         <div class="correct-badge">✓ Correct Answer: {correct_letter}) {correct_option}</div>
         <div class="explanation-box">
           <b>Clinical Context & Key Pearl:</b><br>Review the question and options to understand the correct answer.
         </div>
       </div>"""
            
            cards.append({
                'front': front_html,
                'back': back_html
            })
        
        return cards

# ----------------------------------------------------------------------
# 3. ANKI NOTE TYPE CREATOR WITH CSS STYLING
# ----------------------------------------------------------------------
CSS_STYLING = """
.card {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  font-size: 16px;
  text-align: left;
  color: #2c3e50;
  background-color: #f8f9fa;
  padding: 10px;
}

.card-container {
  max-width: 650px;
  margin: 0 auto;
  background: #ffffff;
  border-radius: 12px;
  padding: 24px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
  border: 1px solid #e9ecef;
}

.question-stem {
  font-size: 18px;
  font-weight: 600;
  line-height: 1.5;
  color: #1e293b;
  margin-bottom: 20px;
}

.mcq-options {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.mcq-option {
  background: #f1f5f9;
  border: 1px solid #cbd5e1;
  padding: 12px 16px;
  border-radius: 8px;
  font-size: 15px;
  transition: all 0.2s ease;
}

.mcq-option:hover {
  background: #e2e8f0;
  border-color: #94a3b8;
}

.answer-container {
  max-width: 650px;
  margin: 15px auto 0 auto;
}

.correct-badge {
  background-color: #d1fae5;
  color: #065f46;
  border: 1px solid #a7f3d0;
  font-weight: 700;
  padding: 12px 16px;
  border-radius: 8px;
  font-size: 16px;
  margin-bottom: 12px;
}

.explanation-box {
  background: #ffffff;
  border-left: 4px solid #3b82f6;
  padding: 16px;
  border-radius: 4px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.04);
  line-height: 1.6;
  color: #334155;
}
"""

def get_or_create_medtrainer_model():
    models = mw.col.models
    model = models.by_name("Medtrainer MCQ")
    
    if not model:
        model = models.new("Medtrainer MCQ")
        f1 = models.new_field("Front")
        f2 = models.new_field("Back")
        models.add_field(model, f1)
        models.add_field(model, f2)
        
        t = models.new_template("Card 1")
        t['qfmt'] = "{{Front}}"
        t['afmt'] = "{{Front}}<hr id=answer>{{Back}}"
        models.add_template(model, t)
        
        model['css'] = CSS_STYLING
        models.add(model)
        
    return model

def add_cards_to_anki(deck_name: str, cards_data: list) -> int:
    deck_id = mw.col.decks.id(deck_name)
    model = get_or_create_medtrainer_model()

    model['did'] = deck_id
    mw.col.models.save(model)

    added_count = 0
    for card in cards_data:
        note = mw.col.new_note(model)
        note["Front"] = card.get("front", "")
        note["Back"] = card.get("back", "")
        mw.col.add_note(note, deck_id)
        added_count += 1

    mw.reset()
    return added_count

import queue

# Global Queue & Worker Setup
task_queue = queue.Queue()
queue_worker_started = False

def process_queue():
    while True:
        task = task_queue.get()
        if task is None:
            break
            
        api_key, year_sem_code, selected_subject, selected_lecture, num_cards, level, deck_name = task
        try:
            cards = generate_mcq_cards(
                api_key, year_sem_code, selected_subject, selected_lecture, num_cards, level
            )
            mw.taskman.run_on_main(lambda d=deck_name, c=cards: add_cards_to_anki(d, c))
            mw.taskman.run_on_main(lambda d=deck_name: showInfo(f"Finished generating deck:\n{d}"))
        except Exception as e:
            err = str(e)
            mw.taskman.run_on_main(lambda e_msg=err: showWarning(f"Queue Task Failed:\n{e_msg}"))
        finally:
            task_queue.task_done()

def ensure_queue_worker():
    global queue_worker_started
    if not queue_worker_started:
        threading.Thread(target=process_queue, daemon=True).start()
        queue_worker_started = True
        
# ----------------------------------------------------------------------
# 4. PYQT DIALOG WINDOW
# ----------------------------------------------------------------------
class AIDeckDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Medtrainer AI Deck Generator")
        self.setMinimumWidth(440)
        
        # Read saved API Key config
        self.config = mw.addonManager.getConfig(__name__) or {}
        saved_api_key = self.config.get("api_key", "")

        layout = QVBoxLayout()

        # API Key Input
        layout.addWidget(QLabel("Gemini API Key:"))
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        if saved_api_key:
            self.api_key_input.setText(saved_api_key)
        layout.addWidget(self.api_key_input)

        # Filter 1: Curriculum Block Dropdown
        layout.addWidget(QLabel("Select Curriculum Block:"))
        self.year_sem_combo = QComboBox()
        self.year_sem_combo.addItems([
            "Year 2 - Semester 2 (y2s2)",
            "Year 2 - Semester 1 (y2s1)",
            "Year 1 - Semester 2 (y1s2)",
            "Year 1 - Semester 1 (y1s1)"
        ])
        self.year_sem_combo.setCurrentIndex(0)
        self.year_sem_combo.currentIndexChanged.connect(self.update_subject_dropdown)
        layout.addWidget(self.year_sem_combo)

        # Filter 2: Dynamic Subject Dropdown
        layout.addWidget(QLabel("Select Subject:"))
        self.subject_combo = QComboBox()
        self.subject_combo.currentIndexChanged.connect(self.update_lecture_dropdown)
        layout.addWidget(self.subject_combo)

        # Filter 3: Dynamic Lecture Dropdown
        layout.addWidget(QLabel("Select Lecture:"))
        self.lecture_combo = QComboBox()
        layout.addWidget(self.lecture_combo)

        # Populate initial dropdown items for default block (y2s2)
        self.update_subject_dropdown()

        # Filter 4: Difficulty Level Slider
        layout.addWidget(QLabel("Difficulty Level (1 - 50):"))
        self.level_slider = QSlider(Qt.Orientation.Horizontal)
        self.level_slider.setRange(1, 50)
        self.level_slider.setValue(1)
        layout.addWidget(self.level_slider)

        # Filter 5: Number of Cards Selector
        layout.addWidget(QLabel("Number of Cards (Max 50):"))
        self.num_cards_spin = QSpinBox()
        self.num_cards_spin.setRange(1, 50)
        self.num_cards_spin.setValue(10)
        layout.addWidget(self.num_cards_spin)

        # Action Button
        self.generate_btn = QPushButton("Generate MCQ Deck")
        self.generate_btn.clicked.connect(self.on_generate)
        layout.addWidget(self.generate_btn)

        self.setLayout(layout)

    def get_selected_block_code(self) -> str:
        year_sem_text = self.year_sem_combo.currentText()
        if "y2s2" in year_sem_text: return "y2s2"
        elif "y2s1" in year_sem_text: return "y2s1"
        elif "y1s2" in year_sem_text: return "y1s2"
        else: return "y1s1"

    def update_subject_dropdown(self):
        """Refreshes the Subject dropdown items when Curriculum Block changes."""
        self.subject_combo.blockSignals(True)
        self.subject_combo.clear()
        
        block_code = self.get_selected_block_code()
        available_subjects = extract_available_subjects(block_code)
        
        dropdown_items = ["All Subjects"] + available_subjects
        self.subject_combo.addItems(dropdown_items)
        self.subject_combo.blockSignals(False)

        # Trigger chain update to lecture dropdown
        self.update_lecture_dropdown()

    def update_lecture_dropdown(self):
        """Refreshes the Lecture dropdown items when Subject changes."""
        self.lecture_combo.clear()
        block_code = self.get_selected_block_code()
        selected_subject = self.subject_combo.currentText()

        available_lectures = extract_available_lectures(block_code, selected_subject)
        dropdown_items = ["All Lectures"] + available_lectures
        self.lecture_combo.addItems(dropdown_items)

    def on_generate(self):
        api_key = self.api_key_input.text().strip()

        if not api_key:
            showWarning("Please enter your Gemini API Key.", parent=self)
            return

        # Save API key to config if updated
        if api_key != self.config.get("api_key", ""):
            self.config["api_key"] = api_key
            mw.addonManager.writeConfig(__name__, self.config)

        year_sem_code = self.get_selected_block_code()
        selected_subject = self.subject_combo.currentText()
        selected_lecture = self.lecture_combo.currentText()
        level = self.level_slider.value()
        num_cards = self.num_cards_spin.value()

        # Build deck name
        title_details = [year_sem_code.upper()]
        if selected_subject and selected_subject != "All Subjects":
            title_details.append(selected_subject.title())
        if selected_lecture and selected_lecture != "All Lectures":
            title_details.append(selected_lecture.title())

        deck_title = " - ".join(title_details)
        deck_name = f"Medtrainer decks::{deck_title} (Level {level})"

        # Start background consumer thread if it isn't running yet
        ensure_queue_worker()

        # Push task parameters into the queue
        task_queue.put((
            api_key, year_sem_code, selected_subject, selected_lecture, num_cards, level, deck_name
        ))

        # Show feedback without locking the UI button
        q_size = task_queue.qsize()
        showInfo(f"Task queued! ({q_size} job(s) in queue)\n\nGenerating deck:\n{deck_title}", parent=self)


# ----------------------------------------------------------------------
# 5. REGISTER ANKI MENU HOOK
# ----------------------------------------------------------------------
def open_dialog():
    dialog = AIDeckDialog(mw)
    dialog.exec()

action = QAction("Medtrainer Deck Generator", mw)
qconnect(action.triggered, open_dialog)
mw.form.menuTools.addAction(action)
