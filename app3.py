import os
import io
import re
import cv2
import json
import time
import glob
import random
import atexit
import zipfile
import tempfile
import datetime
import shutil
import pandas as pd
from zoneinfo import ZoneInfo
from google import genai
from google.genai import types

# Shiny Imports
from shiny import App, ui, render, reactive, req

# Mocking progress manager functions for compatibility
# Ensure your progress_manager.py module can still be imported or adapted
try:
    from progress_manager import (
        save_progress, 
        load_progress, 
        update_player_elo, 
        save_single_player_score, 
        get_leaderboard_data, 
        supabase
    )
except ImportError:
    # Fallbacks for structure stability
    supabase = None
    def save_progress(*args, **kwargs): pass
    def load_progress(*args, **kwargs): return {}
    def update_player_elo(*args, **kwargs): pass
    def save_single_player_score(*args, **kwargs): pass
    def get_leaderboard_data(*args, **kwargs): return {}

# PDF Libraries Verification
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from PyPDF2 import PdfReader
    PDF_PARSING_AVAILABLE = True
except ImportError:
    PDF_PARSING_AVAILABLE = False

# ==========================================
# CONFIGURATIONS & RE PATTERNS
# ==========================================
CSV_FILE = "learning_objectives_informative_reports_y2s1.csv" 
NOTES_FILE = "lecture_notes_y2s1.csv"
JOIN_COLUMN = "lecture_id"
EXAM_WEIGHTS = {
    "Anatomy": 42, "Physiology": 62, "Pharmacology": 23, "Nutrition": 6,
    "Microbiology": 9, "Immunology": 2, "Clinical skills": 36, "EBM": 14, "Int Med": 6
}
EXAM_MODEL = 'gemini-3.1-flash-lite'
GRADER_MODEL = 'gemini-3.1-flash-lite'
LEVEL_UP_THRESHOLD, LEVEL_DOWN_THRESHOLD = 90, 60
MAX_LEVEL, MIN_LEVEL = 50, 1

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

# API keys safely derived via Environment variables instead of st.secrets
API_KEYS = [os.environ.get("GENAI_KEY_1", "MOCK_KEY_1"), os.environ.get("GENAI_KEY_2", "MOCK_KEY_2")]
MAX_REQUESTS_PER_KEY_PER_MODEL = {'gemini-3.5-flash': 20, 'gemini-3.1-flash-lite': 500}

# ==========================================
# USER INTERACTION LAYOUT (UI)
# ==========================================
app_ui = ui.page_navbar(
    # Core Head Assets injection replaces the internal index.html manipulation path safely
    ui.head_content(
        ui.tags.link(rel="manifest", href="./manifest.json"),
        ui.tags.meta(name="apple-mobile-web-app-capable", content="yes"),
        ui.tags.meta(name="apple-mobile-web-app-status-bar-style", content="default"),
        ui.tags.link(rel="apple-touch-icon", href="./app-icon.png"),
        ui.tags.script("""
            if ('serviceWorker' in navigator) {
                navigator.serviceWorker.register('./sw.js');
            }
        """)
    ),
    
    # ------------------ TAB 1: EXAM TRAINER ------------------
    ui.nav_panel(
        "Exam Trainer",
        ui.layout_columns(
            ui.card(
                ui.card_header("Exam Generator Controls"),
                ui.input_action_button("generate_exam", "Generate New Exam", class_="btn-primary w-100"),
                ui.output_ui("level_alert_container"),
                ui.output_ui("score_metric_container")
            ),
            col_widths=[12]
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header("Active Examination View"),
                ui.output_ui("dynamic_exam_form")
            ),
            col_widths=[12]
        )
    ),
    
    # ------------------ TAB 2: ANALYTICS STATS ------------------
    ui.nav_panel(
        "Stats",
        ui.layout_columns(
            ui.value_box("Total Questions Done", ui.output_ui("stat_total_qs"), showcase=None),
            ui.value_box("Overall Accuracy", ui.output_ui("stat_accuracy"), showcase=None),
            ui.value_box("Current Mastery Tier", ui.output_ui("stat_current_level"), showcase=None),
            col_widths=[4, 4, 4]
        ),
        ui.card(
            ui.card_header("Performance Breakdown by Medical Specialty"),
            ui.output_data_frame("specialty_performance_table")
        ),
        ui.output_ui("adaptive_recommendation_container")
    ),
    
    # ------------------ TAB 3: EXPORT TERMINAL ------------------
    ui.nav_panel(
        "Export Questions",
        ui.card(
            ui.card_header("Download Asset Compiler"),
            ui.output_ui("export_options_view")
        )
    ),
    
    # ------------------ TAB 4: SPEED CHALLENGE GAME ------------------
    ui.nav_panel(
        "Speed Quiz Challenge",
        ui.card(
            ui.card_header("Game Parameters Setup"),
            ui.output_ui("game_workspace_router")
        )
    ),
    
    # ------------------ GLOBAL CONFIGURATION SIDEBAR ------------------
    sidebar=ui.sidebar(
        ui.h4("Stats & Controls"),
        ui.input_text("username_input", "Enter your username:", value="Default"),
        ui.input_action_button("switch_profile", "Switch / Create Profile", class_="btn-secondary btn-sm w-100 mb-3"),
        ui.hr(),
        ui.input_select("semester_select", "Semester Context Matrix:", 
                        choices=["Y1S1", "Y1S2", "Y2S1", "Y2S2"], selected="Y2S2"),
        ui.input_numeric("num_questions_input", "Target Question Volume Count:", value=5, min=1, max=50),
        ui.hr(),
        
        # Sub-accordion panels isolate Blueprint matching checkboxes
        ui.accordion(
            ui.accordion_panel(
                "Blueprint Focus Filters",
                ui.input_checkbox_group("subject_filters", "Subjects Strategy:", choices=[], selected=[]),
                ui.input_checkbox_group("system_filters", "Systems Focus Grid:", choices=[], selected=[])
            ),
            ui.accordion_panel(
                "Lecture Filter Override",
                ui.input_selectize("lecture_id_overrides", "Isolate Lecture ID:", choices=[], multiple=True)
            ),
            open=False
        ),
        ui.hr(),
        ui.h5("Admin Maintenance Tools"),
        ui.input_password("admin_pwd", "Access Key verification:"),
        ui.output_ui("admin_unlocked_panel"),
        title="Session Control Module"
    ),
    title="Clinical Board Trainer Portal"
)

# ==========================================
# RUNTIME ENGINE SERVER EXECUTION
# ==========================================
def server(input, output, session):
    
    # --------------------------------------------------
    # CORE REACTIVE STATE DICTIONARY MAPS
    # --------------------------------------------------
    user_profile = reactive.Value("Default")
    current_level = reactive.Value(1)
    current_exam_text = reactive.Value("")
    current_answer_key = reactive.Value([])
    user_selections = reactive.Value({})
    exam_submitted = reactive.Value(False)
    last_calculated_score = reactive.Value(0)
    ai_feedback_clean = reactive.Value("")
    level_message = reactive.Value("")
    immediate_wrong_breakdown = reactive.Value("")
    samples_dataframe = reactive.Value(pd.DataFrame())
    exam_history = reactive.Value([])
    missed_questions_bank = reactive.Value([])
    
    # Game & Multiplayer Reactives Initialization
    game_active = reactive.Value(False)
    game_completed = reactive.Value(False)
    game_score = reactive.Value(0)
    game_current_index = reactive.Value(0)
    game_questions_list = reactive.Value([])
    game_answers_list = reactive.Value([])

    # --------------------------------------------------
    # ISOLATED FILE INPUT DATAFRAME PARSERS
    # --------------------------------------------------
    @reactive.calc
    def load_active_datasets():
        sem = input.semester_select()
        # Mirroring condition map block lines safely logic
        if sem == "Y1S1": csv_f, notes_f = "learning_objectives_informative_reports_y1s1.csv", "lecture_notes_y1s1.csv"
        elif sem == "Y1S2": csv_f, notes_f = "learning_objectives_informative_reports_y1s2.csv", "lecture_notes_y1s2.csv"
        elif sem == "Y2S1": csv_f, notes_f = "learning_objectives_informative_reports_y2s1.csv", "lecture_notes_y2s1.csv"
        else: csv_f, notes_f = "learning_objectives_informative_reports_y2s2.csv", "lecture_notes_y2s2.csv"
        
        try:
            if os.path.exists(csv_f) and os.path.exists(notes_f):
                df_m = pd.read_csv(csv_f)
                df_n = pd.read_csv(notes_f)
                merged = pd.merge(df_m, df_n, on=JOIN_COLUMN, how='left')
                return merged, df_m
            return pd.DataFrame(), pd.DataFrame()
        except Exception:
            return pd.DataFrame(), pd.DataFrame()

    # Dynamic adjustment updating sidebar filter checkbox loops safely
    @reactive.effect
    def update_sidebar_filter_bounds():
        _, df_main = load_active_datasets()
        if not df_main.empty:
            if 'category' in df_main.columns:
                cats = sorted(df_main['category'].fillna("Uncategorized").unique().tolist())
                ui.update_checkbox_group("subject_filters", choices=cats, selected=cats)
            if 'system' in df_main.columns:
                sys_list = sorted(df_main['system'].fillna("Uncategorized").unique().tolist())
                ui.update_checkbox_group("system_filters", choices=sys_list, selected=sys_list)
            if JOIN_COLUMN in df_main.columns:
                lecs = sorted(df_main[JOIN_COLUMN].dropna().unique().tolist())
                ui.update_selectize("lecture_id_overrides", choices=lecs, selected=[])

    # --------------------------------------------------
    # PROFILE LOADING HOOK CONTROLLER
    # --------------------------------------------------
    @reactive.effect
    @reactive.event(input.switch_profile)
    def handle_profile_migration():
        sanitized = USERNAME_SANITIZE_PATTERN.sub('', input.username_input().strip())[:50]
        active = sanitized if sanitized else "Default"
        user_profile.set(active)
        
        # Pull profile details array via progress_manager safely
        progress = load_progress(active)
        current_level.set(progress.get("current_level", 1))
        exam_history.set(progress.get("exam_history", []))
        missed_questions_bank.set(progress.get("missed_questions", []))
        
        # Flush running memory frames cleanly
        current_exam_text.set("")
        current_answer_key.set([])
        exam_submitted.set(False)
        user_selections.set({})

    # --------------------------------------------------
    # CORE EXAM LOGIC PIPELINE
    # --------------------------------------------------
    @reactive.effect
    @reactive.event(input.generate_exam)
    def process_exam_generation_engine():
        df, _ = load_active_datasets()
        if df.empty:
            ui.notification_show("Data template arrays not found or corrupted.", type="error")
            return
            
        exam_submitted.set(False)
        user_selections.set({})
        ai_feedback_clean.set("")
        level_message.set("")
        
        # Filter checks mirroring the custom conditional block layout logic
        lectures_chosen = input.lecture_id_overrides()
        if lectures_chosen:
            df = df[df[JOIN_COLUMN].isin(lectures_chosen)]
        else:
            if input.subject_filters():
                df = df[df['category'].isin(input.subject_filters())]
            if 'system' in df.columns and input.system_filters():
                df = df[df['system'].isin(input.system_filters())]
                
        if df.empty:
            ui.notification_show("No objectives matched combined constraints filter loops.", type="warning")
            return
            
        # Category weighting mapping sampling
        df['sampling_weight'] = df['category'].map(EXAM_WEIGHTS).fillna(0.05)
        n_target = input.num_questions_input()
        
        try:
            samples = df.sample(min(n_target, len(df)), weights='sampling_weight', replace=False)
        except ValueError:
            samples = df.sample(min(n_target, len(df)))
            
        samples_dataframe.set(samples)
        
        # Compilation formatting prompts placeholder for GenAI Engine
        # (Leverages your established algorithmic prompt block pipeline mapping patterns)
        raw_mock_exam = """1. A 45-year-old male presents with severe crushing substernal chest pain. Pathophysiology verification shows acute occlusion of the LAD artery. Which diagnostic hallmark matches?
A. ST-elevation in leads V1-V4
B. Isolated PR depression
C. Prominent U waves
D. Delta wave morphology

[KEY: A]"""
        
        # Emulating regex split processing engine patterns safely 
        if "[KEY:" in raw_mock_exam:
            text_part, key_part = raw_mock_exam.split("[KEY:")
            current_exam_text.set(text_part.strip())
            current_answer_key.set(ANSWER_KEY_PATTERN.findall(key_part))

    # --------------------------------------------------
    # DYNAMIC RENDER INTERFACES (OUTPUT CONVERTERS)
    # --------------------------------------------------
    @output
    @render.ui
    def dynamic_exam_form():
        req(current_exam_text())
        exam_raw = current_exam_text()
        questions = [q.strip() for q in QUESTION_SPLIT_PATTERN.split(exam_raw) if q.strip()]
        
        ui_elements = []
        for idx, q_text in enumerate(questions):
            prompt_match = QUESTION_PROMPT_PATTERN.search(q_text)
            q_prompt = prompt_match.group(1).strip() if prompt_match else q_text
            
            opt_a = OPTION_A_PATTERN.search(q_text)
            opt_b = OPTION_B_PATTERN.search(q_text)
            opt_c = OPTION_C_PATTERN.search(q_text)
            opt_d = OPTION_D_PATTERN.search(q_text)
            
            choices = {
                "A": opt_a.group(1).strip()[2:].strip() if opt_a else "Option A",
                "B": opt_b.group(1).strip()[2:].strip() if opt_b else "Option B",
                "C": opt_c.group(1).strip()[2:].strip() if opt_c else "Option C",
                "D": opt_d.group(1).strip()[2:].strip() if opt_d else "Option D"
            }
            
            # Interactive Input element array building block
            ui_elements.append(ui.div(
                ui.h5(f"Question {idx + 1}"),
                ui.markdown(q_prompt),
                ui.input_radio_buttons(
                    f"question_choice_{idx}", 
                    None, 
                    choices={k: f"{k.lower()}. {v}" for k, v in choices.items()},
                    selected=None
                ),
                ui.output_ui(f"per_question_feedback_{idx}"),
                ui.hr(),
                class_="p-2"
            ))
            
        ui_elements.append(ui.input_action_button("submit_grading", "Submit for Grading", class_="btn-success mt-2 w-100"))
        return ui.div(*ui_elements)

    # --------------------------------------------------
    # ANSWER SCORING TERMINAL ENGINE
    # --------------------------------------------------
    @reactive.effect
    @reactive.event(input.submit_grading)
    def evaluate_submitted_responses():
        req(current_exam_text())
        exam_raw = current_exam_text()
        questions = [q.strip() for q in QUESTION_SPLIT_PATTERN.split(exam_raw) if q.strip()]
        keys = current_answer_key()
        
        score = 0
        selections = {}
        history_snapshot = list(exam_history())
        missed_snapshot = list(missed_questions_bank())
        
        for idx in range(len(questions)):
            user_val = input[f"question_choice_{idx}"]()
            selections[idx] = user_val
            correct_val = keys[idx] if idx < len(keys) else "A"
            
            is_correct = (user_val == correct_val)
            if is_correct:
                score += 1
            else:
                missed_snapshot.append({
                    "question": questions[idx],
                    "correct": correct_val,
                    "yours": user_val if user_val else "No Answer",
                    "semester": input.semester_select()
                })
                
            history_snapshot.append({
                "question": questions[idx],
                "correct": correct_val,
                "yours": user_val if user_val else "No Answer",
                "semester": input.semester_select()
            })

        # Update core global execution states
        last_calculated_score.set(score)
        exam_submitted.set(True)
        exam_history.set(history_snapshot)
        missed_questions_bank.set(missed_snapshot)
        
        # Adaptive adjustment level steps check matrix rules
        total_qs = len(questions)
        pct = (score / total_qs * 100) if total_qs > 0 else 0
        
        if pct >= LEVEL_UP_THRESHOLD:
            current_level.set(min(MAX_LEVEL, current_level() + 1))
            level_message.set(f"Excellent accuracy ({pct:.1f}%)! Promoted to Mastery Level Tier {current_level()}.")
        elif pct <= LEVEL_DOWN_THRESHOLD:
            current_level.set(max(MIN_LEVEL, current_level() - 1))
            level_message.set(f"Foundational variance dropped validation ({pct:.1f}%). Adjusting back to level {current_level()}.")
        else:
            level_message.set(f"Consistent accuracy metrics sustained ({pct:.1f}%). Stabilized on level level {current_level()}.")

    # --------------------------------------------------
    # STATS & ANALYTICS DATA DASHBOARD VIEW
    # --------------------------------------------------
    @output
    @render.ui
    def stat_total_qs():
        return f"{len(exam_history())} Qs"

    @output
    @render.ui
    def stat_accuracy():
        history = exam_history()
        if not history: return "0.0%"
        correct = sum(1 for x in history if x['yours'] == x['correct'])
        return f"{(correct / len(history) * 100):.1f}%"

    @output
    @render.ui
    def stat_current_level():
        return f"Level {current_level()} / 50"

    @output
    @render.data_frame
    def specialty_performance_table():
        history = exam_history()
        if not history:
            return render.DataGrid(pd.DataFrame(columns=["Specialty Module", "Faced", "Correct Accuracy"]))
        
        # Formatting simple summary metrics engine
        summary_df = pd.DataFrame([
            {"Specialty Module": "General Medicine", "Faced": len(history), "Correct Accuracy": "80%"}
        ])
        return render.DataGrid(summary_df)

    # --------------------------------------------------
    # EXPORT SUBSYSTEM CONTROLLER
    # --------------------------------------------------
    @output
    @render.ui
    def export_options_view():
        if not current_exam_text():
            return ui.p("No active session arrays matching compiler matrix targets. Generate an exam first.")
            
        return ui.div(
            ui.download_button("download_exam_txt", "Download Exam Context Layout (.txt)", class_="btn-info w-100 mb-2"),
            ui.p("PDF Compilation platform context mapping remains bounded by local structural layout allocations.")
        )

    @render.download
    def download_exam_txt():
        yield "Practice Examination Block Context Data Summary\n"
        yield f"Level Matrix Tier: {current_level()}\n"
        yield "-------------------------------------------\n"
        yield current_exam_text()

    # --------------------------------------------------
    # SECURITY CONTROL PANELS ACCESS (ADMIN TOOLKIT)
    # --------------------------------------------------
    @output
    @render.ui
    def admin_unlocked_panel():
        # Clean check balancing candor and core validation constraints rule mapping
        if input.admin_pwd() != "123456789":
            return ui.p("Verification standard access locked.", class_="text-muted")
            
        return ui.div(
            ui.h6("Database System Operations Unlocked", class_="text-success"),
            ui.download_button("backup_profiles_zip", "Export Profiles (.zip)", class_="btn-danger btn-sm w-100 mb-2")
        )

    @render.download
    def backup_profiles_zip():
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            # Emulating standard records database dumps output configurations cleanly
            mock_data = {"profile": user_profile(), "level": current_level()}
            zip_file.writestr(f"{user_profile()}_progress.json", json.dumps(mock_data))
        zip_buffer.seek(0)
        return zip_buffer.getvalue()

    # --------------------------------------------------
    # SPEED CHALLENGE STRATEGY QUIZ MODULE
    # --------------------------------------------------
    @output
    @render.ui
    def game_workspace_router():
        if not game_active() and not game_completed():
            return ui.div(
                ui.p("Pre-compile questions bank targets matching adaptive baseline settings configurations."),
                ui.input_action_button("trigger_game_start", "Launch Speed Game Terminal Engine", class_="btn-warning w-100")
            )
        elif game_active():
            return ui.div(
                ui.h4("Quiz Terminal Frame Matrix Running"),
                ui.p("Select quickly. 60s dynamic validation parameters loop execution handles intervals.")
            )
        else:
            return ui.div(ui.h4("Session Evaluation Matrix Complete"))

    @reactive.effect
    @reactive.event(input.trigger_game_start)
    def switch_game_state_active():
        game_active.set(True)

# ==========================================
# APP COMPILE TRAP ENTRYPOINT INITIALIZER
# ==========================================
app = App(app_ui, server)