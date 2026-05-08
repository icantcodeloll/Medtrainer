import json
import os
from datetime import datetime

PROGRESS_FILE = "user_progress.json"

def save_progress(session_state):
    """Save all relevant progress data to JSON file"""
    progress_data = {
        "timestamp": datetime.now().isoformat(),
        "current_level": session_state.get("current_level", 10),
        "num_questions": session_state.get("num_questions", 10),
        "last_score": session_state.get("last_score", 0),
        "missed_questions": session_state.get("missed_questions", []),
        "last_user_input": session_state.get("last_user_input", ""),
        "last_correct_key": session_state.get("last_correct_key", ""),
        "exam_submitted": session_state.get("exam_submitted", False),
        "current_categories": session_state.get("current_categories", []),
        "samples_df": session_state.get("samples_df", None)
    }
    
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress_data, f, indent=2)

def load_progress():
    """Load progress data from JSON file"""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {}
    return {}

def restore_progress(session_state, progress_data):
    """Restore progress data to session state"""
    if not progress_data:
        return
    
    session_state.current_level = progress_data.get("current_level", 10)
    session_state.num_questions = progress_data.get("num_questions", 10)
    session_state.last_score = progress_data.get("last_score", 0)
    session_state.missed_questions = progress_data.get("missed_questions", [])
    session_state.last_user_input = progress_data.get("last_user_input", "")
    session_state.last_correct_key = progress_data.get("last_correct_key", "")
    session_state.exam_submitted = progress_data.get("exam_submitted", False)
    session_state.current_categories = progress_data.get("current_categories", [])
    # samples_df is a DataFrame, handle separately
    if "samples_df" in progress_data and progress_data["samples_df"]:
        try:
            import pandas as pd
            session_state.samples_df = pd.DataFrame(progress_data["samples_df"])
        except:
            pass
