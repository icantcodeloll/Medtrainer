import json
import os
from datetime import datetime
import pandas as pd

def get_file_path(username):
    """Generate a unique file path for each user."""
    # This ensures the filename is safe and doesn't contain weird characters
    clean_name = "".join([c for c in username if c.isalpha() or c.isdigit() or c=='_']).rstrip()
    if not clean_name:
        clean_name = "Default"
    return f"{clean_name}_progress.json"

def save_progress(session_state, username="Default"):
    """Save all relevant progress data to JSON file for a specific user"""
    file_path = get_file_path(username)
    try:
        progress_data = {
            "timestamp": datetime.now().isoformat(),
            "current_level": session_state.get("current_level", 10),
            "exam_model": session_state.get("exam_model", 'gemini-3.1-flash-lite'),
            "num_questions": session_state.get("num_questions", 10),
            "semester": session_state.get("semester", "y2s1"),
            "last_score": session_state.get("last_score", 0),
            "missed_questions": session_state.get("missed_questions", []),
            "exam_history": session_state.get("exam_history", []),
            "current_exam": session_state.get("current_exam", ""),
            "current_key": session_state.get("current_key", []),
            "key_index": session_state.get("key_index", 0),
            "last_user_input": session_state.get("last_user_input", ""),
            "last_correct_key": session_state.get("last_correct_key", ""),
            "exam_submitted": session_state.get("exam_submitted", False),
            "current_categories": session_state.get("current_categories", []),
            # Handle DataFrame serialization
            "samples_df": session_state.get("samples_df").to_dict() if session_state.get("samples_df") is not None else None
        }
        
        with open(file_path, 'w') as f:
            json.dump(progress_data, f, indent=2)
        
        print(f"Progress saved to {file_path}")
        return True
    except Exception as e:
        print(f"Error saving progress: {e}")
        return False

def load_progress(username="Default"):
    """Load progress data from JSON file for a specific user"""
    file_path = get_file_path(username)
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
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