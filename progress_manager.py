import streamlit as st
from supabase import create_client, Client
from datetime import datetime
import pandas as pd

# -------------------------------------------------------------
# SUPABASE CLIENT INITIALIZATION
# -------------------------------------------------------------
@st.cache_resource
def init_supabase() -> Client:
    """Initialize cached connection to Supabase cloud database."""
    url: str = st.secrets["SUPABASE_URL"]
    key: str = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

# Create a single global client instance
supabase = init_supabase()


def save_progress(session_state, username="Default"):
    """
    Save all relevant progress data directly to Supabase for a specific user.
    Exceptions bubble up to let the main application's sidebar catch them.
    """
    if not username:
        username = "Default"

    # Robust handling for samples_df workspace conversion
    raw_samples = session_state.get("samples_df", None)
    if raw_samples is not None:
        if isinstance(raw_samples, pd.DataFrame):
            # FIXED: orient="records" converts rows into clean string-keyed JSON documents
            samples_serialized = raw_samples.to_dict(orient="records")
        elif isinstance(raw_samples, (dict, list)):
            samples_serialized = raw_samples
        else:
            samples_serialized = None
    else:
        samples_serialized = None

    # Structure the progress metadata payload
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
        "samples_df": samples_serialized
    }
    
    row_payload = {
        "username": username,
        "progress_data": progress_data  
    }
    
    # Let database exceptions rise naturally to the main application catch layout
    supabase.table("user_progress").upsert(row_payload).execute()
    return True


def load_progress(username="Default"):
    """Load progress data from Supabase database for a specific user."""
    if not username:
        username = "Default"
        
    try:
        response = supabase.table("user_progress").select("progress_data").eq("username", username).execute()
        
        if response.data and len(response.data) > 0:
            return response.data[0]["progress_data"]
            
    except Exception as e:
        st.error(f"Error loading progress from Supabase: {e}")
    
    return {}


def restore_progress(session_state, progress_data):
    """Restore loaded database records cleanly back into live Streamlit app memory structure."""
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
    
    # Reconstruct pandas DataFrame safely from database list format structure
    if "samples_df" in progress_data and progress_data["samples_df"]:
        try:
            session_state.samples_df = pd.DataFrame(progress_data["samples_df"])
        except Exception:
            pass