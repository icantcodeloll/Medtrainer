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
    try:
        url: str = st.secrets["SUPABASE_URL"]
        key: str = st.secrets["SUPABASE_KEY"]
        
        if not url or not key:
            raise ValueError("Supabase URL or key is missing from secrets")
            
        return create_client(url, key)
    except Exception as e:
        st.error(f"Failed to initialize Supabase client: {e}")
        raise

# Create a single global client instance
supabase = init_supabase()


def save_progress(session_state, username: str = "Default") -> bool:
    """
    Save all relevant progress data directly to Supabase for a specific user.
    Handles NaN float clearing to ensure JSON compliance.
    
    Args:
        session_state: Streamlit session state object
        username: Username for the current session
        
    Returns:
        bool: True if save was successful, False otherwise
    """
    if not username:
        username = "Default"

    # Structure the progress dictionary
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
        "current_categories": session_state.get("current_categories", [])
    }
    
    row_payload = {
        "username": username,
        "progress_data": progress_data  
    }
    
    supabase.table("user_progress").upsert(row_payload).execute()
    return True

def load_progress(username: str = "Default") -> dict:
    """
    Load progress data from Supabase database for a specific user.
    
    Args:
        username: Username for the current session
        
    Returns:
        dict: Progress data dictionary or empty dict if not found/error
    """
    if not username:
        username = "Default"
        
    try:
        response = supabase.table("user_progress").select("progress_data").eq("username", username).execute()
        
        if response.data and len(response.data) > 0:
            progress_data = response.data[0]["progress_data"]
            # Apply migrations to ensure compatibility
            return migrate_progress_data(progress_data)
            
    except Exception as e:
        st.error(f"Error loading progress from Supabase: {e}")
    
    return {}


def restore_progress(session_state, progress_data: dict) -> None:
    """
    Restore loaded database records cleanly back into live Streamlit app memory structure.
    
    Args:
        session_state: Streamlit session state object
        progress_data: Progress data dictionary to restore
    """
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

def migrate_progress_data(progress_data: dict) -> dict:
    """
    Migrate progress data to handle schema changes between versions.
    This ensures backward compatibility when the data structure changes.
    
    Args:
        progress_data: Progress data dictionary from database
        
    Returns:
        dict: Migrated progress data compatible with current schema
    """
    if not progress_data:
        return {}
    
    # Migration v1: Remove samples_df (no longer saved to Supabase)
    if "samples_df" in progress_data:
        del progress_data["samples_df"]
    
    # Add future migrations here as needed
    # Example: if "new_field" not in progress_data:
    #     progress_data["new_field"] = default_value
    
    return progress_data