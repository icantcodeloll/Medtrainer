# data_manager.py
import os
import shutil
import datetime
import pandas as pd
import streamlit as st

def backup_user_data(user_csv):
    """Creates a timestamped backup of the user's data."""
    if os.path.exists(user_csv):
        backup_dir = "user_backups"
        os.makedirs(backup_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{os.path.basename(user_csv)}.{timestamp}.bak"
        backup_path = os.path.join(backup_dir, backup_filename)
        shutil.copy(user_csv, backup_path)

def synchronize_profile(master_csv, user_csv, join_column):
    """Safely syncs columns and rows from the master file to the user's personal file."""
    if not os.path.exists(master_csv):
        raise FileNotFoundError(f"Master template file '{master_csv}' not found!")
        
    df_master = pd.read_csv(master_csv)
    if not os.path.exists(user_csv):
        df_master.to_csv(user_csv, index=False)
        return True
    
    df_user = pd.read_csv(user_csv)
    is_updated = False
    
    # Sync Missing Columns
    missing_cols = [col for col in df_master.columns if col not in df_user.columns]
    if missing_cols:
        for col in missing_cols:
            mapping = df_master.set_index(join_column)[col].to_dict()
            df_user[col] = df_user[join_column].map(mapping)
        is_updated = True
        
    # Sync Missing Rows
    missing_rows = df_master[~df_master[join_column].isin(df_user[join_column])]
    if not missing_rows.empty:
        df_user = pd.concat([df_user, missing_rows], ignore_index=True)
        is_updated = True
        
    if is_updated:
        df_user.to_csv(user_csv, index=False)
        
    return is_updated

def get_weighted_sample(user_csv, notes_file, join_column, focus_mode, exam_filter, system_filter, exam_weights, mastery_mode, num_questions):
    """Loads datasets, processes filters, builds strict sampling models, and isolates sample segments."""
    df_main = pd.read_csv(user_csv)
    df_notes = pd.read_csv(notes_file)
    df = pd.merge(df_main, df_notes, on=join_column, how='left')

    # Apply filters
    if focus_mode != "All Topics":
        df = df[df['category'] == focus_mode]
    if exam_filter != "All Exams" and 'exam' in df.columns:
        df = df[df['exam'] == exam_filter]
    if system_filter != "All Systems" and 'system' in df.columns:
        df = df[df['system'] == system_filter]

    if 'include' in df.columns:
        df = df[df['include'].astype(str).str.lower().str.strip() == 'y']
        
    if df.empty:
        return pd.DataFrame(), "No active objectives found matching your filter criteria."

    # Smart weights calculation
    df['topic_weight'] = df['category'].map(exam_weights).fillna(0.05)

    if mastery_mode == "on" and 'mastery_score' in df.columns:
        df['mastery_score'] = pd.to_numeric(df['mastery_score'], errors='coerce').fillna(1).astype(int)
        df['mastery_modifier'] = 6 - df['mastery_score']
        df['sampling_weight'] = df['topic_weight'] * df['mastery_modifier']
    else:
        df['sampling_weight'] = df['topic_weight']

    try:
        samples_df = df.sample(min(num_questions, len(df)), weights='sampling_weight', replace=False)
    except ValueError:
        samples_df = df.sample(min(num_questions, len(df)))
        
    return samples_df, None