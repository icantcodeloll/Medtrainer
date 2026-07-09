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
        "current_categories": session_state.get("current_categories", []),
        "profile_picture": session_state.get("profile_picture", None),
        "leaderboard_opt_in": session_state.get("leaderboard_opt_in", False)
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

def calculate_elo_rating(winner_elo: int, loser_elo: int, k_factor: int = 32) -> tuple[int, int]:
    """
    Calculate new ELO ratings after a match using the standard ELO formula.
    
    Args:
        winner_elo: Current ELO rating of the winner
        loser_elo: Current ELO rating of the loser
        k_factor: K-factor determines how much ratings change (default 32)
        
    Returns:
        tuple: (new_winner_elo, new_loser_elo)
    """
    # Calculate expected scores
    expected_winner = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
    expected_loser = 1 / (1 + 10 ** ((winner_elo - loser_elo) / 400))
    
    # Calculate new ratings
    new_winner_elo = int(winner_elo + k_factor * (1 - expected_winner))
    new_loser_elo = int(loser_elo + k_factor * (0 - expected_loser))
    
    return new_winner_elo, new_loser_elo

def update_player_elo(username: str, opponent_username: str, result: str) -> bool:
    """
    Update ELO ratings for both players after a multiplayer match.
    
    Args:
        username: First player's username
        opponent_username: Second player's username
        result: 'win', 'loss', or 'tie' from the perspective of username
        
    Returns:
        bool: True if update was successful
    """
    try:
        # Get current ELO ratings
        response = supabase.table("player_elo_ratings").select("*").in_("username", [username, opponent_username]).execute()
        
        players = {p["username"]: p for p in response.data} if response.data else {}
        
        # Initialize players if they don't exist
        if username not in players:
            players[username] = {"username": username, "elo_rating": 1000, "games_played": 0, "games_won": 0, "games_lost": 0, "games_tied": 0}
            supabase.table("player_elo_ratings").insert(players[username]).execute()
        
        if opponent_username not in players:
            players[opponent_username] = {"username": opponent_username, "elo_rating": 1000, "games_played": 0, "games_won": 0, "games_lost": 0, "games_tied": 0}
            supabase.table("player_elo_ratings").insert(players[opponent_username]).execute()
        
        player_elo = players[username]["elo_rating"]
        opponent_elo = players[opponent_username]["elo_rating"]
        
        # Calculate new ELO ratings
        if result == "win":
            new_player_elo, new_opponent_elo = calculate_elo_rating(player_elo, opponent_elo)
            players[username]["games_won"] += 1
            players[opponent_username]["games_lost"] += 1
        elif result == "loss":
            new_opponent_elo, new_player_elo = calculate_elo_rating(opponent_elo, player_elo)
            players[username]["games_lost"] += 1
            players[opponent_username]["games_won"] += 1
        else:  # tie
            # For ties, both players move toward the average
            avg_elo = (player_elo + opponent_elo) / 2
            new_player_elo = int(player_elo + 16 * (0.5 - (1 / (1 + 10 ** ((opponent_elo - player_elo) / 400)))))
            new_opponent_elo = int(opponent_elo + 16 * (0.5 - (1 / (1 + 10 ** ((player_elo - opponent_elo) / 400)))))
            players[username]["games_tied"] += 1
            players[opponent_username]["games_tied"] += 1
        
        # Update both players
        players[username]["elo_rating"] = new_player_elo
        players[username]["games_played"] += 1
        
        players[opponent_username]["elo_rating"] = new_opponent_elo
        players[opponent_username]["games_played"] += 1
        
        supabase.table("player_elo_ratings").update(players[username]).eq("username", username).execute()
        supabase.table("player_elo_ratings").update(players[opponent_username]).eq("username", opponent_username).execute()
        
        return True
    except Exception as e:
        print(f"Error updating ELO ratings: {e}")
        return False

def save_single_player_score(username: str, score: int, total_questions: int, accuracy: float, difficulty: int, time_taken: float, categories: list) -> bool:
    """
    Save a single player speed quiz score to the leaderboard.
    
    Args:
        username: Player's username
        score: Number of correct answers
        total_questions: Total number of questions
        accuracy: Accuracy percentage
        difficulty: Difficulty level (1-50)
        time_taken: Time taken in seconds
        categories: List of categories used
        
    Returns:
        bool: True if save was successful
    """
    try:
        score_data = {
            "username": username,
            "score": score,
            "total_questions": total_questions,
            "accuracy": accuracy,
            "difficulty": difficulty,
            "time_taken": time_taken,
            "categories": categories
        }
        supabase.table("single_player_scores").insert(score_data).execute()
        return True
    except Exception as e:
        print(f"Error saving single player score: {e}")
        return False

def get_leaderboard_data() -> dict:
    """
    Fetch leaderboard data for both single player and multiplayer.
    Also calculates overall accuracy and total questions completed per user.
    
    Returns:
        dict: Dictionary containing 'elo_leaderboard', 'single_player_leaderboard', 
              'accuracy_leaderboard', and 'questions_leaderboard'
    """
    try:
        # Get ELO leaderboard (top 50)
        elo_response = supabase.table("player_elo_ratings").select("*").order("elo_rating", desc=True).limit(50).execute()
        elo_leaderboard = elo_response.data if elo_response.data else []
        
        # Get single player leaderboard (top 50 by score)
        single_response = supabase.table("single_player_scores").select("*").order("score", desc=True).limit(50).execute()
        single_leaderboard = single_response.data if single_response.data else []
        
        # Get all single player scores for accuracy and questions calculations
        all_scores_response = supabase.table("single_player_scores").select("*").execute()
        all_scores = all_scores_response.data if all_scores_response.data else []
        
        # Calculate overall accuracy per user
        user_accuracy = {}
        for score in all_scores:
            username = score['username']
            if username not in user_accuracy:
                user_accuracy[username] = {'total_correct': 0, 'total_questions': 0}
            user_accuracy[username]['total_correct'] += score['score']
            user_accuracy[username]['total_questions'] += score['total_questions']
        
        # Convert to list and sort by accuracy
        accuracy_leaderboard = []
        for username, data in user_accuracy.items():
            overall_accuracy = (data['total_correct'] / data['total_questions'] * 100) if data['total_questions'] > 0 else 0
            accuracy_leaderboard.append({
                'username': username,
                'overall_accuracy': overall_accuracy,
                'total_correct': data['total_correct'],
                'total_questions': data['total_questions']
            })
        
        # Sort by overall accuracy (descending)
        accuracy_leaderboard.sort(key=lambda x: x['overall_accuracy'], reverse=True)
        
        # Calculate total questions completed per user
        user_questions = {}
        for score in all_scores:
            username = score['username']
            if username not in user_questions:
                user_questions[username] = 0
            user_questions[username] += score['total_questions']
        
        # Convert to list and sort by total questions
        questions_leaderboard = []
        for username, total_questions in user_questions.items():
            questions_leaderboard.append({
                'username': username,
                'total_questions': total_questions
            })
        
        # Sort by total questions (descending)
        questions_leaderboard.sort(key=lambda x: x['total_questions'], reverse=True)
        
        return {
            "elo_leaderboard": elo_leaderboard,
            "single_player_leaderboard": single_leaderboard,
            "accuracy_leaderboard": accuracy_leaderboard,
            "questions_leaderboard": questions_leaderboard
        }
    except Exception as e:
        print(f"Error fetching leaderboard data: {e}")
        return {
            "elo_leaderboard": [],
            "single_player_leaderboard": [],
            "accuracy_leaderboard": [],
            "questions_leaderboard": []
        }