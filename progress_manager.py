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
    
    # Migration v2: Remove profile_picture (no longer supported)
    if "profile_picture" in progress_data:
        del progress_data["profile_picture"]
    
    # Migration v3: Migrate ELO to Glicko rating system
    if "elo_rating" in progress_data and "rating" not in progress_data:
        progress_data["rating"] = progress_data["elo_rating"]
        progress_data["rd"] = 350  # Default RD for migrated players
        progress_data["volatility"] = 0.06  # Default volatility
        progress_data["last_played"] = datetime.now().isoformat()
    
    # Add future migrations here as needed
    # Example: if "new_field" not in progress_data:
    #     progress_data["new_field"] = default_value
    
    return progress_data

def calculate_glicko_rating(rating: float, rd: float, opponent_rating: float, opponent_rd: float, score: float) -> tuple[float, float]:
    """
    Calculate new Glicko rating and rating deviation after a match.
    
    Args:
        rating: Current rating of the player
        rd: Current rating deviation (uncertainty) of the player
        opponent_rating: Rating of the opponent
        opponent_rd: Rating deviation of the opponent
        score: Actual score (1.0 for win, 0.5 for tie, 0.0 for loss)
        
    Returns:
        tuple: (new_rating, new_rd)
    """
    import math
    
    # Glicko constants
    q = math.log(10) / 400
    c = 20  # RD increase per period
    default_rd = 350  # Initial RD for new players
    
    # Calculate g(RD_j)
    g_opponent = 1 / math.sqrt(1 + 3 * q**2 * opponent_rd**2 / math.pi**2)
    
    # Calculate expected score
    expected_score = 1 / (1 + 10 ** (-g_opponent * (rating - opponent_rating) / 400))
    
    # Calculate d^2
    d_squared = 1 / (q**2 * g_opponent**2 * expected_score * (1 - expected_score))
    
    # Calculate new RD
    new_rd_squared = 1 / (1/rd**2 + 1/d_squared)
    new_rd = math.sqrt(new_rd_squared)
    
    # Calculate new rating
    new_rating = rating + q * new_rd_squared * g_opponent * (score - expected_score)
    
    return new_rating, new_rd

def update_player_elo(username: str, opponent_username: str, result: str) -> bool:
    """
    Update Glicko ratings for both players after a multiplayer match.
    
    Args:
        username: First player's username
        opponent_username: Second player's username
        result: 'win', 'loss', or 'tie' from the perspective of username
        
    Returns:
        bool: True if update was successful
    """
    import math
    from datetime import datetime, timedelta
    
    try:
        # Get current Glicko ratings
        response = supabase.table("player_elo_ratings").select("*").in_("username", [username, opponent_username]).execute()
        
        players = {p["username"]: p for p in response.data} if response.data else {}
        
        # Glicko constants
        default_rating = 1500
        default_rd = 350
        rd_increase_period = 30  # days
        rd_increase_factor = 20  # RD increase per period
        
        # Initialize players if they don't exist
        if username not in players:
            players[username] = {
                "username": username, 
                "rating": default_rating, 
                "rd": default_rd,
                "volatility": 0.06,
                "games_played": 0, 
                "games_won": 0, 
                "games_lost": 0, 
                "games_tied": 0,
                "last_played": datetime.now().isoformat()
            }
            supabase.table("player_elo_ratings").insert(players[username]).execute()
        else:
            # Migrate existing ELO data to Glicko if needed
            if "elo_rating" in players[username] and "rating" not in players[username]:
                players[username]["rating"] = players[username]["elo_rating"]
                players[username]["rd"] = default_rd
                players[username]["volatility"] = 0.06
                players[username]["last_played"] = datetime.now().isoformat()
        
        if opponent_username not in players:
            players[opponent_username] = {
                "username": opponent_username, 
                "rating": default_rating, 
                "rd": default_rd,
                "volatility": 0.06,
                "games_played": 0, 
                "games_won": 0, 
                "games_lost": 0, 
                "games_tied": 0,
                "last_played": datetime.now().isoformat()
            }
            supabase.table("player_elo_ratings").insert(players[opponent_username]).execute()
        else:
            # Migrate existing ELO data to Glicko if needed
            if "elo_rating" in players[opponent_username] and "rating" not in players[opponent_username]:
                players[opponent_username]["rating"] = players[opponent_username]["elo_rating"]
                players[opponent_username]["rd"] = default_rd
                players[opponent_username]["volatility"] = 0.06
                players[opponent_username]["last_played"] = datetime.now().isoformat()
        
        # Apply time decay to RD for inactive players
        current_time = datetime.now()
        for player_name, player_data in players.items():
            if "last_played" in player_data and player_data["last_played"]:
                last_played = datetime.fromisoformat(player_data["last_played"])
                days_inactive = (current_time - last_played).days
                if days_inactive > 0:
                    # Increase RD based on time inactive
                    periods_inactive = days_inactive / rd_increase_period
                    player_data["rd"] = min(350, math.sqrt(player_data["rd"]**2 + (periods_inactive * rd_increase_factor)**2))
        
        player_rating = players[username]["rating"]
        player_rd = players[username]["rd"]
        opponent_rating = players[opponent_username]["rating"]
        opponent_rd = players[opponent_username]["rd"]
        
        # Calculate scores based on result
        if result == "win":
            player_score = 1.0
            opponent_score = 0.0
            players[username]["games_won"] += 1
            players[opponent_username]["games_lost"] += 1
        elif result == "loss":
            player_score = 0.0
            opponent_score = 1.0
            players[username]["games_lost"] += 1
            players[opponent_username]["games_won"] += 1
        else:  # tie
            player_score = 0.5
            opponent_score = 0.5
            players[username]["games_tied"] += 1
            players[opponent_username]["games_tied"] += 1
        
        # Calculate new Glicko ratings
        new_player_rating, new_player_rd = calculate_glicko_rating(
            player_rating, player_rd, opponent_rating, opponent_rd, player_score
        )
        new_opponent_rating, new_opponent_rd = calculate_glicko_rating(
            opponent_rating, opponent_rd, player_rating, player_rd, opponent_score
        )
        
        # Update both players
        players[username]["rating"] = new_player_rating
        players[username]["rd"] = new_player_rd
        players[username]["games_played"] += 1
        players[username]["last_played"] = current_time.isoformat()
        
        players[opponent_username]["rating"] = new_opponent_rating
        players[opponent_username]["rd"] = new_opponent_rd
        players[opponent_username]["games_played"] += 1
        players[opponent_username]["last_played"] = current_time.isoformat()
        
        # Update database with new fields
        supabase.table("player_elo_ratings").update(players[username]).eq("username", username).execute()
        supabase.table("player_elo_ratings").update(players[opponent_username]).eq("username", opponent_username).execute()
        
        return True
    except Exception as e:
        print(f"Error updating Glicko ratings: {e}")
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
        # Get Glicko leaderboard (top 50 by rating)
        # Try to use 'rating' field first, fallback to 'elo_rating' for backward compatibility
        try:
            elo_response = supabase.table("player_elo_ratings").select("*").order("rating", desc=True).limit(50).execute()
        except Exception:
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