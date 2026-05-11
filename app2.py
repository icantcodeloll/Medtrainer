from flask import Flask, render_template, request, jsonify, session
import pandas as pd
from google import genai
from google.genai import types 
import time
import re
import os
import json
from datetime import datetime
from progress_manager import save_progress, load_progress, restore_progress

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production

# API Keys (same as original)
API_KEYS = [
    'your-api-key-1',
    'your-api-key-2',
    # Add more keys as needed
]

# File paths
CSV_FILE = "learning_objectives_informative_reports.csv"
NOTES_FILE = "lecture_notes.csv"
JOIN_COLUMN = "lecture_id"
EXAM_MODEL = 'gemini-2.5-flash'
GRADER_MODEL = 'gemini-2.5-flash-lite'

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def get_client():
    """Get Gemini client with current API key"""
    return genai.Client(api_key=API_KEYS[0])  # Simplified for Flask

def call_gemini_with_rotation(prompt, model_to_use, use_search=False, timeout_per_question=3):
    """Call Gemini with timeout and retry logic"""
    try:
        client = get_client()
        
        # Calculate timeout based on number of questions
        num_questions_match = re.search(r'EXACTLY (\d+)', prompt)
        num_questions = int(num_questions_match.group(1)) if num_questions_match else 10
        total_timeout = timeout_per_question * num_questions
        
        response = client.models.generate_content(
            model=model_to_use, 
            contents=prompt,
            config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())] if use_search else None
        )
        return response.text
    except Exception as e:
        if "429" in str(e):
            return None
        elif "503" in str(e):
            time.sleep(5)
        elif "timeout" in str(e).lower() or "deadline" in str(e).lower():
            return None
        else:
            return None

def validate_exam_format(exam_text, expected_questions):
    """Validate that AI response follows the correct format"""
    if not exam_text or not exam_text.strip():
        return False, "Empty response"
    
    # Check if starts with question number
    if not re.match(r'^\s*1\.\s', exam_text.strip()):
        return False, "Does not start with '1. '"
    
    # Check for correct number of questions
    question_pattern = r'^\s*\d+\.\s'
    questions = re.findall(question_pattern, exam_text, re.MULTILINE)
    if len(questions) != expected_questions:
        return False, f"Expected {expected_questions} questions, found {len(questions)}"
    
    # Check for answer key format
    key_pattern = r'\[KEY:\s*[A-D,\s]+\]$'
    if not re.search(key_pattern, exam_text.strip()):
        return False, "Missing or malformed answer key"
    
    # Check each question has A, B, C, D options
    lines = exam_text.split('\n')
    current_question = 0
    option_count = 0
    
    for line in lines:
        line = line.strip()
        if re.match(r'^\d+\.\s', line):
            current_question += 1
            option_count = 0
        elif re.match(r'^[A-D]\.\s', line):
            option_count += 1
    
    if current_question != expected_questions:
        return False, f"Question count mismatch: {current_question} vs {expected_questions}"
    
    return True, "Valid format"

def get_blind_exam(topics_list, level, num_questions):
    """Generate exam questions with difficulty calibration"""
    combined_content = "\n\n".join([f"Source {i+1}: {t}" for i, t in enumerate(topics_list)])
    
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
        complexity_guide = "multi-system integration where standard rules don't apply, latest research breakthroughs that overturn conventional wisdom, complex clinical reasoning requiring recognition of exceptions, niche subspecialty knowledge where intuitive answers are wrong, molecular-level pathophysiology that defies simple explanations, emerging treatment protocols with paradoxical mechanisms, rare disease patterns that mimic opposite conditions, advanced diagnostic challenges where obvious answer is incorrect"
    
    prompt = f"""
    You are a medical board examiner. 
    TASK: Generate EXACTLY {num_questions} Multiple Choice Questions (1 per snippet provided below).
    DIFFICULTY LEVEL: {level}/50.
    DIFFICULTY DESCRIPTION: {difficulty_desc}.
    COMPLEXITY GUIDANCE: {complexity_guide}.

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
    6. The VERY LAST line must be: [KEY: A, B, C, D, A, B, C, D, A, B]

    CONTENT REQUIREMENTS:
    7. Use the STUDY MATERIAL provided below as the base.
    8. USE GOOGLE SEARCH to supplement with latest medical guidelines and realistic clinical cases.
    9. Ensure questions match difficulty level {level}:
       - Level 1-5: Intuitive basics - straightforward concepts that follow common sense, obvious anatomy/physiology
       - Level 6-15: Logical progression - predictable clinical patterns, standard protocols, common conditions with textbook presentations
       - Level 16-25: Complex but predictable - applied knowledge with clear patterns, differential diagnosis with logical elimination
       - Level 26-35: Challenging patterns - specialized knowledge with some counterintuitive elements, presentations deviating from textbook
       - Level 36-45: Counterintuitive expert - knowledge that defies common medical assumptions, paradoxical responses, conditions presenting opposite to expected
       - Level 46-50: Supreme counterintuition - medical paradoxes and exceptions where intuitive answers are wrong, conditions that mimic opposite presentations, treatments with paradoxical mechanisms, diagnostic challenges where obvious answer is incorrect
    10. **STRICT ANSWER DISTRIBUTION**: Each letter (A, B, C, D) correct exactly 2-3 times.
    11. All options must be plausible distractors.

    STUDY MATERIAL:
    {combined_content}

    REMEMBER: Start with '1. ' immediately. No introduction. End with [KEY: format].
    """
    
    # Retry logic with simple validation
    max_retries = 3
    for attempt in range(max_retries):
        exam_text = call_gemini_with_rotation(prompt, EXAM_MODEL, use_search=True, timeout_per_question=3)
        
        # Validate response
        is_valid, error_msg = validate_exam_format(exam_text, num_questions)
        
        if is_valid:
            return exam_text
        else:
            if attempt < max_retries - 1:
                # Add more specific instructions for retry
                retry_prompt = prompt + f"""
                
ATTENTION: Your previous response FAILED validation. Error: {error_msg}
Please regenerate with STRICT adherence to the format requirements above.
Start immediately with '1. ' and end with the correct [KEY: format].
"""
                exam_text = call_gemini_with_rotation(retry_prompt, EXAM_MODEL, use_search=True, timeout_per_question=3)
                is_valid, error_msg = validate_exam_format(exam_text, num_questions)
                if is_valid:
                    return exam_text
        
        # If all retries fail, return the last attempt with a warning
        return exam_text

def get_ai_grading(exam_text, user_answers, correct_key, score):
    """Get AI grading for submitted exam"""
    prompt = f"""
    Here is the input:
    EXAM QUESTIONS: {exam_text}
    CORRECT KEY: {correct_key}
    STUDENT ANSWERS: {user_answers}
    SCORE: {score}

    You are a medical instructor. Grade the student's performance.
    ### GRADING PROTOCOL:
    1. Compare the student's answer for each index (1 through 10) against the correct key.
    2. USE GOOGLE SEARCH to check if the correct key is correct, 
    3. If they match, it is correct. If they differ, it is incorrect.
    4. Provide a brief explanation for any incorrect answers.
    """
    
    # Using search during grading ensures explanations match current guidelines
    return call_gemini_with_rotation(prompt, GRADER_MODEL, use_search=True)

# ==========================================
# 3. FLASK ROUTES
# ==========================================

@app.route('/')
def index():
    """Main page - exam generation and taking"""
    if request.method == 'GET':
        # Load CSV data
        try:
            df_main = pd.read_csv(CSV_FILE)
            df_notes = pd.read_csv(NOTES_FILE)
            df = pd.merge(df_main, df_notes, on=JOIN_COLUMN, how='left')
        except Exception as e:
            return render_template('error.html', error=str(e))
        
        # Get form data
        categories = df['category'].fillna("Uncategorized").astype(str).unique().tolist()
        all_categories = ["All Topics"] + sorted(categories)
        
        exams = df['exam'].fillna("Uncategorized").astype(str).unique().tolist() if 'exam' in df.columns else []
        all_exams = ["All Exams"] + sorted(exams)
        
        return render_template('index.html', 
                         categories=all_categories,
                         exams=all_exams,
                         current_level=session.get('current_level', 10),
                         num_questions=session.get('num_questions', 10))

@app.route('/generate', methods=['POST'])
def generate_exam():
    """Generate exam questions"""
    data = request.get_json()
    
    # Validate inputs
    if not data or 'level' not in data or 'num_questions' not in data:
        return jsonify({'error': 'Missing required fields'})
    
    level = int(data['level'])
    num_questions = int(data['num_questions'])
    focus_mode = data.get('focus_mode', 'All Topics')
    exam_filter = data.get('exam_filter', 'All Exams')
    
    try:
        # Load and filter data
        df_main = pd.read_csv(CSV_FILE)
        df_notes = pd.read_csv(NOTES_FILE)
        df = pd.merge(df_main, df_notes, on=JOIN_COLUMN, how='left')
        
        # Apply filters
        if focus_mode != "All Topics":
            df = df[df['category'] == focus_mode]
            if df.empty:
                return jsonify({'error': f"No questions found for {focus_mode}"})
        
        if exam_filter != "All Exams" and 'exam' in df.columns:
            df = df[df['exam'] == exam_filter]
            if df.empty:
                return jsonify({'error': f"No questions found for {exam_filter}"})
        
        # Smart sampling
        if 'mastery_score' in df.columns:
            df['mastery_score'] = pd.to_numeric(df['mastery_score'], errors='coerce').fillna(1).astype(int)
            weak_pool = df[df['mastery_score'] <= 3]
            if len(weak_pool) >= num_questions:
                samples_df = weak_pool.sample(num_questions)
            else:
                samples_df = df.sample(min(num_questions, len(df)))
        else:
            samples_df = df.sample(min(num_questions, len(df)))
        
        # Generate topics list
        samples = (samples_df['explanation'] + "\n[Notes: " + samples_df['content'].fillna('') + "]").tolist()
        
        # Generate exam
        raw_response = get_blind_exam(samples, level, num_questions)
        
        if not raw_response:
            return jsonify({'error': 'Failed to generate exam'})
        
        # Parse response
        text, key_part = raw_response.split("[KEY:")
        questions = [q.strip() for q in text.split('\n') if q.strip() and re.match(r'^\d+\.', q.strip())]
        key = [ans.strip() for ans in key_part.split(',')]
        
        # Store in session
        session['current_level'] = level
        session['num_questions'] = num_questions
        session['current_exam'] = raw_response
        session['current_key'] = key
        session['questions'] = questions
        
        return jsonify({
            'success': True,
            'questions': questions,
            'key': key,
            'level': level
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/submit', methods=['POST'])
def submit_exam():
    """Submit exam for grading"""
    data = request.get_json()
    
    if not data or 'answers' not in data:
        return jsonify({'error': 'Missing answers'})
    
    answers = data['answers']
    
    # Get stored exam data
    if 'current_exam' not in session or 'current_key' not in session:
        return jsonify({'error': 'No active exam found'})
    
    questions = session.get('questions', [])
    correct_key = session.get('current_key', [])
    
    # Calculate score
    score = sum(1 for i, ans in enumerate(answers) if i < len(correct_key) and ans == correct_key[i])
    
    # Get AI grading
    user_input = "\n".join([f"Q{i+1}: {ans if ans else 'No Answer'}" for i, ans in enumerate(answers)])
    correct_key_formatted = "\n".join([f"Q{i+1}: {ans}" for i, ans in enumerate(correct_key)])
    
    feedback = get_ai_grading(
        session.get('current_exam', ''),
        user_input,
        correct_key_formatted,
        score
    )
    
    return jsonify({
        'success': True,
        'score': score,
        'total': len(correct_key),
        'feedback': feedback,
        'percentage': round((score / len(correct_key)) * 100, 1)
    })

# ==========================================
# 4. HTML TEMPLATES
# ==========================================

# Create templates directory if it doesn't exist
if not os.path.exists('templates'):
    os.makedirs('templates')

# Base template
base_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medtrainer - Medical Exam Trainer</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .header {
            text-align: center;
            color: #2c3e50;
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #555;
        }
        select, input {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
        }
        button {
            background-color: #2c3e50;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            width: 100%;
        }
        button:hover {
            background-color: #45a049;
        }
        .question {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 15px;
            border-left: 4px solid #2c3e50;
        }
        .question-text {
            font-size: 18px;
            margin-bottom: 15px;
            line-height: 1.5;
        }
        .options {
            margin-bottom: 15px;
        }
        .option {
            display: block;
            padding: 10px;
            margin-bottom: 8px;
            background-color: #fff;
            border: 1px solid #e9ecef;
            border-radius: 5px;
            cursor: pointer;
        }
        .option:hover {
            background-color: #f0f0f0;
        }
        .option input:checked + label {
            background-color: #e3f2fd;
            border-color: #2c3e50;
        }
        .result {
            background-color: #d4edda;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
        }
        .score {
            font-size: 24px;
            font-weight: bold;
            color: #155724;
            margin-bottom: 10px;
        }
        .feedback {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #155724;
            white-space: pre-wrap;
            font-family: monospace;
        }
        .error {
            background-color: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 5px;
            border: 1px solid #f5c6cb;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Medtrainer - Medical Exam Trainer</h1>
            <p>Master medical concepts through adaptive AI-powered questions</p>
        </div>
        {content}
    </div>
</body>
</html>
"""

# Error template
error_template = base_template.replace('{content}', """
    <div class="container">
        <div class="error">
            <h2>Error</h2>
            <p>{error}</p>
            <p><a href="/">← Back to Home</a></p>
        </div>
    </div>
""")

# Index template
index_template = base_template.replace('{content}', """
    <div class="container">
        <div class="header">
            <h1>Medtrainer - Medical Exam Trainer</h1>
            <p>Master medical concepts through adaptive AI-powered questions</p>
        </div>
        
        <div class="container">
            <h2>Generate New Exam</h2>
            <form method="POST" action="/generate">
                <div class="form-group">
                    <label for="level">Difficulty Level (1-50):</label>
                    <select name="level" id="level">
                        {% for i in range(1, 51) %}
                            <option value="{{ i }}" {% if current_level == i %}selected{% endif %}>{{ i }}</option>
                        {% endfor %}
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="num_questions">Number of Questions:</label>
                    <select name="num_questions" id="num_questions">
                        {% for i in range(1, 21) %}
                            <option value="{{ i }}" {% if num_questions == i %}selected{% endif %}>{{ i }}</option>
                        {% endfor %}
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="focus_mode">Focus Mode:</label>
                    <select name="focus_mode" id="focus_mode">
                        <option value="All Topics" {% if focus_mode == "All Topics" %}selected{% endif %}>All Topics</option>
                        {% for category in categories %}
                            <option value="{{ category }}" {% if focus_mode == category %}selected{% endif %}>{{ category }}</option>
                        {% endfor %}
                    </select>
                </div>
                
                {% if exams %}
                <div class="form-group">
                    <label for="exam_filter">Exam Filter:</label>
                    <select name="exam_filter" id="exam_filter">
                        <option value="All Exams" {% if exam_filter == "All Exams" %}selected{% endif %}>All Exams</option>
                        {% for exam in exams %}
                            <option value="{{ exam }}" {% if exam_filter == exam %}selected{% endif %}>{{ exam }}</option>
                        {% endfor %}
                    </select>
                </div>
                {% endif %}
                
                <button type="submit">Generate Exam</button>
            </form>
        </div>
    </div>
""")

# Exam template
exam_template = base_template.replace('{content}', """
    <div class="container">
        <div class="header">
            <h1>Exam - Level {{ level }}</h1>
            <p>Answer all questions to the best of your ability</p>
        </div>
        
        <div class="container">
            <form method="POST" action="/submit">
                {% for i, question in enumerate(questions) %}
                <div class="question">
                    <div class="question-text">
                        {{ i+1 }}. {{ question }}
                    </div>
                    <div class="options">
                        {% for option in ['A', 'B', 'C', 'D'] %}
                        <div class="option">
                            <input type="radio" name="answer_{{ i }}" id="answer_{{ i }}_{{ option }}" value="{{ option }}">
                            <label for="answer_{{ i }}_{{ option }}">{{ option }}. </label>
                        </div>
                        {% endfor %}
                    </div>
                </div>
                {% endfor %}
                
                <button type="submit">Submit for Grading</button>
            </form>
        </div>
    </div>
""")

# Result template
result_template = base_template.replace('{content}', """
    <div class="container">
        <div class="header">
            <h1>Exam Results</h1>
            <p>Your performance has been evaluated</p>
        </div>
        
        <div class="result">
            <div class="score">
                Score: {{ score }}/{{ total }}
                <br>Percentage: {{ percentage }}%
            </div>
            
            <div class="feedback">
                <h3>AI Instructor Feedback:</h3>
                <pre>{{ feedback }}</pre>
            </div>
        </div>
        
        <div style="text-align: center; margin-top: 30px;">
            <a href="/">← Generate New Exam</a>
            <a href="/progress">Progress Management</a>
        </div>
    </div>
""")

# Progress management template
progress_template = base_template.replace('{content}', """
    <div class="container">
        <div class="header">
            <h1>Progress Management</h1>
            <p>Track your learning journey and manage your progress</p>
        </div>
        
        <div class="container">
            <h2>Current Progress</h2>
            <p>Level: {{ current_level }}/50</p>
            <p>Questions per exam: {{ num_questions }}</p>
            
            <h3>Actions</h3>
            <a href="/">Generate New Exam</a> |
            <a href="/save">Save Progress</a> |
            <a href="/reset">Reset Progress</a>
        </div>
    </div>
""")

# ==========================================
# 5. ADDITIONAL ROUTES
# ==========================================

@app.route('/progress')
def progress():
    """Progress management page"""
    return render_template('progress_template.html',
                         current_level=session.get('current_level', 10),
                         num_questions=session.get('num_questions', 10))

@app.route('/save')
def save_progress_route():
    """Save progress to file"""
    try:
        progress_data = {
            'timestamp': datetime.now().isoformat(),
            'current_level': session.get('current_level', 10),
            'num_questions': session.get('num_questions', 10),
            'last_score': session.get('last_score', 0),
            'missed_questions': session.get('missed_questions', []),
        }
        
        with open('user_progress.json', 'w') as f:
            json.dump(progress_data, f, indent=2)
        
        return render_template('progress_template.html',
                         current_level=session.get('current_level', 10),
                         num_questions=session.get('num_questions', 10))
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/reset')
def reset_progress():
    """Reset all progress"""
    try:
        # Clear session
        session.clear()
        
        # Delete progress file
        if os.path.exists('user_progress.json'):
            os.remove('user_progress.json')
        
        return render_template('progress_template.html',
                         current_level=10,
                         num_questions=10)
    except Exception as e:
        return render_template('error.html', error=str(e))

# ==========================================
# 6. RUN APPLICATION
# ==========================================
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
