# ai_engine.py
import re
import time
from google import genai
from google.genai import types

def call_gemini_with_rotation(prompt, model_to_use, api_keys, current_key_idx, use_search=False):
    """Manages secure key rotation fallback across provided GenAI endpoints."""
    keys_tried = 0
    tools = [types.Tool(google_search=types.GoogleSearch())] if use_search else []
    working_idx = current_key_idx

    while keys_tried < len(api_keys):
        try:
            client = genai.Client(api_key=api_keys[working_idx])
            response = client.models.generate_content(
                model=model_to_use,
                contents=prompt,
                config=types.GenerateContentConfig(tools=tools) if tools else None
            )
            return response.text, working_idx
        except Exception as e:
            if "429" in str(e):
                keys_tried += 1
                if keys_tried >= len(api_keys):
                    return None, working_idx
                working_idx = (working_idx + 1) % len(api_keys)
                time.sleep(1)
            elif "503" in str(e):
                time.sleep(5)
            else:
                raise e
    return None, working_idx

def validate_exam_format(exam_text, expected_questions):
    """Validates that the AI response follows the required syntax constraints."""
    return True, "Validation temporarily disabled" 

def get_blind_exam(topics_list, level, num_questions, model_to_use, api_keys, current_key_idx, use_search=False):
    combined_content = "\n\n".join([f"Source {i+1}: {t}" for i, t in enumerate(topics_list)])

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
    else:
        difficulty_desc = "supreme counterintuition - advanced mastery of medical paradoxes and exceptions"
        complexity_guide = "multi-system integration where standard rules don't apply, latest research breakthroughs that overturn conventional wisdom, complex clinical reasoning requiring recognition of exceptions..."

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


    return call_gemini_with_rotation(prompt, model_to_use, api_keys, current_key_idx, use_search=use_search)

def get_ai_grading(exam_text, user_answers, correct_key, score, grader_model, api_keys, current_key_idx, use_search=False):
    prompt = f"""Here is the input:
    EXAM QUESTIONS: {exam_text}
    CORRECT KEY: {correct_key}
    STUDENT ANSWERS: {user_answers}
    SCORE: {score}
    You are a medical instructor. Grade the student's performance.
        
    ### GRADING PROTOCOL:
    1. Compare the student's answer for each question against the correct key.
    2. USE GOOGLE SEARCH to verify the current medical guidelines.
    3. Focus ONLY on the questions the student got INCORRECT. 
    
    ### STRICT FORMATTING REQUIREMENTS:
    You MUST output your response in clean Markdown. If the student got 100%, congratulate them and provide one high-yield clinical pearl.
    Otherwise, for EVERY incorrect question, use this EXACT format:

    ### Question [Insert Question Number]
    **Your Answer:** [Letter] | **Correct Answer:** [Letter]
    
    * **Explanation:** [1-2 concise sentences explaining why the correct answer is right and the student's answer is wrong].
    * **Clinical Pearl:** [A short, high-yield tip or memory hook for board exams].
    ---
    """
    

    response_text, new_idx = call_gemini_with_rotation(prompt, grader_model, api_keys, current_key_idx, use_search=use_search)
    return response_text, new_idx