# pdf_generator.py
import os
import tempfile

try:
    from fpdf import FPDF
except ImportError:
    FPDF = None

def has_pdf_support():
    return FPDF is not None

def create_exam_pdf(exam_text, answer_key, user_answers=None, score=None, max_score=None):
    """Generates a PDF containing the exam questions, answer key, and optionally user selections."""
    if not FPDF:
        return None

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title
    pdf.set_font("Arial", "B", 16)
    if score is not None and max_score is not None:
        pdf.cell(0, 10, f"Practice Exam Results - Score: {score}/{max_score}", ln=True, align="C")
    else:
        pdf.cell(0, 10, "Practice Exam", ln=True, align="C")
    pdf.ln(5)

    # Clean text to prevent Unicode encoding errors in FPDF
    clean_text = exam_text.replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')
    clean_text = clean_text.encode('latin-1', 'replace').decode('latin-1')

    # Print Questions
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 7, clean_text)

    # Add Answer Key & User Answers on a new page
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Exam Summary", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Arial", "", 12)
    for i, ans in enumerate(answer_key):
        text = f"Question {i+1}: Correct Key: {ans}"
        if user_answers and i < len(user_answers):
            u_ans = user_answers[i] if user_answers[i] else "No Answer"
            match_text = " (CORRECT)" if u_ans == ans else " (INCORRECT)"
            text += f" | Your Answer: {u_ans}{match_text}"

        pdf.cell(0, 8, text, ln=True)

    # Save to temp file and return bytes
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        pdf.output(tmp.name)
        tmp_path = tmp.name

    with open(tmp_path, "rb") as f:
        pdf_bytes = f.read()

    os.unlink(tmp_path) # Clean up temp file
    return pdf_bytes