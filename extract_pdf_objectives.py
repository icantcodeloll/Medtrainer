from pypdf import PdfReader
import csv
import re

# Load lecture titles from lecture_notes file
lecture_titles = []
with open('lecture_notes_y1s1.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    lecture_titles = [row[0] for row in reader if row]

# Extract text from PDF
reader = PdfReader('Year 1 S1 learning objectives.pdf')
full_text = ""
for page in reader.pages:
    full_text += page.extract_text() + "\n"

# Print session outlines and lecture titles for comparison
print("Session Outlines from PDF:")
lines = full_text.split('\n')
session_outlines = []
for line in lines:
    if 'Session Outline' in line:
        session_outlines.append(line.strip())
        print(f"{len(session_outlines)}. {line.strip()}")

print(f"\nLecture Titles from CSV:")
for i, title in enumerate(lecture_titles, 1):
    print(f"{i}. {title}")

print(f"\nTotal session outlines: {len(session_outlines)}")
print(f"Total lecture titles: {len(lecture_titles)}")

# Create mapping from session outline to lecture title using content matching
def map_session_to_lecture(session_outline, lecture_titles):
    """Map session outline to lecture title based on content matching"""
    session_lower = session_outline.lower()
    
    # Content-based mappings
    for title in lecture_titles:
        title_lower = title.lower()
        
        # Check for key content matches
        if 'cells i' in session_lower and 'cells 1' in title_lower:
            return title
        elif 'cells ii' in session_lower and 'cells 2' in title_lower:
            return title
        elif 'cell cycle' in session_lower and 'cell cycle' in title_lower:
            return title
        elif 'body fluids' in session_lower and 'body fluids' in title_lower:
            return title
        elif 'molecular biology of the cell 1' in session_lower and 'molecular biology of the cell 1' in title_lower:
            return title
        elif 'molecular biology of the cell 2' in session_lower and 'molecular biology of the cell 2' in title_lower:
            return title
        elif 'cellular energetics' in session_lower and 'cellular energetics' in title_lower:
            return title
        elif 'fuel molecules' in session_lower and 'fuel molecules' in title_lower:
            return title
        elif 'population genetics' in session_lower and 'population genetics' in title_lower:
            return title
        elif 'chromosome disorders' in session_lower and 'chromosome disorders' in title_lower:
            return title
        elif 'applications of gene technologies' in session_lower and 'applications of gene technologies' in title_lower:
            return title
        elif 'genomics' in session_lower and 'genetics genomics' in title_lower:
            return title
        elif 'innate responses' in session_lower and 'innate responses' in title_lower:
            return title
        elif 'adaptive immune' in session_lower and 'adaptive immune' in title_lower:
            return title
        elif 'infection and immunity' in session_lower and 'infection and immunity' in title_lower:
            return title
        elif 'antimicrobial resistance' in session_lower and 'antimicrobial resistance' in title_lower:
            return title
        elif 'parasitic' in session_lower and 'parasitic' in title_lower:
            return title
        elif 'pharmacokinetics' in session_lower and 'pharmacokinetics' in title_lower:
            return title
        elif 'antibiotics' in session_lower and 'antibiotics' in title_lower:
            return title
        elif 'antivirals' in session_lower and 'antivirals' in title_lower:
            return title
        elif 'epithelium' in session_lower and 'epithelium' in title_lower:
            return title
        elif 'connective tissue' in session_lower and 'connective tissue' in title_lower:
            return title
        elif 'bones' in session_lower and 'bones' in title_lower:
            return title
        elif 'resting membrane potential' in session_lower and 'resting membrane potential' in title_lower:
            return title
        elif 'action potential' in session_lower and 'action potential' in title_lower:
            return title
        elif 'synaptic' in session_lower and 'synpatic' in title_lower:
            return title
        elif 'spinal cord' in session_lower and 'spinal cord' in title_lower:
            return title
        elif 'histology of muscle' in session_lower and 'histology of muscle' in title_lower:
            return title
        elif 'introduction to pharmacology' in session_lower and 'introduction to pharmacology' in title_lower:
            return title
        elif 'drugs and neurotransmission' in session_lower and 'drugs and neurotransmission' in title_lower:
            return title
        elif 'introduction to cancer' in session_lower and 'introduction to cancer' in title_lower:
            return title
        elif 'blood composition' in session_lower and 'composition of blood' in title_lower:
            return title
        elif 'haemostasis' in session_lower and 'haemostasis' in title_lower:
            return title
        elif 'anticancer drugs' in session_lower and 'anticancer drugs' in title_lower:
            return title
        elif 'cellular response to injury' in session_lower and 'cellular and tissue response' in title_lower:
            return title
        elif 'introduction to microbes' in session_lower and 'introduction to microbes' in title_lower:
            return title
        elif 'natural barriers' in session_lower and 'natural barriers' in title_lower:
            return title
        elif 'strategies' in session_lower and 'strategies' in title_lower:
            return title
        elif 'fungal' in session_lower and 'fungal' in title_lower:
            return title
    
    # If no match found, return empty string (will be filtered out)
    return ""

# Parse learning objectives - look for bullet points under "Learning Outcomes"
lines = full_text.split('\n')
objectives_with_lecture = []
in_learning_outcomes = False
current_objective = ""
current_session_outline = ""

for line in lines:
    line = line.strip()
    
    # Capture session outline (only when not in Learning Outcomes)
    if not in_learning_outcomes and 'Session Outline' in line:
        current_session_outline = line
    
    # Check if we're entering a Learning Outcomes section
    if 'Learning Outcomes' in line or 'Learning outcomes' in line:
        in_learning_outcomes = True
        continue
    
    # Check if we're leaving a Learning Outcomes section (new session outline)
    if in_learning_outcomes and 'Session Outline' in line:
        in_learning_outcomes = False
        # Save any accumulated objective
        if current_objective:
            mapped_lecture = map_session_to_lecture(current_session_outline, lecture_titles)
            objectives_with_lecture.append((current_objective, mapped_lecture))
            current_objective = ""
        current_session_outline = line
        continue
    
    # Extract bullet points while in Learning Outcomes section
    if in_learning_outcomes:
        if line.startswith('•'):
            # Save previous objective if exists
            if current_objective:
                mapped_lecture = map_session_to_lecture(current_session_outline, lecture_titles)
                objectives_with_lecture.append((current_objective, mapped_lecture))
            # Start new objective
            current_objective = line[1:].strip()
        elif current_objective:
            # Continue multi-line objective
            current_objective += " " + line

# Save last objective
if current_objective:
    mapped_lecture = map_session_to_lecture(current_session_outline, lecture_titles)
    objectives_with_lecture.append((current_objective, mapped_lecture))

# Clean up objectives with lectures
cleaned_objectives = []
for obj, lecture in objectives_with_lecture:
    # Remove page numbers and other metadata from objective
    obj = re.sub(r'\s+\d+\s*$', '', obj)  # Remove trailing numbers with space
    obj = re.sub(r'^\d+\s*', '', obj)  # Remove leading numbers
    obj = re.sub(r'\s+Lecture\s+\d+.*$', '', obj)  # Remove trailing lecture info
    obj = re.sub(r'\s+Week\s+\d+.*$', '', obj)  # Remove trailing week info
    obj = re.sub(r'\s+Theme.*$', '', obj)  # Remove trailing theme info
    obj = re.sub(r'\s+', ' ', obj)  # Normalize whitespace
    obj = obj.strip()
    
    # Clean lecture info
    lecture = re.sub(r'\s+\d+\s*$', '', lecture)  # Remove trailing numbers
    lecture = re.sub(r'\s+', ' ', lecture)  # Normalize whitespace
    lecture = lecture.strip()
    
    # Only keep objectives that have matching lecture titles
    if obj and len(obj) > 10 and lecture:
        cleaned_objectives.append((obj, lecture))

# Remove duplicates while preserving order
seen = set()
unique_objectives = []
for obj, lecture in cleaned_objectives:
    if obj not in seen:
        seen.add(obj)
        unique_objectives.append((obj, lecture))

print(f"Found {len(unique_objectives)} unique learning objectives:")
for i, (obj, lecture) in enumerate(unique_objectives[:20], 1):  # Show first 20
    print(f"{i}. {obj} (Lecture: {lecture})")

# Write to CSV
with open('learning_objectives_informative_reports_y1s1.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['objective', 'lecture'])  # Header
    for obj, lecture in unique_objectives:
        writer.writerow([obj, lecture])

print(f"\nWrote {len(unique_objectives)} objectives to learning_objectives_informative_reports_y1s1.csv")
