from pypdf import PdfReader
import csv
import re

# Load lecture titles from lecture_notes file
lecture_titles = []
with open('lecture_notes_y1s2.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    lecture_titles = [row[0] for row in reader if row]

# Extract text from PDF
reader = PdfReader('Y1S2 learning outcomes.pdf')
full_text = ""
for page in reader.pages:
    full_text += page.extract_text() + "\n"

# Create mapping from session outline to lecture title using content matching
def map_session_to_lecture(session_outline, lecture_titles):
    """Map session outline to lecture title based on content matching"""
    session_lower = session_outline.lower()
    
    # Content-based mappings
    for title in lecture_titles:
        title_lower = title.lower()
        
        # Check for key content matches
        if 'upper limb' in session_lower and 'upper limb' in title_lower:
            return title
        elif 'motor systems' in session_lower and 'motor systems' in title_lower:
            return title
        elif 'skeletal muscle' in session_lower and 'skeletal muscle' in title_lower:
            return title
        elif 'electrical activity of the heart' in session_lower and 'electrical activity of the heart' in title_lower:
            return title
        elif 'heart failure' in session_lower and 'heart failure' in title_lower:
            return title
        elif 'anti-arrythmic' in session_lower and 'anti-arrythmic' in title_lower:
            return title
        elif 'cardiac biomarkers' in session_lower and 'cardiac biomarkers' in title_lower:
            return title
        elif 'arteries' in session_lower and 'arteries' in title_lower:
            return title
        elif 'microcirculation' in session_lower and 'microcirculation' in title_lower:
            return title
        elif 'regulation of map' in session_lower and 'regulation of map' in title_lower:
            return title
        elif 'vasodilator' in session_lower and 'vasodilator' in title_lower:
            return title
        elif 'stroke' in session_lower and 'stroke' in title_lower:
            return title
        elif 'infectious diseases' in session_lower and 'infectious diseases' in title_lower:
            return title
        elif 'coronary circulation' in session_lower and 'coronary circulation' in title_lower:
            return title
        elif 'vascular endothelium' in session_lower and 'vascular endothelium' in title_lower:
            return title
        elif 'atherosclerosis' in session_lower and 'atherosclerosis' in title_lower:
            return title
        elif 'lipid metabolism' in session_lower and 'lipid metabolism' in title_lower:
            return title
        elif 'lipid lowering' in session_lower and 'lipid lowering' in title_lower:
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
    
    # Keep all objectives (not just those with matching lecture titles)
    if obj and len(obj) > 10:
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
with open('learning_objectives_informative_reports_y1s2.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['objective', 'lecture'])  # Header
    for obj, lecture in unique_objectives:
        writer.writerow([obj, lecture])

print(f"\nWrote {len(unique_objectives)} objectives to learning_objectives_informative_reports_y1s2.csv")
