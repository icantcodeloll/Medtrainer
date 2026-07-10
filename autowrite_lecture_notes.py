import os
import pandas as pd
from google import genai
from google.genai import types

# 1. Initialize the Gemini Client
client = genai.Client(api_key="AIzaSyCWAry8Yx5jbs5VTso9w5sTfpJjMy_ffk4")

# Load the CSV file
input_filename = "learning_objectives_informative_reports_y2s2.csv"
output_filename = "learning_objectives_informative_reports_y2s2_updated.csv"

print(f"Reading {input_filename}...")
try:
    df = pd.read_csv(input_filename, encoding='utf-8')
except UnicodeDecodeError:
    print("UTF-8 decoding failed, trying 'latin-1' encoding...")
    df = pd.read_csv(input_filename, encoding='latin-1')

# Clean up string columns to strip any hidden spaces that mess up empty checks
df['include'] = df['include'].astype(str).str.strip().str.lower()
if 'explanation' in df.columns:
    df['explanation'] = df['explanation'].fillna('').astype(str).str.strip()
else:
    df['explanation'] = ""

# 2. Define the prompt template
prompt_template = """
You are a medical instructor and I am a student. I am going to give you a learning objective/lecture topic. 
Write me a comprehensive, detailed informative article (around 500 words) based on this topic. 
Do not leave out any clinical or scientific detail. 

Topic: {objective}
Category: {category}
Lecture: {lecture_id}
"""

print("Processing rows and calling Gemini API...")

# 3. Iterate through the dataframe rows
for index, row in df.iterrows():
    # Process if include is 'y' and explanation is completely empty
    if row['include'] == 'y' and row['explanation'] == "":
        print(f"Generating article for Row {index}: {row['objective'][:50]}...")
        
        formatted_prompt = prompt_template.format(
            objective=row['objective'],
            category=row.get('category', 'Medical'),
            lecture_id=row.get('lecture_id', 'General')
        )
        
        try:
            # Call the Gemini model
            response = client.models.generate_content(
                model='gemini-3.1-flash-lite',
                contents=formatted_prompt,
            )
            
            # Use .loc to properly assign back into the master DataFrame
            df.loc[index, 'explanation'] = response.text
            
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            continue

# 4. Save using 'utf-8-sig' so Excel handles paragraphs, linebreaks, and medical symbols correctly
df.to_csv(output_filename, index=False, encoding='utf-8-sig')
print(f"\nTask complete! Updated file saved as: {output_filename}")