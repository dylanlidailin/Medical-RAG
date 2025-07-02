import openai
import time
import pandas as pd
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Step 1: Define mini knowledge base
mini_kb = {
    "Atorvastatin": "Used to treat hyperlipidemia (high cholesterol).",
    "Ibuprofen": "Used for pain relief and as an anti-inflammatory, often on an as-needed (PRN) basis.",
    "Levothyroxine": "Treats hypothyroidism (underactive thyroid).",
    "Losartan": "Prescribed for hypertension (high blood pressure).",
    "Metformin": "Treats Type 2 Diabetes Mellitus.",
    "Metoprolol": "Used to manage hypertension (high blood pressure).",
    "Sertraline": "Treats depression and anxiety disorders.",
    "Gabapentin": "Used for neuropathic pain and seizure prophylaxis.",
    "Alendronate": "Supports bone density in patients with osteoporosis.",
    "Furosemide": "Used to treat fluid overload or edema, such as in heart failure."
}

# Step 2: Load patient dataset
df = pd.read_csv("chest_pain_patients - chest_pain_patients.csv")

# Step 3: Helper function to normalize med names
def normalize_medication(med):
    return med.strip().split()[0].capitalize()

# Step 4: Generate prompts for each medication per patient
prompts = []
for index, row in df.iterrows():
    medications = [normalize_medication(med) for med in str(row['Outpatient_Medications']).split(',')]
    problems = [prob.strip() for prob in str(row['Past_Medical_History']).split(',') if prob.strip()]
    
    for med in medications:
        if med in mini_kb:
            info = mini_kb[med]
            prompt = f"""You are a clinical decision support assistant.
Answer “I don’t know” if the question asked is not recorded in the mini-knowledge base.

Medication: {med}
Info: {info}
Patient Problems: {problems}
Which problem(s) from the list does this medication treat?"""
            prompts.append({
                "Patient_ID": index,
                "Medication": med,
                "Prompt": prompt,
                "Patient_Problems": problems
            })

# Step 5: Convert to DataFrame
prompt_df = pd.DataFrame(prompts)

# Step 6: Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Replace this with your actual API key

# Step 7: Function to call the model
def query_llm(prompt, model="gpt-4", temperature=0):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a clinical decision support assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

# Step 8: Query the LLM for each prompt
responses = []
for i, row in prompt_df.iterrows():
    response = query_llm(row["Prompt"])
    responses.append(response)
    time.sleep(1)  # avoid rate limit

prompt_df["LLM_Response"] = responses

# Step 9: Extract structured medication→problem mapping
def extract_mapping(response, patient_problems):
    if not isinstance(response, str):
        return []
    if response.lower().startswith("error") or response.strip().lower() == "i don’t know":
        return []
    # Try to match any patient problem that appears in the response
    return [prob for prob in patient_problems if prob.lower() in response.lower()]

prompt_df["Mapped_Problems"] = prompt_df.apply(
    lambda row: extract_mapping(row["LLM_Response"], row["Patient_Problems"]),
    axis=1
)

# Step 10: Save results
prompt_df.to_csv("prompts_with_responses.csv", index=False)
print("LLM responses and medication→problem mappings saved to prompts_with_responses.csv")
