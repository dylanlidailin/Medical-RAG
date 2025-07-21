import os
import time
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Step 0: Load environment variables and initialize OpenAI client
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Missing OPENAI_API_KEY in environment variables.")
client = OpenAI(api_key=api_key)

# Step 1: Load drug knowledge base from CSV file
kb_df = pd.read_csv("rxnorm_enriched_chunks.csv")

# Build dictionary: 
# Use STR (preferred name) as key, and Text_Chunk as value
mini_kb = dict(zip(kb_df["STR"].str.capitalize(), kb_df["Text_Chunk"]))

# Step 2: Load patient dataset
df = pd.read_csv("chest_pain_patients.csv")

# Step 3: Normalize medication name
def normalize_medication(med_name: str) -> str:
    return med_name.strip().split()[0].capitalize()

# Step 4: Prepare and save prompt template
prompt_template = (
    "You are a clinical decision support assistant."
    " Use the medication information and patient's problem list to identify which problem(s) the medication treats."
    " If the medication is not in the knowledge base, reply 'I don’t know'.\n\n"
    "Medication: {Medication}\n"
    "Info: {Info}\n"
    "Patient Problems: {Patient_Problems}\n"
    "Which problem(s) from the list does this medication treat?"
)

with open("prompt_template.txt", "w") as f:
    f.write(prompt_template)

# Step 5: Generate prompts
prompts = []
for idx, row in df.iterrows():
    raw_meds = str(row.get("Outpatient_Medications", ""))
    patient_problems = [p.strip() for p in str(row.get("Past_Medical_History", "")).split(',') if p.strip()]
    meds = [normalize_medication(m) for m in raw_meds.split(',')]

    for med in meds:
        info = mini_kb.get(med)
        prompt_text = prompt_template.format(
            Medication=med,
            Info=info or "Description not available.",
            Patient_Problems=str(patient_problems)
        )
        prompt_filename = f"prompt_{idx}_{med}.txt"
        with open(prompt_filename, "w") as f:
            f.write(prompt_text)

        prompts.append({
            "Patient_ID": idx,
            "Medication": med,
            "Prompt_File": prompt_filename,
            "Prompt_Text": prompt_text,
            "Info": info or "Description not available.",
            "Patient_Problems": patient_problems
        })

# Step 6: Query OpenAI Chat API
def query_openai(prompt_text: str, model: str = "gpt-4", temperature: float = 0.0) -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a clinical decision support assistant."},
                {"role": "user", "content": prompt_text}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

responses = []
for entry in prompts:
    resp = query_openai(entry["Prompt_Text"])
    responses.append(resp)
    time.sleep(1)  # to avoid rate limit

# Step 7: Extract mapping from LLM response
def extract_mapping(response: str, problems: list) -> list:
    if not response or response.lower().startswith("i don’t know"):
        return []
    return [p for p in problems if p.lower() in response.lower()]

# Step 8: Save results
result_df = pd.DataFrame(prompts)
result_df["LLM_Response"] = responses
result_df["Treated_Problems"] = result_df.apply(
    lambda row: extract_mapping(row["LLM_Response"], row["Patient_Problems"]),
    axis=1
)

result_df.to_csv("medication_problem_mapping.csv", index=False)

print("✅ Done. Outputs:")
print(" - prompt_template.txt")
print(" - individual prompt_*.txt files")
print(" - medication_problem_mapping.csv")
