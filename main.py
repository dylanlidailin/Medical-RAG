import os
import time
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variabless
load_dotenv()

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Missing OPENAI_API_KEY in environment variables.")
client = OpenAI(api_key=api_key)

# Step 1: Define mini-knowledge base with short, reliable drug descriptions
mini_kb = {
    "Atorvastatin": "Indicated for hyperlipidemia (high cholesterol).",
    "Ibuprofen": "Indicated for pain relief and inflammation (PRN).",
    "Levothyroxine": "Indicated for hypothyroidism (underactive thyroid).",
    "Losartan": "Indicated for hypertension (high blood pressure).",
    "Metformin": "Indicated for Type 2 Diabetes Mellitus.",
    "Metoprolol": "Indicated for hypertension (high blood pressure).",
    "Sertraline": "Indicated for depression and anxiety disorders.",
    "Gabapentin": "Indicated for neuropathic pain and seizure prophylaxis.",
    "Alendronate": "Indicated for osteoporosis (to support bone density).",
    "Furosemide": "Indicated for edema and fluid overload (e.g., in heart failure)."
}

# Step 2: Load patient dataset
df = pd.read_csv("chest_pain_patients.csv")

# Step 3: Helper to normalize medication names

def normalize_medication(med_name: str) -> str:
    """Return standardized medication key from raw outpatient meds entry."""
    return med_name.strip().split()[0].capitalize()

# Step 4: Prepare prompt template and save it to file
prompt_template = (
    "You are a clinical decision support assistant."
    " Use the medication information and patient's problem list to identify which problem(s) the medication treats."
    " If the medication is not in the knowledge base, reply 'I don’t know'.\n\n"
    "Medication: {Medication}\n"
    "Info: {Info}\n"
    "Patient Problems: {Patient_Problems}\n"
    "Which problem(s) from the list does this medication treat?"
)

with open("prompt_template.txt", "w") as tmpl_file:
    tmpl_file.write(prompt_template)

# Step 5: Generate prompts and save each to its own .txt file
prompts = []
for idx, row in df.iterrows():
    raw_meds = str(row.get("Outpatient_Medications", ""))
    patient_problems = [prob.strip() for prob in str(row.get("Past_Medical_History", "")).split(',') if prob.strip()]
    meds = [normalize_medication(m) for m in raw_meds.split(',')]

    for med in meds:
        info = mini_kb.get(med)
        # Format problems as a Python-style list string
        prob_list_str = str(patient_problems)
        prompt = prompt_template.format(
            Medication=med,
            Info=info or "Description not available.",
            Patient_Problems=prob_list_str
        )
        # Save individual prompt file
        filename = f"prompt_{idx}_{med}.txt"
        with open(filename, "w") as f:
            f.write(prompt)

        prompts.append({
            "Patient_ID": idx,
            "Medication": med,
            "Prompt_File": filename,
            "Prompt_Text": prompt,
            "Patient_Problems": patient_problems
        })

# Step 6: Query the LLM for each prompt
responses = []
def query_llm(prompt_text: str, model: str = "gpt-4", temperature: float = 0.0) -> str:
    """Call OpenAI chat completion with a given prompt."""
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a clinical decision support assistant."},
            {"role": "user", "content": prompt_text}
        ],
        temperature=temperature
    )
    return completion.choices[0].message.content.strip()

for entry in prompts:
    response = query_llm(entry["Prompt_Text"])
    responses.append(response)
    time.sleep(1)  # avoid rate limits

# Step 7: Map LLM responses back to structured problems

def extract_mapping(response: str, problems: list) -> list:
    """Extract list of treated problems based on LLM response matching."""
    if not response or response.lower().startswith("i don’t know"):
        return []
    treated = [prob for prob in problems if prob.lower() in response.lower()]
    return treated

# Append responses and mappings to DataFrame and save
result_df = pd.DataFrame(prompts)
result_df["LLM_Response"] = responses
result_df["Treated_Problems"] = result_df.apply(
    lambda r: extract_mapping(r["LLM_Response"], r["Patient_Problems"]), axis=1
)

result_df.to_csv("medication_problem_mapping.csv", index=False)
print("Done. Outputs:\n - prompt_template.txt\n - individual prompt_*.txt files\n - medication_problem_mapping.csv")
