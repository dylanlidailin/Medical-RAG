import os
import time
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI

# --- Step 0: Setup ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ Missing OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# --- Step 1: API Health Check ---
def check_openai_api():
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello!"}
            ],
            temperature=0
        )
        print("✅ OpenAI API is available.")
        return True
    except Exception as e:
        print("❌ API test failed:", str(e))
        return False

if not check_openai_api():
    exit("🚫 Exiting due to OpenAI API error.")

# --- Step 2: Load Knowledge Base ---
def load_knowledge_base(filepath="rxnorm_enriched_chunks.csv") -> dict:
    df = pd.read_csv(filepath)
    return dict(zip(df["STR"].str.strip().str.capitalize(), df["Text_Chunk"]))

# --- Step 3: Medication & Problem Extraction ---
def extract_medications(med_str: str) -> list:
    return [m.strip().split()[0].capitalize() for m in str(med_str).split(",") if m.strip()]

def extract_problems(problem_str: str) -> list:
    return [p.strip() for p in str(problem_str).split(",") if p.strip()]

# --- Step 4: Prompt Builder ---
def build_prompt(med: str, info: str, problems: list) -> str:
    return (
        "You are a clinical decision support assistant.\n"
        "Use the medication information and patient's problem list to identify which problem(s) the medication treats.\n"
        "If the medication is not in the knowledge base, reply 'I don’t know'.\n\n"
        f"Medication: {med}\n"
        f"Info: {info}\n"
        f"Patient Problems: {problems}\n"
        "Which problem(s) from the list does this medication treat?"
    )

# --- Step 5: OpenAI Query ---
def query_openai(prompt: str, model="gpt-3.5-turbo", temperature=0.0) -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a clinical decision support assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# --- Step 6: Problem Matching ---
def match_problems(response: str, problems: list) -> list:
    if not response or response.lower().startswith("i don’t know"):
        return []
    return [p for p in problems if p.lower() in response.lower()]

# --- Step 7: Process One Patient ---
def process_patient(row, kb_dict) -> dict:
    patient_id = row["Patient_ID"]
    meds = extract_medications(row.get("Outpatient_Medications", ""))
    problems = extract_problems(row.get("Past_Medical_History", ""))

    result = {
        "Patient_ID": patient_id,
        "Medications": meds,
        "Treated_Problems_by_Medication": {}
    }

    if not meds or not problems:
        return result

    for med in meds:
        info = lookup_description(med, kb_dict)
        prompt = build_prompt(med, info, problems)
        response = query_openai(prompt)
        matched = match_problems(response, problems)
        result["Treated_Problems_by_Medication"][med] = matched
        time.sleep(0.5)  # Reduced wait to speed up

    return result

# --- Helper for description ---
def lookup_description(med: str, kb: dict) -> str:
    return kb.get(med, "Description not available.")

# --- Step 8: Main Pipeline ---
def main():
    kb = load_knowledge_base("rxnorm_enriched_chunks.csv")
    df = pd.read_csv("chest_pain_patients.csv")
    df["Patient_ID"] = df.index

    results = []
    for _, row in tqdm(df.head(5).iterrows(), total=5, desc="Processing patients"):
        result = process_patient(row, kb)
        results.append(result)

    final_df = pd.DataFrame(results)
    final_df.to_csv("medication_problem_mapping_summary.csv", index=False)
    print("✅ Output saved: medication_problem_mapping_summary.csv")

if __name__ == "__main__":
    main()
