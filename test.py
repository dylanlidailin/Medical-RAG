import os
import time
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# --- Step 0: Setup ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Missing OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# --- Step 1: Load Knowledge Base ---
def load_knowledge_base(filepath="rxnorm_enriched_chunks.csv") -> dict:
    df = pd.read_csv(filepath)
    kb = dict(zip(df["STR"].str.strip().str.capitalize(), df["Text_Chunk"]))
    return kb

# --- Step 2: Medication Extraction ---
def extract_medications(med_str: str) -> list:
    # Extract only the first word (assumed to be the drug name)
    return [m.strip().split()[0].capitalize() for m in str(med_str).split(",") if m.strip()]

# --- Step 3: Problem Extraction ---
def extract_problems(problem_str: str) -> list:
    return [p.strip() for p in str(problem_str).split(",") if p.strip()]

# --- Step 4: Knowledge Base Lookup ---
def lookup_description(med: str, kb: dict) -> str:
    return kb.get(med, "Description not available.")

# --- Step 5: Prompt Construction ---
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

# --- Step 6: Query OpenAI ---
def query_openai(prompt: str, model="gpt-4", temperature=0.0) -> str:
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

# --- Step 7: Match Problems ---
def match_problems(response: str, problems: list) -> list:
    if not response or response.lower().startswith("i don’t know"):
        return []
    return [p for p in problems if p.lower() in response.lower()]

# --- Step 8: Process Single Patient ---
def process_patient(row, kb_dict) -> dict:
    patient_id = row["Patient_ID"]
    med_str = row.get("Outpatient_Medications", "")
    prob_str = row.get("Past_Medical_History", "")

    meds = extract_medications(med_str)
    problems = extract_problems(prob_str)

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
        time.sleep(1)  # Avoid rate limit

    return result

# --- Step 9: Main Pipeline ---
def main():
    kb = load_knowledge_base("rxnorm_enriched_chunks.csv")
    df = pd.read_csv("chest_pain_patients.csv")
    df["Patient_ID"] = df.index

    summary_results = []

    for idx, row in df.iterrows():
        print(f"Processing patient {idx + 1}/{len(df)}...")
        result = process_patient(row, kb)
        summary_results.append(result)

    final_df = pd.DataFrame(summary_results)
    final_df.to_csv("medication_problem_mapping_summary.csv", index=False)
    print("✅ Output saved: medication_problem_mapping_summary.csv")

if __name__ == "__main__":
    main()