# main.py
import os
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from agent_utils import (
    build_faiss_index,
    retrieve_top_k,
    build_agent_prompt,
    query_with_retry,
    fuzzy_match_problems,
)

# --- Step 0: Setup ---
load_dotenv()

# --- Step 1: Load Data ---
def load_knowledge_base(path="rxnorm_enriched_chunks.csv"):
    df = pd.read_csv(path)
    return df["STR"].tolist(), dict(zip(df["STR"].str.strip().str.capitalize(), df["Text_Chunk"]))

def load_patients(path="chest_pain_patients.csv"):
    df = pd.read_csv(path)
    df["Patient_ID"] = df.index
    return df

# --- Step 2: Extract Info ---
def extract_medications(med_str):
    return [m.strip().split()[0].capitalize() for m in str(med_str).split(",") if m.strip()]

def extract_problems(problem_str):
    return [p.strip() for p in str(problem_str).split(",") if p.strip()]

# --- Step 3: Process One Patient ---
def process_patient(row, index, term_list, kb):
    pid = row["Patient_ID"]
    meds = extract_medications(row.get("Outpatient_Medications", ""))
    problems = extract_problems(row.get("Past_Medical_History", ""))
    result = {"Patient_ID": pid, "Medications": meds, "Treated_Problems_by_Medication": {}}

    for med in meds:
        best_match = retrieve_top_k(med, index, term_list, None)
        description = kb.get(best_match, "Description not available.")
        prompt = build_agent_prompt(med, description, problems)
        response = query_with_retry(prompt)
        matched_problems = fuzzy_match_problems(response, problems)
        result["Treated_Problems_by_Medication"][med] = matched_problems

    return result

# --- Step 4: Main ---
def main():
    terms, kb_dict = load_knowledge_base()
    index, term_list, _ = build_faiss_index(terms)
    df = load_patients()

    results = []
    for _, row in tqdm(df.iterrows(), desc="Processing patients"):
        results.append(process_patient(row, index, term_list, kb_dict))

    pd.DataFrame(results).to_csv("Mapping/medication_problem_agent_mapping_summary.csv", index=False)

if __name__ == "__main__":
    main()

#Processing patients: 100it [48:47, 29.28s/it]