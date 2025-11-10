import os
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from agent_utils import (
    build_agent_prompt,
    query_with_retry,
    fuzzy_match_problems,
    get_fhir_medication,
)

# --- Step 0: Setup ---
load_dotenv()

# --- Step 1: Load Data ---
def load_patients(path="chest_pain_patients_test_problems_only.csv"):
    # Using 'engine=python' for robust CSV parsing
    df = pd.read_csv(path, engine='python')
    df["Patient_ID"] = df.index
    return df

# --- Step 2: Extract Info from DataFrame Rows ---
def extract_medications(med_str):
    return [m.strip().capitalize() for m in str(med_str).split(",") if m.strip()]

def extract_problems(problem_str):
    return [p.strip() for p in str(problem_str).split(",") if p.strip()]

# --- Step 3: Process One Patient ---
def process_patient(row):
    pid = row["Patient_ID"]
    meds = extract_medications(row.get("Outpatient_Medications", ""))
    problems = extract_problems(row.get("Past_Medical_History", ""))
    result = {"Patient_ID": pid, "Medications": meds, "Treated_Problems_by_Medication": {}}

    for med in meds:
        med_info = get_fhir_medication(med)
        indication = med_info.get("indication", "No indication found.")
        standardized_med_name = med_info.get("name", med)

        prompt = build_agent_prompt(standardized_med_name, indication, problems)
        
        agent_response = query_with_retry(prompt)
        
        # Extract the detailed lists from the agent's response
        direct_treatments = agent_response.get("direct_treatment", [])
        related_conditions = agent_response.get("related_conditions", [])
        
        # Fuzzy match to ensure the agent's findings are valid and from the list
        matched_direct = fuzzy_match_problems(direct_treatments, problems)
        matched_related = fuzzy_match_problems(related_conditions, problems)

        # --- CORRECTED LOGIC TO BUILD THE FINAL MAPPING ---
        final_mapping = []
        if matched_direct:
            # CORRECTED: The list comprehension is now inside the extend() method
            final_mapping.extend([f"{p} (Direct)" for p in matched_direct])
        
        if matched_related:
            # CORRECTED: The list comprehension is now inside the extend() method
            final_mapping.extend([f"{p} (Related)" for p in matched_related])

        # If after all that, no links were found, fall back to the primary indication
        if not final_mapping:
            primary_indication = agent_response.get('primary_indication', 'Unknown')
            if primary_indication not in ['Unknown', 'Error', 'None']:
                final_mapping.append(f"{primary_indication} (Inferred)")
        
        result["Treated_Problems_by_Medication"][med] = final_mapping if final_mapping else []

    return result

# --- Step 4: Main Execution ---
def main():
    df = load_patients()
    df = df.head(5) # For testing, limit to first 5 patients

    results = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing patients"):
        results.append(process_patient(row))

    # Ensure the output directory exists
    os.makedirs("Mapping", exist_ok=True)
    pd.DataFrame(results).to_csv("Mapping/Validation_testing.csv", index=False)

if __name__ == "__main__":
    main()