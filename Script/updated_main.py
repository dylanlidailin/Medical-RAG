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
def load_patients(path="chest_pain_patients.csv"):
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
        
        # The agent returns a dictionary with its findings
        agent_response = query_with_retry(prompt)
        
        # Extract the list of problems the agent found in the patient's list
        response_problems = agent_response.get("treated_problems_from_list", [])

        # Fuzzy match to confirm the agent's findings are valid
        matched_problems = fuzzy_match_problems(response_problems, problems)
        
        # --- NEW LOGIC ---
        if matched_problems:
            # If we found a valid match in the patient's existing list, use it.
            result["Treated_Problems_by_Medication"][med] = matched_problems
        else:
            # If no match was found, the patient's record is likely incomplete.
            # We will trust the agent's identified primary indication.
            primary_indication = agent_response.get('primary_indication', 'Unknown')
            
            # Add the agent's finding as the treated problem, unless it's an error.
            if primary_indication not in ['Unknown', 'Error', 'None']:
                result["Treated_Problems_by_Medication"][med] = [primary_indication]
            else:
                # If the agent couldn't determine an indication, leave it empty.
                result["Treated_Problems_by_Medication"][med] = []

    return result

# --- Step 4: Main Execution ---
def main():
    df = load_patients()
    df = df.head(10)  # Process the first 10 patients

    results = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing patients"):
        results.append(process_patient(row))

    # Ensure the output directory exists
    os.makedirs("Mapping", exist_ok=True)
    pd.DataFrame(results).to_csv("Mapping/medication_problem_agent_mapping_summary.csv", index=False)
    print("\nProcessing complete. Output saved to Mapping/medication_problem_agent_mapping_summary.csv")

if __name__ == "__main__":
    main()