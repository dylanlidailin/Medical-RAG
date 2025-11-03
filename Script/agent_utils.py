# agent_utils.py
import os
import json
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from openai import OpenAI
from dotenv import load_dotenv
from fhirclient import client
from fhirclient.models import medication

load_dotenv()

# --- Setup for OpenAI and FHIR Clients ---
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

fhir_settings = {
    'app_id': 'my_medical_app',
    'api_base': 'https://hapi.fhir.org/baseR4'
}
fhir_client = client.FHIRClient(settings=fhir_settings)

# --- Retrieve FHIR Medication by Name ---
def get_fhir_medication(med_name: str) -> dict:
    """
    Searches the FHIR server for a medication by its name.
    This version converts the response Bundle object to a JSON dictionary
    to be resilient to malformed server data.
    """
    try:
        search_params = {'code:text': med_name}
        # .perform() returns a Bundle object, not a dict
        response_bundle = medication.Medication.where(struct=search_params).perform(fhir_client.server)

        # --- THIS IS THE FIX ---
        # Convert the Bundle object to a dictionary to safely parse it
        bundle = response_bundle.as_json()

        # Manually and safely navigate the JSON structure
        if bundle and bundle.get('entry') and len(bundle['entry']) > 0:
            resource = bundle['entry'][0].get('resource', {})
            
            code_concept = resource.get('code', {})
            name_text = code_concept.get('text', med_name)
            
            coding_list = code_concept.get('coding', [])
            code_val = coding_list[0].get('code') if coding_list else None

            return {
                "name": name_text,
                "code": code_val,
                "indication": name_text
            }
    except Exception as e:
        print(f"FHIR API Error for '{med_name}': {e}")

    # Fallback if search fails or returns no results
    return {"name": med_name, "code": None, "indication": f"No standard information found for {med_name}."}

# --- Prompt Construction ---
def build_agent_prompt(med_name: str, med_indication: str, problems: list) -> str:
    problem_list_str = ", ".join(f'"{p}"' for p in problems)
    
    return (
        "You are an expert clinical decision support assistant. You MUST follow all instructions exactly and only output a single JSON object.\n\n"
        f"Medication: {med_name}\n"
        f"Known Primary Indication: {med_indication}\n"
        f"Patient's Problem List: [{problem_list_str}]\n\n"
        "Your task is to populate three keys:\n"
        "1. 'primary_indication': A string confirming the main condition this medication treats (e.g., 'Hypertension').\n"
        "2. 'direct_treatment': A list of problems *from the patient's list* that this medication is known to treat.\n"
        "3. 'related_conditions': A list of problems *from the patient's list* that are common comorbidities with the 'primary_indication', but are NOT treated by the medication.\n\n"
        
        # --- THIS IS THE STRONGER FIX ---
        "**CRITICAL RULES (READ CAREFULLY):**\n"
        "1. If no problem in the patient's list is a **clear, established direct treatment** for the medication, 'direct_treatment' **MUST** be an empty list `[]`.\n"
        "2. If no problem in the patient's list is a **common, known comorbidity** of the 'primary_indication', 'related_conditions' **MUST** be an empty list `[]`.\n"
        "3. **DO NOT GUESS OR INVENT LINKS.** If you are not 100% certain of a clinical association, you **MUST** return an empty list for that key. It is **critical** that you do not provide false information.\n"
        # --- END OF FIX ---

        'Example (Good Match): {"primary_indication": "Hypertension", "direct_treatment": ["Anxiety"], "related_conditions": ["Chronic Kidney Disease"]}\n'
        'Example (No Match / Required): {"primary_indication": "Hypertension", "direct_treatment": [], "related_conditions": []}'
    )

# --- Call OpenAI with Retry Strategy ---
def query_with_retry(prompt: str, model="gpt-4-1106-preview") -> dict: # <-- Changed return type
    """
    Queries the OpenAI API and returns the structured dictionary.
    """
    try:
        response = openai_client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a clinical decision support assistant that outputs JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        return json.loads(response.choices[0].message.content) # <-- Return the whole object
    except Exception as e:
        print(f"OpenAI API Error: {e}")
        # Return a default structure on error
        return {"primary_indication": "Error", "treated_problems_from_list": []}

# --- Fuzzy Match Output ---
def fuzzy_match_problems(response_problems: list, patient_problems: list, threshold=80) -> list:
    """
    Compares problems returned by the AI against the patient's actual problem list.
    """
    matched = []
    for p_patient in patient_problems:
        for p_response in response_problems:
            if fuzz.partial_ratio(p_patient.lower(), p_response.strip().lower()) >= threshold:
                matched.append(p_patient)
                break
    return matched