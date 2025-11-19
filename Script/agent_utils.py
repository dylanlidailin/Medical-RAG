# agent_utils.py
import os
import json
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
# Re-adding the Gemini API import
import google.generativeai as genai 
from dotenv import load_dotenv
from fhirclient import client
from fhirclient.models import medication

# --- KEEP Retriever Import ---
from .retriever_setup import retrieve_context 

load_dotenv()

# --- Setup for Gemini and FHIR Clients ---
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
except Exception as e:
    print(f"Error configuring Gemini API. Make sure GOOGLE_API_KEY is set in your .env file. {e}")

fhir_settings = {
    'app_id': 'my_medical_app',
    'api_base': 'https://hapi.fhir.org/baseR4'
}
fhir_client = client.FHIRClient(settings=fhir_settings)

# --- Retrieve FHIR Medication by Name ---
def get_fhir_medication(med_name: str) -> dict:
    """
    Retrieves medication info from a FHIR server. 
    """
    try:
        search_params = {'code:text': med_name}
        response_bundle = medication.Medication.where(struct=search_params).perform(fhir_client.server)
        bundle = response_bundle.as_json()

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
    return {"name": med_name, "code": None, "indication": f"No standard information found for {med_name}."}


# --- Prompt Construction (RAG-Enabled) ---
# This is the RAG-enabled function using the local retriever
def build_agent_prompt(med_name: str, med_indication: str, problems: list) -> str:
    """
    Builds the RAG prompt by retrieving context related to the medication
    and inserting it into the prompt for the Gemini LLM.
    """
    problem_list_str = ", ".join(f'"{p}"' for p in problems)
    
    # --- RAG: Retrieval Step (Uses PubMedBERT/Faiss) ---
    retrieval_query = f"Clinical use and comorbidities for {med_name} and its indication {med_indication}"
    retrieved_context = retrieve_context(retrieval_query, top_k=2)
    context_str = "\n".join([f"- {c}" for c in retrieved_context])
    
    return (
        "You are an expert clinical decision support assistant. You MUST follow all instructions exactly and only output a single JSON object.\n\n"
        
        f"**CRITICAL CONTEXT (GROUND TRUTH):**\n{context_str}\n\n" # <-- Inserted Retrieved Context
        f"Medication to Analyze: {med_name}\n"
        f"Patient's Problem List: [{problem_list_str}]\n\n"
        "Your task is to use the **CRITICAL CONTEXT** provided above to determine the clinical relationship between the Medication and the Patient's Problem List. You MUST prioritize the retrieved context over your general knowledge.\n"
        "Populate three keys:\n"
        "1. 'primary_indication': A string confirming the main condition this medication treats, based on the **CRITICAL CONTEXT**.\n"
        "2. 'direct_treatment': A list of problems *from the patient's list* that this medication is known to treat, based on the **CRITICAL CONTEXT**.\n"
        "3. 'related_conditions': A list of problems *from the patient's list* that are common comorbidities with the 'primary_indication', but are NOT treated by the medication, based on the **CRITICAL CONTEXT**.\n\n"
        
        "**CRITICAL RULES (READ CAREFULLY):**\n"
        "1. Only link a problem if the association is **explicitly supported** by the **CRITICAL CONTEXT** or the provided 'Known Primary Indication'.\n"
        "2. If no problem in the patient's list is clearly linked, return an empty list `[]` for that key.\n"
        "3. **DO NOT GUESS OR INVENT LINKS.** If the context is missing, be conservative.\n"

        'Example: {"primary_indication": "Hypertension", "direct_treatment": ["Hypertension"], "related_conditions": ["Chronic Kidney Disease"]}'
    )

# --- Call Gemini with Retry Strategy (Reverted to API) ---
def query_with_retry(prompt: str, model="gemini-2.5-flash-lite") -> dict:
    """
    Queries the Gemini API and returns the structured dictionary.
    """
    try:
        generation_config = {
            "temperature": 0.0,
            "response_mime_type": "application/json",
        }
        
        gemini_model = genai.GenerativeModel(
            model_name=model, 
            generation_config=generation_config,
            system_instruction="You are a clinical decision support assistant that outputs JSON."
        )

        response = gemini_model.generate_content(prompt)
        
        return json.loads(response.text)
    
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return {"primary_indication": "Error", "direct_treatment": [], "related_conditions": []}

# --- Fuzzy Match Output ---
def fuzzy_match_problems(response_problems: list, patient_problems: list, threshold=80) -> list:
    """
    Performs fuzzy matching between problems found by the LLM and the patient's list.
    """
    matched = []
    for p_patient in patient_problems:
        for p_response in response_problems:
            if fuzz.partial_ratio(p_patient.lower(), p_response.strip().lower()) >= threshold:
                matched.append(p_patient)
                break
    return matched