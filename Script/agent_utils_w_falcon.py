import os
import json
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from dotenv import load_dotenv
from fhirclient import client
from fhirclient.models import medication

# --- NEW IMPORTS for Falcon (Generator) ---
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

# --- NEW IMPORT for Retriever Integration ---
from .retriever_setup import retrieve_context

load_dotenv()

# --- Setup for FHIR Client (Unchanged) ---
# ... (FHIR client setup remains the same)
fhir_settings = {
    'app_id': 'my_medical_app',
    'api_base': 'https://hapi.fhir.org/baseR4'
}
fhir_client = client.FHIRClient(settings=fhir_settings)

# --- Setup for Falcon Local Inference (Generator) ---
# This block ensures the Falcon model is loaded only once when the script starts
llm_pipeline = None # Initialize globally
llm_tokenizer = None # Initialize globally
try:
    # Model configuration from previous discussion
    MODEL_NAME = "tiiuae/falcon-7b-instruct" 
    # Determine device: 0 for GPU, -1 for CPU (if no CUDA available)
    DEVICE = 0 if torch.cuda.is_available() else -1
    
    # Initialize the tokenizer, model, and pipeline once
    llm_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    llm_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16, # Use bfloat16 for modern GPUs
        device_map="auto" # Distribute model across available resources
    )
    llm_pipeline = pipeline(
        "text-generation",
        model=llm_model,
        tokenizer=llm_tokenizer,
        device=DEVICE,
        max_new_tokens=512, 
        do_sample=False,
        eos_token_id=llm_tokenizer.eos_token_id
    )
    print(f"Loaded Falcon model {MODEL_NAME} for local inference on device: {DEVICE}")

except Exception as e:
    print(f"Error setting up local LLM inference. Ensure 'transformers', 'torch', and 'accelerate' are installed. Fallback to None. {e}")


# --- Retrieve FHIR Medication by Name (Unchanged) ---
def get_fhir_medication(med_name: str) -> dict:
    """
    Retrieves medication info from a FHIR server. (Unchanged for brevity)
    """
    # ... (function body remains the same as your previous code)
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
# THIS FUNCTION IS UPDATED to include the retrieval step
def build_agent_prompt(med_name: str, med_indication: str, problems: list) -> str:
    """
    Builds the RAG prompt by retrieving context related to the medication
    and inserting it into the prompt for the Falcon LLM.
    """
    problem_list_str = ", ".join(f'"{p}"' for p in problems)
    
    # --- RAG: Retrieval Step (Uses PubMedBERT/Faiss) ---
    retrieval_query = f"Clinical use and comorbidities for {med_name} and its indication {med_indication}"
    # Use the function imported from retriever_setup.py
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


# --- Call Local LLM with Retry Strategy ---
# THIS FUNCTION IS REPLACED to use the local Falcon pipeline
def query_with_retry(prompt: str) -> dict:
    """
    Queries the local Falcon LLM pipeline and returns the structured dictionary.
    """
    # Check if the model failed to load during setup
    if llm_pipeline is None or llm_tokenizer is None:
        print("Local Falcon pipeline not loaded. Returning fallback error.")
        return {"primary_indication": "Error", "direct_treatment": [], "related_conditions": []}

    try:
        # 1. Format the prompt using a chat template (required for instruct models like Falcon)
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # Apply the chat template to turn the list of dicts into a single prompt string
        prompt_text = llm_tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 2. Generate the content using the pipeline
        output = llm_pipeline(prompt_text, num_return_sequences=1)
        
        # 3. Extract the generated text and remove the prompt repetition
        generated_text = output[0]['generated_text']
        response_text = generated_text.replace(prompt_text, "", 1).strip()
        
        # 4. Find the JSON object in the response text and parse it
        json_start = response_text.find('{')
        json_end = response_text.rfind('}')
        
        if json_start != -1 and json_end != -1 and json_end > json_start:
            json_str = response_text[json_start : json_end + 1]
            return json.loads(json_str)
        else:
            print(f"Could not find or parse JSON from response: {response_text}")
            return {"primary_indication": "Parse Error", "direct_treatment": [], "related_conditions": []}

    except Exception as e:
        print(f"Local LLM Generation Error: {e}")
        return {"primary_indication": "Error", "direct_treatment": [], "related_conditions": []}

# --- Fuzzy Match Output (Unchanged) ---
def fuzzy_match_problems(response_problems: list, patient_problems: list, threshold=80) -> list:
    """
    (This function is unchanged for brevity)
    """
    matched = []
    for p_patient in patient_problems:
        for p_response in response_problems:
            if fuzz.partial_ratio(p_patient.lower(), p_response.strip().lower()) >= threshold:
                matched.append(p_patient)
                break
    return matched