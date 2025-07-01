import openai
import time
import pandas as pd

# Set OpenAI API key
openai.api_key = "OPENAI_API_KEY"

# Import mini-knowledge base
mini_kb = {
    "Atorvastatin": "Used to treat hyperlipidemia (high cholesterol).",
    "Ibuprofen": "Used for pain relief and as an anti-inflammatory, often on an as-needed (PRN) basis.",
    "Levothyroxine": "Treats hypothyroidism (underactive thyroid).",
    "Losartan": "Prescribed for hypertension (high blood pressure).",
    "Metformin": "Treats Type 2 Diabetes Mellitus.",
    "Metoprolol": "Used to manage hypertension (high blood pressure).",
    "Sertraline": "Treats depression and anxiety disorders.",
    "Gabapentin": "Used for neuropathic pain and seizure prophylaxis.",
    "Alendronate": "Supports bone density in patients with osteoporosis.",
    "Furosemide": "Used to treat fluid overload or edema, such as in heart failure."
}

# Load patients data
df = pd.read_csv("chest_pain_patients - chest_pain_patients.csv")

# Normalize medication names (Part of Data Preprocessing)
def normalize_medication(med):
    return med.strip().split()[0].capitalize()

# Generate prompts using mini-KB and patient data (Retrieval & Prompting)
prompts = []
for index, row in df.iterrows():
    # Extracting patient medications and problems
    medications = [normalize_medication(med) for med in str(row['Outpatient_Medications']).split(',')]
    problems = [prob.strip() for prob in str(row['Past_Medical_History']).split(',') if prob.strip()]
    
    # Looping over medications to create prompts
    for med in medications:
        if med in mini_kb:
            info = mini_kb[med]
            prompt = f"""You are a clinical decision support assistant.
Answer “I don’t know” if the question asked is not recorded in the mini-knowledge base.

Medication: {med}
Info: {info}
Patient Problems: {problems}
Which problem(s) from the list does this medication treat?"""
            prompts.append({
                "Patient_ID": index,
                "Medication": med,
                "Prompt": prompt
            })

# Convert to DataFrame for easier inspection (Structured Output)
prompt_df = pd.DataFrame(prompts)

# Query using LLM
def query_llm(prompt, model="gpt-4", temperature=0):
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a clinical decision support assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error: {e}"

# Add LLM response column to the prompt DataFrame
responses = []
for i, row in prompt_df.iterrows():
    response = query_llm(row["Prompt"])
    responses.append(response)
    time.sleep(1)  # prevent rate limiting (adjust based on your OpenAI plan)

prompt_df["LLM_Response"] = responses

# Save and export the DataFrame with LLM responses
prompt_df.to_csv("prompts_with_responses.csv", index=False)
print("LLM responses added and saved to prompts_with_responses.csv")