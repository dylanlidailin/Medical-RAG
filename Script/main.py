import os
import time
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# --- Step 0: Setup ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Missing OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

print("OpenAI API Key loaded successfully.")

# Run a simple query using OpenAI client to verify connection
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How to cure cancer?"}
    ],
    temperature=0.0
)
print("Connection to OpenAI API successful. Response:", response.choices[0].message.content)

# --- Step 1: 加载知识库为字典 ---
def load_knowledge_base(filepath="rxnorm_enriched_chunks.csv") -> dict:
    df = pd.read_csv(filepath)
    kb = dict(zip(df["STR"].str.capitalize(), df["Text_Chunk"]))
    return kb

print("Knowledge base loaded successfully.")

# --- Step 2: 提取患者药物 ---
def extract_medications(med_str: str) -> list:
    meds = [m.strip().split()[0].capitalize() for m in str(med_str).split(",") if m.strip()]
    return meds

print("Medication extraction function ready.")

# --- Step 3: 提取患者问题列表 ---
def extract_problems(problem_str: str) -> list:
    return [p.strip() for p in str(problem_str).split(",") if p.strip()]

print("Problem extraction function ready.")

# --- Step 4: 从知识库获取药品描述 ---
def lookup_description(med: str, kb: dict) -> str:
    return kb.get(med, "Description not available.")

print("Knowledge base lookup function ready.")

# --- Step 5: 构造 prompt ---
def build_prompt(med: str, info: str, problems: list) -> str:
    template = (
        "You are a clinical decision support assistant.\n"
        "Use the medication information and patient's problem list to identify which problem(s) the medication treats.\n"
        "If the medication is not in the knowledge base, reply 'I don’t know'.\n\n"
        f"Medication: {med}\n"
        f"Info: {info}\n"
        f"Patient Problems: {problems}\n"
        "Which problem(s) from the list does this medication treat?"
    )
    return template

print("Prompt construction function ready.")

# --- Step 6: 调用 OpenAI API ---
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

print("OpenAI API query function ready.")

# --- Step 7: 从 LLM 响应中提取被治疗的问题 ---
def match_problems(response: str, problems: list) -> list:
    if not response or response.lower().startswith("i don’t know"):
        return []
    return [p for p in problems if p.lower() in response.lower()]

print("Problem matching function ready.")

# --- Step 8: 处理一个病人 ---
def process_patient(row, kb_dict) -> dict:
    patient_id = row["Patient_ID"]
    meds = extract_medications(row["Outpatient_Medications"])
    problems = extract_problems(row["Past_Medical_History"])

    result = {
        "Patient_ID": patient_id,
        "Medications": meds,
        "Treated_Problems_by_Medication": {}
    }

    for med in meds:
        info = lookup_description(med, kb_dict)
        prompt = build_prompt(med, info, problems)
        response = query_openai(prompt)
        matched = match_problems(response, problems)
        result["Treated_Problems_by_Medication"][med] = matched
        time.sleep(1)  # avoid rate limit

    return result

print("Patient processing function ready.")

# --- Step 9: 主入口 ---

def main():
    kb = load_knowledge_base("rxnorm_enriched_chunks.csv")
    df = pd.read_csv("chest_pain_patients.csv")
    df["Patient_ID"] = df.index  # 添加唯一标识

    summary_results = []

    for _, row in tqdm(df.head(5).iterrows(), total=5, desc="Processing patients"):
        result = process_patient(row, kb)
        summary_results.append(result)

    final_df = pd.DataFrame(summary_results)
    final_df.to_csv("Mapping/medication_problem_main_mapping_summary.csv", index=False)

if __name__ == "__main__":
    main()
