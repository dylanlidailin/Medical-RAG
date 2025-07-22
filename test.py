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

print("OpenAI API Key loaded successfully.")

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
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import openai

@retry(
    wait=wait_exponential(multiplier=2, min=1, max=30),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(openai.RateLimitError)
)
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
from concurrent.futures import ThreadPoolExecutor, as_completed

def main():
    kb = load_knowledge_base("rxnorm_enriched_chunks.csv")
    df = pd.read_csv("chest_pain_patients.csv")
    df["Patient_ID"] = df.index  # 添加唯一标识

    # 准备所有患者输入
    patient_inputs = [row for _, row in df.iterrows()]

    summary_results = []

    def run_one(row):
        return process_patient(row, kb)

    # 多线程并发执行（5线程可调）
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_row = {executor.submit(run_one, row): row for row in patient_inputs}
        for future in as_completed(future_to_row):
            try:
                result = future.result()
                summary_results.append(result)
            except Exception as e:
                print("❌ 出错:", e)

    final_df = pd.DataFrame(summary_results)
    final_df.to_csv("medication_problem_mapping_summary.csv", index=False)
    print("✅ Done: medication_problem_mapping_summary.csv")


if __name__ == "__main__":
    main()

