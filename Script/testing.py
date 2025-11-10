import json
import os
import pandas as pd
from dotenv import load_dotenv
from agent_utils import (
    build_agent_prompt,
    query_with_retry,
    fuzzy_match_problems,
    get_fhir_medication,
)

# --- Load .env variables (for GOOGLE_API_KEY) ---
load_dotenv()

# --- Re-define core logic from your main script for testability ---

def process_patient_logic(med_name: str, problems_list: list) -> str:
    """
    A testable version of your process_patient logic.
    It takes a single medication name and a list of patient problems,
    runs the full RAG pipeline, and returns the final mapping list
    as a JSON string for easy comparison.
    """
    # 1. Get info from FHIR (external knowledge)
    med_info = get_fhir_medication(med_name)
    indication = med_info.get("indication", "No indication found.")
    standardized_med_name = med_info.get("name", med_name)

    # 2. Build the prompt for the LLM
    prompt = build_agent_prompt(standardized_med_name, indication, problems_list)
    
    # 3. Query the LLM
    agent_response = query_with_retry(prompt)
    
    # 4. Extract LLM findings
    direct_treatments = agent_response.get("direct_treatment", [])
    related_conditions = agent_response.get("related_conditions", [])
    
    # 5. Apply business logic: Fuzzy match against the original list
    matched_direct = fuzzy_match_problems(direct_treatments, problems_list)
    matched_related = fuzzy_match_problems(related_conditions, problems_list)

    # 6. Apply business logic: Build final mapping
    final_mapping = []
    if matched_direct:
        final_mapping.extend([f"{p} (Direct)" for p in matched_direct])
    
    if matched_related:
        final_mapping.extend([f"{p} (Related)" for p in matched_related])

    # 7. Apply business logic: Fallback to inferred indication
    if not final_mapping:
        primary_indication = agent_response.get('primary_indication', 'Unknown')
        # Check against exclusion list
        if primary_indication not in ['Unknown', 'Error', 'None']:
            final_mapping.append(f"{primary_indication} (Inferred)")
    
    # Return as a string to match the JSON answer options
    return json.dumps(final_mapping if final_mapping else [])

def test_post_processing_logic(agent_response: dict, patient_problems: list) -> str:
    """
    Tests only the post-processing and fallback logic (for Q8).
    It skips the FHIR and LLM calls and uses a mock agent_response.
    """
    direct_treatments = agent_response.get("direct_treatment", [])
    related_conditions = agent_response.get("related_conditions", [])
    
    matched_direct = fuzzy_match_problems(direct_treatments, patient_problems)
    matched_related = fuzzy_match_problems(related_conditions, patient_problems)

    final_mapping = []
    if matched_direct:
        final_mapping.extend([f"{p} (Direct)" for p in matched_direct])
    
    if matched_related:
        final_mapping.extend([f"{p} (Related)" for p in matched_related])

    if not final_mapping:
        primary_indication = agent_response.get('primary_indication', 'Unknown')
        if primary_indication not in ['Unknown', 'Error', 'None']:
            final_mapping.append(f"{primary_indication} (Inferred)")
            
    return json.dumps(final_mapping if final_mapping else [])


def test_general_knowledge(question_obj: dict) -> (bool, str, str):
    """
    Tests the LLM's general knowledge by formatting the quiz question
    into a prompt and asking the LLM to choose the answer.
    """
    question = question_obj['question']
    options = [opt['text'] for opt in question_obj['answerOptions']]
    
    # Construct a prompt to force a JSON choice
    prompt = f"""
    You are an expert medical knowledge evaluator. Answer the following multiple-choice question.
    You MUST output a single JSON object with one key: "selected_answer".
    The "selected_answer" MUST be one of the exact strings from the "options" list.

    Question: {question}
    Options: {json.dumps(options)}

    Example Output:
    {{"selected_answer": "Lisinopril"}}
    """
    
    try:
        # Use the same query_with_retry function from agent_utils
        response_json = query_with_retry(prompt)
        model_answer_text = response_json.get("selected_answer", "")
    except Exception as e:
        print(f"  [!] Error parsing LLM response for Q{question_obj['questionNumber']}: {e}")
        model_answer_text = "[Error Parsing Response]"

    correct_answer_text = ""
    for opt in question_obj['answerOptions']:
        if opt['isCorrect']:
            correct_answer_text = opt['text']
            break
            
    is_correct = (model_answer_text == correct_answer_text)
    return is_correct, model_answer_text, correct_answer_text

# --- Manually defined inputs for pipeline-specific questions ---
# This is necessary because the inputs (med_name, problem_list) 
# are described in the question text, not as data.
pipeline_test_cases = {
    5: {
        "type": "full_pipeline",
        "med": "Lisinopril",
        "problems": ["Hypertension", "Gout"]
    },
    6: {
        "type": "full_pipeline",
        "med": "Atorvastatin",
        "problems": ["Anxiety", "Asthma"]
    },
    7: {
        "type": "full_pipeline",
        "med": "Metformin",
        "problems": ["Chronic Kidney Disease", "Hypertension"]
    },
    8: {
        "type": "post_processing",
        "agent_response": {
            "primary_indication": "Depression",
            "direct_treatment": ["Depression"],
            "related_conditions": ["Anxiety"]
        },
        "problems": ["Depression"]
    },
    9: {
        "type": "full_pipeline",
        "med": "Metformin",
        "problems": ["High Blood Pressure", "Diabetes II"]
    },
    10: {
        "type": "full_pipeline",
        "med": "FakeDrug123",
        "problems": ["Hypertension"]
    }
}


def main():
    quiz_file = "test/clinical_quiz_questions.json"
    try:
        with open(quiz_file, 'r') as f:
            quiz_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {quiz_file}. Make sure it is in the same directory.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not parse {quiz_file}. Check for JSON syntax errors.")
        return

    questions = quiz_data.get("questions", [])
    if not questions:
        print("No questions found in the quiz file.")
        return

    print(f"--- Starting Clinical RAG Model Evaluation ---")
    print(f"Loaded {len(questions)} questions from {quiz_file}.\n")

    score = 0
    results_log = []

    for q in questions:
        q_num = q['questionNumber']
        q_text = q['question']
        print(f"--- Question {q_num}: {q_text} ---")

        correct_answer_text = ""
        for opt in q['answerOptions']:
            if opt['isCorrect']:
                correct_answer_text = opt['text']
                break
        
        model_answer_text = ""
        is_correct = False

        if q_num <= 4:
            # General Knowledge Test
            print("  [Test Type: General LLM Knowledge]")
            is_correct, model_answer_text, _ = test_general_knowledge(q)
        
        elif q_num in pipeline_test_cases:
            case = pipeline_test_cases[q_num]
            
            try:
                if case['type'] == 'full_pipeline':
                    print(f"  [Test Type: Full RAG Pipeline]")
                    print(f"  [Input: Med='{case['med']}', Problems={case['problems']}]")
                    model_answer_text = process_patient_logic(case['med'], case['problems'])
                
                elif case['type'] == 'post_processing':
                    print(f"  [Test Type: Post-Processing Logic]")
                    print(f"  [Input: Problems={case['problems']}, Mock Response={case['agent_response']}]")
                    model_answer_text = test_post_processing_logic(case['agent_response'], case['problems'])
                
                # Compare the stringified JSON output
                is_correct = (model_answer_text == correct_answer_text)

            except Exception as e:
                print(f"  [!] Pipeline error during Q{q_num}: {e}")
                model_answer_text = f"[Pipeline Error: {e}]"
                is_correct = False
        
        else:
            print(f"  [!] No test case defined for Q{q_num}. Skipping.")
            continue

        if is_correct:
            score += 1
            print(f"  [PASS] Correct!")
            print(f"   -> Model Output:   {model_answer_text}")
        else:
            print(f"  [FAIL] Incorrect.")
            print(f"   -> Model Output:   {model_answer_text}")
            print(f"   -> Correct Answer: {correct_answer_text}")

        results_log.append({
            "Question": q_num,
            "Result": "PASS" if is_correct else "FAIL",
            "Model_Answer": model_answer_text,
            "Correct_Answer": correct_answer_text,
            "Rationale (for correct answer)": q['answerOptions'][ [opt['text'] for opt in q['answerOptions']].index(correct_answer_text) ]['rationale']
        })
        print("-" * 50)

    # --- Final Report ---
    total = len(questions)
    accuracy = (score / total) * 100 if total > 0 else 0
    
    print("\n" + "=" * 50)
    print("--- EVALUATION SUMMARY ---")
    print(f"Total Score: {score} / {total} ({accuracy:.1f}%)")
    print("=" * 50 + "\n")
    
    print("Detailed Failure Analysis:")
    failures = [log for log in results_log if log['Result'] == 'FAIL']
    if not failures:
        print("All tests passed!")
    else:
        for log_entry in failures:
            print(f" Q{log_entry['Question']}: {log_entry['Result']}")
            print(f"    Model:   {log_entry['Model_Answer']}")
            print(f"    Correct: {log_entry['Correct_Answer']}")
            print(f"    Rule:    {log_entry['Rationale (for correct answer)']}\n")

if __name__ == "__main__":
    main()