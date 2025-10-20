# Clinical RAG Agent: Medication-to-Problem Mapping

![new_pipeline](new_pipeline.png)

This project implements a Retrieval-Augmented Generation (RAG) pipeline to intelligently map a patient's outpatient medications to their corresponding medical problems. It is designed to be resilient to incomplete or inconsistent patient data by leveraging external knowledge from a FHIR server and the clinical reasoning capabilities of a Large Language Model (LLM).

The primary goal is to take a list of patient records and produce a structured summary that accurately links each medication to the specific condition it treats.

## How It Works: The RAG Pipeline

The script processes each patient record through a multi-step pipeline that combines data retrieval with AI-powered generation and analysis.

### 1\. Data Ingestion & Extraction

The process begins by loading patient data from a CSV file (`chest_pain_patients.csv`). For each patient, the script extracts their list of `Outpatient_Medications` and `Past_Medical_History`.

### 2\. Retrieval: Medication Standardization

For each medication name extracted, the pipeline performs the **Retrieval** step. It queries a public HAPI FHIR server to fetch a standardized name and, most importantly, the medication's primary clinical **indication** (the condition it is meant to treat). This step grounds the agent with factual, external medical knowledge.

### 3\. Augment & Generate: LLM-Based Reasoning

This is the **Augmented Generation** step. The retrieved indication is combined with the patient's specific problem list and sent to an LLM (GPT-4). The LLM is prompted to:

  * Confirm the medication's **`primary_indication`**.
  * Identify which problems from the patient's list are treated by the medication (`treated_problems_from_list`).

The LLM acts as a clinical reasoning engine to analyze the retrieved data in the context of the patient's record.

### 4\. Gap-Filling & Output

The script applies a final layer of logic to the LLM's response:

  * If the LLM finds a direct match in the patient's problem list, that match is recorded.
  * If **no match is found** (indicating an incomplete patient record), the script **trusts the `primary_indication` identified by the LLM** and uses it to fill the gap.

The final, clean mappings are then saved to an output CSV file (`Mapping/medication_problem_agent_mapping_summary.csv`).

## Key Technologies

  * **Python**: Core programming language.
  * **Pandas**: For data loading and manipulation.
  * **OpenAI API (GPT-4)**: Serves as the core reasoning engine for mapping and gap-filling.
  * **FHIR (HAPI FHIR Server)**: Used as the external knowledge base for retrieving standardized medication indications.
  * **`fhirclient`**: Python client for interacting with the FHIR server.
  * **`fuzzywuzzy`**: For robust matching between the LLM's output and the patient's problem list.

## Setup and Usage

### 1\. Prerequisites

  * Python 3.9+

### 2\. Installation

Clone the repository and install the required packages:

```bash
git clone <your-repo-url>
cd <your-repo-directory>
pip install -r requirements.txt
```

### 3\. Configuration

Create a `.env` file in the root directory and add your OpenAI API key:

```
OPENAI_API_KEY="your_openai_api_key_here"
```

### 4\. Running the Script

Execute the main script from your terminal:

```bash
python main.py
```

The script will process the patients defined in `main.py` (e.g., the first 4 rows) and save the results in the `Mapping/` directory.
