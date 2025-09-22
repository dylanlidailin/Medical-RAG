import os
import time
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import re
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FixedRAGWithEmbeddings:
    def __init__(self, kb_filepath: str = "rxnorm_enriched_chunks.csv"):
        self.client = self._setup_openai_client()
        self.kb_df = None
        self.kb_embeddings = None
        self.embedding_cache = {}
        self.fallback_kb = self._create_fallback_kb()
        
        # Load and clean knowledge base
        self._load_and_clean_kb(kb_filepath)
        
    def _setup_openai_client(self) -> OpenAI:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY")
        return OpenAI(api_key=api_key)
    
    def _create_fallback_kb(self) -> Dict[str, str]:
        """Create fallback knowledge base with common medications"""
        return {
            'enalapril': 'ACE inhibitor used for hypertension, heart failure, and diabetic nephropathy. Reduces blood pressure by blocking angiotensin-converting enzyme.',
            'propranolol': 'Beta-blocker used for hypertension, angina, arrhythmias, anxiety, and migraine prevention. Blocks beta-adrenergic receptors.',
            'glipizide': 'Sulfonylurea antidiabetic medication used for type 2 diabetes mellitus. Stimulates insulin release from pancreatic beta cells.',
            'reserpine': 'Antihypertensive agent that depletes norepinephrine stores. Used for high blood pressure and certain psychiatric conditions.',
            'hydrocortisone': 'Corticosteroid with anti-inflammatory and immunosuppressive properties. Used for inflammation, allergies, and hormonal replacement.',
            'sotalol': 'Antiarrhythmic beta-blocker used for atrial fibrillation, ventricular arrhythmias, and ventricular tachycardia.',
            'insulin': 'Hormone used for diabetes mellitus type 1 and type 2. Regulates glucose metabolism and lowers blood sugar levels.',
            'methotrexate': 'Immunosuppressant and chemotherapy agent used for rheumatoid arthritis, psoriasis, and certain cancers.',
            'felodipine': 'Calcium channel blocker used for hypertension and angina. Relaxes blood vessels to reduce blood pressure.',
            'sulfasalazine': 'Anti-inflammatory medication used for rheumatoid arthritis, inflammatory bowel disease, and ulcerative colitis.',
            'semaglutide': 'GLP-1 receptor agonist used for type 2 diabetes and weight management. Improves glucose control and promotes weight loss.',
            'mesalamine': 'Anti-inflammatory medication used for inflammatory bowel disease, ulcerative colitis, and Crohn\'s disease.',
            'everolimus': 'mTOR inhibitor used as immunosuppressant after organ transplantation and for certain cancers.',
            'minoxidil': 'Vasodilator used for severe hypertension and androgenetic alopecia (hair loss). Dilates blood vessels and promotes hair growth.',
            'metformin': 'Biguanide antidiabetic medication for type 2 diabetes. Decreases glucose production and improves insulin sensitivity.',
            'lisinopril': 'ACE inhibitor used for hypertension, heart failure, and post-myocardial infarction. Blocks angiotensin-converting enzyme.',
            'atorvastatin': 'HMG-CoA reductase inhibitor (statin) used for hypercholesterolemia and cardiovascular disease prevention.',
            'amlodipine': 'Calcium channel blocker used for hypertension and angina. Long-acting dihydropyridine that relaxes blood vessels.'
        }
    
    def _load_and_clean_kb(self, filepath: str):
        """Load and clean knowledge base, filter out poor quality entries"""
        logger.info("Loading and cleaning knowledge base...")
        
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} entries from knowledge base")
            
            # Clean and filter the knowledge base
            cleaned_entries = []
            
            for _, row in df.iterrows():
                med_name = str(row.get('STR', '')).strip()
                description = str(row.get('Text_Chunk', '')).strip()
                
                # Skip entries with poor descriptions
                if (not description or 
                    len(description) < 10 or 
                    description.endswith(' - .') or
                    description.count(' ') < 3):
                    continue
                
                # Clean medication name
                clean_med_name = self._clean_medication_name(med_name)
                if not clean_med_name:
                    continue
                
                cleaned_entries.append({
                    'original_name': med_name,
                    'clean_name': clean_med_name,
                    'description': description
                })
            
            self.kb_df = pd.DataFrame(cleaned_entries)
            logger.info(f"After cleaning: {len(self.kb_df)} quality entries retained")
            
            if len(self.kb_df) == 0:
                logger.warning("No quality entries found in knowledge base, using fallback only")
                self.kb_df = self._create_fallback_dataframe()
            
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            logger.info("Using fallback knowledge base")
            self.kb_df = self._create_fallback_dataframe()
    
    def _create_fallback_dataframe(self) -> pd.DataFrame:
        """Create DataFrame from fallback knowledge base"""
        fallback_data = []
        for med, desc in self.fallback_kb.items():
            fallback_data.append({
                'original_name': med.capitalize(),
                'clean_name': med.lower(),
                'description': desc
            })
        return pd.DataFrame(fallback_data)
    
    def _clean_medication_name(self, med_name: str) -> str:
        """Extract clean medication name"""
        if not med_name or pd.isna(med_name):
            return ""
        
        # Remove dosage information and formulations
        clean_name = str(med_name).lower()
        clean_name = re.sub(r'\d+\s*(mg|mcg|g|ml|%|units?)\b.*', '', clean_name)
        clean_name = re.sub(r'\[.*?\]', '', clean_name)  # Remove brand names in brackets
        clean_name = re.sub(r'\(.*?\)', '', clean_name)  # Remove parenthetical info
        clean_name = re.sub(r'\s+(tablet|capsule|injection|cream|gel|solution|pill).*', '', clean_name)
        
        # Extract first meaningful word
        words = clean_name.split()
        if words:
            first_word = words[0].strip()
            if len(first_word) > 3:  # Avoid very short words
                return first_word
        
        return ""
    
    def smart_medication_search(self, medication: str, problems: List[str]) -> str:
        """Smart search combining exact matching, fuzzy matching, and fallback KB"""
        medication_clean = self._clean_medication_name(medication)
        
        # Strategy 1: Exact match in cleaned names
        exact_matches = self.kb_df[self.kb_df['clean_name'] == medication_clean]
        if not exact_matches.empty:
            best_match = exact_matches.iloc[0]
            return f"Medication: {best_match['original_name']}\nDescription: {best_match['description']}\nMatch Type: Exact"
        
        # Strategy 2: Fuzzy matching
        best_ratio = 0
        best_match = None
        
        for _, row in self.kb_df.iterrows():
            ratio = SequenceMatcher(None, medication_clean, row['clean_name']).ratio()
            if ratio > best_ratio and ratio > 0.8:  # High threshold for fuzzy matching
                best_ratio = ratio
                best_match = row
        
        if best_match is not None:
            return f"Medication: {best_match['original_name']}\nDescription: {best_match['description']}\nMatch Type: Fuzzy ({best_ratio:.2f})"
        
        # Strategy 3: Fallback knowledge base
        if medication_clean in self.fallback_kb:
            return f"Medication: {medication}\nDescription: {self.fallback_kb[medication_clean]}\nMatch Type: Fallback KB"
        
        # Strategy 4: Partial matching in fallback
        for known_med, desc in self.fallback_kb.items():
            if (medication_clean in known_med or known_med in medication_clean) and len(medication_clean) > 3:
                return f"Medication: {medication}\nDescription: {desc}\nMatch Type: Fallback Partial"
        
        return f"Medication: {medication}\nDescription: No detailed information available for this medication.\nMatch Type: Not Found"
    
    def build_enhanced_prompt(self, med: str, kb_info: str, problems: List[str]) -> str:
        """Build enhanced prompt"""
        problems_str = ", ".join(problems) if problems else "No specific medical conditions listed"
        
        template = f"""You are a clinical pharmacist assistant analyzing medication-condition relationships.

MEDICATION: {med}

MEDICATION INFORMATION:
{kb_info}

PATIENT'S MEDICAL CONDITIONS: {problems_str}

TASK: Based on the medication information provided, determine which of the patient's medical conditions this medication is commonly used to treat.

GUIDELINES:
1. Only match conditions if there's clear therapeutic indication
2. If medication information is insufficient, respond: "Insufficient information"
3. If no patient conditions match the medication's uses, respond: "No matching conditions"
4. List matching conditions exactly as they appear in the patient's list
5. Separate multiple conditions with semicolons

Which of the patient's conditions does this medication treat?"""
        
        return template
    
    def query_openai(self, prompt: str, model: str = "gpt-4", temperature: float = 0.0) -> str:
        """Query OpenAI with error handling"""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a clinical pharmacist with expertise in medication-condition matching."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=150
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return f"API Error: {str(e)}"
    
    def match_problems_improved(self, response: str, problems: List[str]) -> List[str]:
        """Improved problem matching"""
        if not response or any(phrase in response.lower() for phrase in [
            "insufficient information", "no matching conditions", "api error", "not found"
        ]):
            return []
        
        matched_problems = []
        response_lower = response.lower()
        
        for problem in problems:
            problem_lower = problem.lower().strip()
            
            # Exact match
            if problem_lower in response_lower:
                matched_problems.append(problem)
                continue
            
            # Word-level matching for compound conditions
            problem_words = [w for w in problem_lower.split() if len(w) > 3]
            if len(problem_words) > 1:
                matching_words = sum(1 for word in problem_words if word in response_lower)
                if matching_words >= len(problem_words) * 0.6:  # 60% of words must match
                    matched_problems.append(problem)
                    continue
            
            # Medical abbreviation expansion
            medical_abbrevs = {
                'dm': 'diabetes',
                'htn': 'hypertension',
                'cad': 'coronary artery disease',
                'chf': 'heart failure',
                'copd': 'chronic obstructive pulmonary disease'
            }
            
            expanded_problem = medical_abbrevs.get(problem_lower, problem_lower)
            if expanded_problem != problem_lower and expanded_problem in response_lower:
                matched_problems.append(problem)
        
        return matched_problems
    
    def extract_medications(self, med_str: str) -> List[str]:
        """Extract medications"""
        if pd.isna(med_str) or not str(med_str).strip():
            return []
        
        medications = re.split(r'[,;]\s*', str(med_str))
        return [med.strip() for med in medications if med.strip()]
    
    def extract_problems(self, problem_str: str) -> List[str]:
        """Extract problems"""
        if pd.isna(problem_str) or not str(problem_str).strip():
            return []
        problems = re.split(r'[,;]\s*', str(problem_str))
        return [p.strip() for p in problems if p.strip()]
    
    def process_patient_fixed(self, row) -> Dict:
        """Fixed patient processing"""
        patient_id = row.get("Patient_ID", row.name)
        
        medications = self.extract_medications(row.get("Outpatient_Medications", ""))
        problems = self.extract_problems(row.get("Past_Medical_History", ""))
        
        result = {
            "Patient_ID": patient_id,
            "Total_Medications": len(medications),
            "Total_Problems": len(problems),
            "Medications": medications,
            "Problems": problems,
            "Fixed_Results": {},
            "Processing_Summary": {
                "exact_matches": 0,
                "fuzzy_matches": 0,
                "fallback_matches": 0,
                "not_found": 0,
                "successful_conditions": 0
            }
        }
        
        for med in medications:
            try:
                # Smart search for medication info
                kb_info = self.smart_medication_search(med, problems)
                
                # Track match type
                if "Exact" in kb_info:
                    result["Processing_Summary"]["exact_matches"] += 1
                elif "Fuzzy" in kb_info:
                    result["Processing_Summary"]["fuzzy_matches"] += 1
                elif "Fallback" in kb_info:
                    result["Processing_Summary"]["fallback_matches"] += 1
                else:
                    result["Processing_Summary"]["not_found"] += 1
                
                # Build prompt and query LLM
                prompt = self.build_enhanced_prompt(med, kb_info, problems)
                response = self.query_openai(prompt)
                
                # Match problems
                matched_problems = self.match_problems_improved(response, problems)
                
                if matched_problems:
                    result["Processing_Summary"]["successful_conditions"] += len(matched_problems)
                
                result["Fixed_Results"][med] = {
                    "retrieved_info": kb_info,
                    "llm_response": response,
                    "matched_problems": matched_problems,
                    "search_method": "smart_hybrid_search"
                }
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error processing {med}: {e}")
                result["Fixed_Results"][med] = {
                    "error": str(e),
                    "matched_problems": []
                }
        
        return result

def main_fixed():
    """Fixed main function"""
    try:
        logger.info("Starting FIXED RAG processing...")
        
        # Initialize fixed RAG system
        rag = FixedRAGWithEmbeddings("rxnorm_enriched_chunks.csv")
        
        # Load patient data
        df = pd.read_csv("chest_pain_patients.csv")
        
        if "Patient_ID" not in df.columns:
            df["Patient_ID"] = df.index
        
        # Process patients
        results = []
        sample_size = min(3, len(df))  # Start with fewer patients to test
        
        logger.info(f"Processing {sample_size} patients with FIXED approach...")
        
        for _, row in tqdm(df.head(sample_size).iterrows(), 
                          total=sample_size, 
                          desc="Processing patients"):
            result = rag.process_patient_fixed(row)
            results.append(result)
            
            # Print progress for first patient
            if len(results) == 1:
                logger.info("First patient results:")
                summary = result["Processing_Summary"]
                logger.info(f"  Exact matches: {summary['exact_matches']}")
                logger.info(f"  Fuzzy matches: {summary['fuzzy_matches']}")
                logger.info(f"  Fallback matches: {summary['fallback_matches']}")
                logger.info(f"  Not found: {summary['not_found']}")
                logger.info(f"  Condition matches: {summary['successful_conditions']}")
        
        # Save results
        os.makedirs("Mapping", exist_ok=True)
        results_df = pd.DataFrame(results)
        results_df.to_csv("Mapping/fixed_results.csv", index=False)
        
        # Overall summary
        total_exact = sum(r["Processing_Summary"]["exact_matches"] for r in results)
        total_fuzzy = sum(r["Processing_Summary"]["fuzzy_matches"] for r in results)
        total_fallback = sum(r["Processing_Summary"]["fallback_matches"] for r in results)
        total_not_found = sum(r["Processing_Summary"]["not_found"] for r in results)
        total_conditions = sum(r["Processing_Summary"]["successful_conditions"] for r in results)
        total_meds = sum(r["Total_Medications"] for r in results)
        
        logger.info(f"\n{'='*50}")
        logger.info(f"FIXED PROCESSING SUMMARY")
        logger.info(f"{'='*50}")
        logger.info(f"Total medications: {total_meds}")
        logger.info(f"Exact matches: {total_exact} ({total_exact/total_meds*100:.1f}%)")
        logger.info(f"Fuzzy matches: {total_fuzzy} ({total_fuzzy/total_meds*100:.1f}%)")
        logger.info(f"Fallback matches: {total_fallback} ({total_fallback/total_meds*100:.1f}%)")
        logger.info(f"Not found: {total_not_found} ({total_not_found/total_meds*100:.1f}%)")
        logger.info(f"Successful condition matches: {total_conditions}")
        logger.info(f"Results saved to: Mapping/fixed_results.csv")
        
        return results
        
    except Exception as e:
        logger.error(f"Fixed processing failed: {e}")
        raise

if __name__ == "__main__":
    main_fixed()