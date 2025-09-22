import os
import time
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from typing import Dict, List, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedRAGWithEmbeddings:
    def __init__(self, kb_filepath: str = "rxnorm_enriched_chunks.csv"):
        self.client = self._setup_openai_client()
        self.kb_df = None
        self.kb_embeddings = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.embedding_cache = {}
        
        # Load and vectorize knowledge base
        self._load_and_vectorize_kb(kb_filepath)
        
    def _setup_openai_client(self) -> OpenAI:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)
        logger.info("OpenAI API connection successful")
        return client
    
    def _load_and_vectorize_kb(self, filepath: str):
        """Load knowledge base and create embeddings"""
        logger.info("Loading and vectorizing knowledge base...")
        
        # Load knowledge base
        self.kb_df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(self.kb_df)} entries from knowledge base")
        
        # Check if embeddings exist in cache
        embeddings_cache_file = "kb_embeddings_cache.pkl"
        tfidf_cache_file = "tfidf_cache.pkl"
        
        if os.path.exists(embeddings_cache_file) and os.path.exists(tfidf_cache_file):
            logger.info("Loading cached embeddings...")
            with open(embeddings_cache_file, 'rb') as f:
                self.kb_embeddings = pickle.load(f)
            with open(tfidf_cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.tfidf_vectorizer = cache_data['vectorizer']
                self.tfidf_matrix = cache_data['matrix']
        else:
            # Create embeddings
            self._create_embeddings()
            self._create_tfidf_vectors()
            
            # Cache embeddings
            logger.info("Caching embeddings for future use...")
            with open(embeddings_cache_file, 'wb') as f:
                pickle.dump(self.kb_embeddings, f)
            with open(tfidf_cache_file, 'wb') as f:
                pickle.dump({
                    'vectorizer': self.tfidf_vectorizer,
                    'matrix': self.tfidf_matrix
                }, f)
    
    def _create_embeddings(self):
        """Create OpenAI embeddings for knowledge base"""
        logger.info("Creating OpenAI embeddings...")
        embeddings = []
        
        # Combine medication name and description for better context
        texts_to_embed = []
        for _, row in self.kb_df.iterrows():
            # Create rich text combining name and description
            med_name = str(row.get('STR', ''))
            description = str(row.get('Text_Chunk', ''))
            
            # Enhanced text for embedding
            combined_text = f"Medication: {med_name}. Description: {description}"
            texts_to_embed.append(combined_text)
        
        # Create embeddings in batches to avoid rate limits
        batch_size = 100
        for i in tqdm(range(0, len(texts_to_embed), batch_size), desc="Creating embeddings"):
            batch = texts_to_embed[i:i+batch_size]
            
            try:
                response = self.client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=batch
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error creating embeddings for batch {i}: {e}")
                # Create zero vectors as fallback
                batch_embeddings = [np.zeros(1536) for _ in batch]
                embeddings.extend(batch_embeddings)
        
        self.kb_embeddings = np.array(embeddings)
        logger.info(f"Created {len(embeddings)} embeddings of dimension {self.kb_embeddings.shape[1]}")
    
    def _create_tfidf_vectors(self):
        """Create TF-IDF vectors as backup/complement to embeddings"""
        logger.info("Creating TF-IDF vectors...")
        
        # Prepare texts for TF-IDF (medication names and descriptions)
        texts = []
        for _, row in self.kb_df.iterrows():
            med_name = str(row.get('STR', ''))
            description = str(row.get('Text_Chunk', ''))
            combined_text = f"{med_name} {description}"
            texts.append(combined_text)
        
        # Create TF-IDF vectorizer with medical-focused parameters
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams for medical terms
            min_df=2,  # Ignore very rare terms
            max_df=0.8,  # Ignore very common terms
            lowercase=True,
            token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b'  # Medical terms pattern
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        logger.info(f"Created TF-IDF matrix: {self.tfidf_matrix.shape}")
    
    def get_medication_embedding(self, medication: str, problems: List[str]) -> np.ndarray:
        """Get embedding for medication query with patient context"""
        # Create contextual query
        problems_text = "; ".join(problems) if problems else ""
        query_text = f"Patient has conditions: {problems_text}. Looking for medication: {medication}"
        
        # Check cache first
        if query_text in self.embedding_cache:
            return self.embedding_cache[query_text]
        
        try:
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=[query_text]
            )
            
            embedding = np.array(response.data[0].embedding)
            self.embedding_cache[query_text] = embedding
            return embedding
            
        except Exception as e:
            logger.error(f"Error creating query embedding: {e}")
            return np.zeros(1536)  # Return zero vector as fallback
    
    def semantic_search(self, medication: str, problems: List[str], top_k: int = 5) -> List[Tuple[int, float, str]]:
        """Perform semantic search using embeddings"""
        
        # Get query embedding
        query_embedding = self.get_medication_embedding(medication, problems)
        
        if np.all(query_embedding == 0):
            return []  # No embedding available
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], self.kb_embeddings)[0]
        
        # Get top-k most similar entries
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            similarity_score = similarities[idx]
            kb_entry = self.kb_df.iloc[idx]
            med_name = kb_entry.get('STR', '')
            description = kb_entry.get('Text_Chunk', '')
            
            results.append((idx, similarity_score, f"Medication: {med_name}\nDescription: {description}"))
        
        return results
    
    def tfidf_search(self, medication: str, problems: List[str], top_k: int = 5) -> List[Tuple[int, float, str]]:
        """Backup search using TF-IDF"""
        
        # Create query text
        problems_text = " ".join(problems) if problems else ""
        query_text = f"{medication} {problems_text}"
        
        # Vectorize query
        query_vector = self.tfidf_vectorizer.transform([query_text])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            similarity_score = similarities[idx]
            kb_entry = self.kb_df.iloc[idx]
            med_name = kb_entry.get('STR', '')
            description = kb_entry.get('Text_Chunk', '')
            
            results.append((idx, similarity_score, f"Medication: {med_name}\nDescription: {description}"))
        
        return results
    
    def hybrid_search(self, medication: str, problems: List[str], top_k: int = 3) -> str:
        """Combine semantic and TF-IDF search for better results"""
        
        # Get results from both methods
        semantic_results = self.semantic_search(medication, problems, top_k)
        tfidf_results = self.tfidf_search(medication, problems, top_k)
        
        # Combine and rank results
        all_results = {}
        
        # Add semantic results with higher weight
        for idx, score, text in semantic_results:
            all_results[idx] = {
                'text': text,
                'semantic_score': score,
                'tfidf_score': 0.0,
                'combined_score': score * 0.7  # Higher weight for semantic
            }
        
        # Add TF-IDF results
        for idx, score, text in tfidf_results:
            if idx in all_results:
                all_results[idx]['tfidf_score'] = score
                all_results[idx]['combined_score'] += score * 0.3  # Lower weight for TF-IDF
            else:
                all_results[idx] = {
                    'text': text,
                    'semantic_score': 0.0,
                    'tfidf_score': score,
                    'combined_score': score * 0.3
                }
        
        # Sort by combined score
        sorted_results = sorted(all_results.items(), key=lambda x: x[1]['combined_score'], reverse=True)
        
        # Return top result or indicate not found
        if sorted_results and sorted_results[0][1]['combined_score'] > 0.1:  # Minimum threshold
            best_result = sorted_results[0][1]
            return f"{best_result['text']}\n\nRelevance Score: {best_result['combined_score']:.3f}"
        else:
            return "No relevant medication information found in knowledge base."
    
    def extract_medications(self, med_str: str) -> List[str]:
        """Extract medications (same as before but cleaned up)"""
        if pd.isna(med_str) or not str(med_str).strip():
            return []
        
        medications = re.split(r'[,;]\s*', str(med_str))
        processed_meds = []
        
        for med in medications:
            med = med.strip()
            if med:
                # Light cleaning - preserve most of original text for better matching
                clean_med = re.sub(r'\s+', ' ', med).strip()
                processed_meds.append(clean_med)
        
        return processed_meds
    
    def extract_problems(self, problem_str: str) -> List[str]:
        """Extract problems (same as before)"""
        if pd.isna(problem_str) or not str(problem_str).strip():
            return []
        problems = re.split(r'[,;]\s*', str(problem_str))
        return [p.strip() for p in problems if p.strip()]
    
    def build_enhanced_prompt(self, med: str, kb_info: str, problems: List[str]) -> str:
        """Build enhanced prompt with retrieved context"""
        problems_str = ", ".join(problems) if problems else "No specific problems listed"
        
        template = f"""You are a clinical decision support assistant with access to comprehensive medication information.

MEDICATION: {med}

RETRIEVED MEDICAL INFORMATION:
{kb_info}

PATIENT'S MEDICAL CONDITIONS: {problems_str}

TASK: Based on the retrieved medication information and the patient's conditions, identify which of the patient's medical conditions this medication is most likely prescribed to treat.

INSTRUCTIONS:
1. If the retrieved information is insufficient or irrelevant, respond: "Insufficient information"
2. Focus on matching the medication's known uses with the patient's specific conditions
3. List only the conditions from the patient's list that this medication treats
4. Be conservative - only match if there's clear evidence
5. Separate multiple conditions with semicolons

RESPONSE:"""
        
        return template
    
    def query_openai_enhanced(self, prompt: str, model: str = "gpt-4", temperature: float = 0.0) -> str:
        """Enhanced OpenAI query with better error handling"""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert clinical decision support assistant. Use the provided medication information to make accurate treatment matching decisions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return f"API Error: {str(e)}"
    
    def match_problems_enhanced(self, response: str, problems: List[str]) -> List[str]:
        """Enhanced problem matching with fuzzy matching"""
        if not response or any(phrase in response.lower() for phrase in [
            "insufficient information", "api error", "no relevant"
        ]):
            return []
        
        matched_problems = []
        response_lower = response.lower()
        
        for problem in problems:
            problem_lower = problem.lower().strip()
            
            # Multiple matching strategies
            matched = False
            
            # Exact substring match
            if problem_lower in response_lower:
                matched = True
            
            # Individual word matching for compound conditions
            elif len(problem.split()) > 1:
                problem_words = [w.lower() for w in problem.split() if len(w) > 3]
                if problem_words:
                    # Require at least 50% of significant words to match
                    matching_words = sum(1 for word in problem_words if word in response_lower)
                    if matching_words >= len(problem_words) * 0.5:
                        matched = True
            
            # Single word matching for exact medical terms
            else:
                if len(problem_lower) > 4 and problem_lower in response_lower:
                    matched = True
            
            if matched:
                matched_problems.append(problem)
        
        return matched_problems
    
    def process_patient_enhanced(self, row) -> Dict:
        """Enhanced patient processing with vector search"""
        patient_id = row.get("Patient_ID", row.name)
        
        medications = self.extract_medications(row.get("Outpatient_Medications", ""))
        problems = self.extract_problems(row.get("Past_Medical_History", ""))
        
        result = {
            "Patient_ID": patient_id,
            "Total_Medications": len(medications),
            "Total_Problems": len(problems),
            "Medications": medications,
            "Problems": problems,
            "Enhanced_Results": {},
            "Processing_Summary": {
                "semantic_searches": 0,
                "successful_matches": 0,
                "api_errors": 0
            }
        }
        
        for med in medications:
            try:
                # Use hybrid search to get relevant KB information
                kb_info = self.hybrid_search(med, problems)
                result["Processing_Summary"]["semantic_searches"] += 1
                
                # Build enhanced prompt with retrieved context
                prompt = self.build_enhanced_prompt(med, kb_info, problems)
                
                # Query LLM with enhanced context
                response = self.query_openai_enhanced(prompt)
                
                if "API Error" in response:
                    result["Processing_Summary"]["api_errors"] += 1
                
                # Enhanced problem matching
                matched_problems = self.match_problems_enhanced(response, problems)
                
                if matched_problems:
                    result["Processing_Summary"]["successful_matches"] += 1
                
                result["Enhanced_Results"][med] = {
                    "retrieved_info": kb_info,
                    "llm_response": response,
                    "matched_problems": matched_problems,
                    "search_method": "hybrid_vector_search"
                }
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error processing medication {med}: {e}")
                result["Enhanced_Results"][med] = {
                    "error": str(e),
                    "matched_problems": []
                }
        
        return result

def main_enhanced():
    """Enhanced main function"""
    try:
        # Initialize enhanced RAG system
        rag = ImprovedRAGWithEmbeddings("rxnorm_enriched_chunks.csv")
        
        # Load patient data
        df = pd.read_csv("chest_pain_patients.csv")
        
        if "Patient_ID" not in df.columns:
            df["Patient_ID"] = df.index
        
        # Process patients
        results = []
        sample_size = min(5, len(df))
        
        logger.info(f"Processing {sample_size} patients with enhanced vector search...")
        
        for _, row in tqdm(df.head(sample_size).iterrows(), 
                          total=sample_size, 
                          desc="Processing patients"):
            result = rag.process_patient_enhanced(row)
            results.append(result)
        
        # Save results
        os.makedirs("Mapping", exist_ok=True)
        results_df = pd.DataFrame(results)
        results_df.to_csv("Mapping/enhanced_vector_results.csv", index=False)
        
        # Print summary
        total_searches = sum(r["Processing_Summary"]["semantic_searches"] for r in results)
        total_matches = sum(r["Processing_Summary"]["successful_matches"] for r in results)
        total_errors = sum(r["Processing_Summary"]["api_errors"] for r in results)
        
        logger.info(f"Enhanced processing complete!")
        logger.info(f"Vector searches performed: {total_searches}")
        logger.info(f"Successful matches: {total_matches}")
        logger.info(f"API errors: {total_errors}")
        logger.info(f"Results saved to: Mapping/enhanced_vector_results.csv")
        
        return results
        
    except Exception as e:
        logger.error(f"Enhanced processing failed: {e}")
        raise

if __name__ == "__main__":
    main_enhanced()