import pandas as pd
import json
import re
from typing import Dict, List, Set, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict

class AutomatedAccuracyValidator:
    def __init__(self):
        # Comprehensive known medication-condition mappings
        self.known_mappings = {
            # Cardiovascular medications
            'metoprolol': ['hypertension', 'high blood pressure', 'heart failure', 'atrial fibrillation', 'angina'],
            'lisinopril': ['hypertension', 'high blood pressure', 'heart failure', 'diabetic nephropathy'],
            'atorvastatin': ['hyperlipidemia', 'high cholesterol', 'dyslipidemia', 'cardiovascular disease'],
            'simvastatin': ['hyperlipidemia', 'high cholesterol', 'dyslipidemia', 'cardiovascular disease'],
            'amlodipine': ['hypertension', 'high blood pressure', 'angina', 'coronary artery disease'],
            'carvedilol': ['heart failure', 'hypertension', 'high blood pressure'],
            'losartan': ['hypertension', 'high blood pressure', 'diabetic nephropathy', 'heart failure'],
            'hydrochlorothiazide': ['hypertension', 'high blood pressure', 'edema', 'heart failure'],
            'furosemide': ['heart failure', 'edema', 'hypertension', 'fluid overload'],
            'aspirin': ['cardiovascular disease', 'stroke prevention', 'myocardial infarction', 'chest pain', 'coronary artery disease'],
            'clopidogrel': ['stroke prevention', 'myocardial infarction', 'peripheral artery disease', 'cardiovascular disease'],
            'warfarin': ['atrial fibrillation', 'blood clots', 'stroke prevention', 'deep vein thrombosis'],
            'digoxin': ['heart failure', 'atrial fibrillation'],
            
            # Diabetes medications
            'metformin': ['diabetes', 'type 2 diabetes', 'diabetes mellitus', 'insulin resistance'],
            'insulin': ['diabetes', 'type 1 diabetes', 'type 2 diabetes', 'diabetes mellitus'],
            'glipizide': ['diabetes', 'type 2 diabetes', 'diabetes mellitus'],
            'glyburide': ['diabetes', 'type 2 diabetes', 'diabetes mellitus'],
            'pioglitazone': ['diabetes', 'type 2 diabetes', 'diabetes mellitus', 'insulin resistance'],
            
            # Respiratory medications
            'albuterol': ['asthma', 'copd', 'bronchospasm', 'chronic obstructive pulmonary disease'],
            'prednisone': ['asthma', 'copd', 'inflammation', 'autoimmune disease', 'allergic reactions'],
            'montelukast': ['asthma', 'allergic rhinitis', 'seasonal allergies'],
            'fluticasone': ['asthma', 'allergic rhinitis', 'nasal congestion'],
            
            # Gastrointestinal medications
            'omeprazole': ['gerd', 'gastroesophageal reflux', 'peptic ulcer', 'heartburn', 'acid reflux'],
            'pantoprazole': ['gerd', 'gastroesophageal reflux', 'peptic ulcer', 'heartburn'],
            'ranitidine': ['gerd', 'gastroesophageal reflux', 'peptic ulcer', 'heartburn'],
            'sucralfate': ['peptic ulcer', 'gastric ulcer', 'duodenal ulcer'],
            
            # Pain and inflammation
            'ibuprofen': ['pain', 'inflammation', 'arthritis', 'headache', 'fever'],
            'naproxen': ['pain', 'inflammation', 'arthritis', 'headache'],
            'acetaminophen': ['pain', 'fever', 'headache'],
            'tramadol': ['pain', 'chronic pain', 'moderate pain'],
            'morphine': ['severe pain', 'chronic pain', 'cancer pain'],
            
            # Antibiotics
            'amoxicillin': ['bacterial infection', 'pneumonia', 'urinary tract infection', 'sinusitis'],
            'azithromycin': ['bacterial infection', 'pneumonia', 'bronchitis', 'sinusitis'],
            'ciprofloxacin': ['bacterial infection', 'urinary tract infection', 'pneumonia'],
            'doxycycline': ['bacterial infection', 'pneumonia', 'acne', 'lyme disease'],
            
            # Mental health
            'sertraline': ['depression', 'anxiety', 'panic disorder', 'obsessive compulsive disorder'],
            'fluoxetine': ['depression', 'anxiety', 'panic disorder', 'bulimia'],
            'escitalopram': ['depression', 'anxiety', 'generalized anxiety disorder'],
            'alprazolam': ['anxiety', 'panic disorder', 'panic attacks'],
            'lorazepam': ['anxiety', 'panic disorder', 'insomnia', 'seizures'],
            'zolpidem': ['insomnia', 'sleep disorder'],
            
            # Thyroid
            'levothyroxine': ['hypothyroidism', 'thyroid disorder', 'goiter'],
            'methimazole': ['hyperthyroidism', 'graves disease', 'thyroid disorder'],
            
            # Seizures/Neurological
            'phenytoin': ['seizures', 'epilepsy', 'seizure disorder'],
            'carbamazepine': ['seizures', 'epilepsy', 'trigeminal neuralgia', 'bipolar disorder'],
            'gabapentin': ['neuropathy', 'seizures', 'nerve pain', 'diabetic neuropathy'],
            
            # Other common medications
            'allopurinol': ['gout', 'hyperuricemia', 'kidney stones'],
            'colchicine': ['gout', 'gout attack', 'familial mediterranean fever'],
            'tamsulosin': ['benign prostatic hyperplasia', 'enlarged prostate', 'urinary retention'],
            'finasteride': ['benign prostatic hyperplasia', 'enlarged prostate', 'male pattern baldness'],
        }
        
        # Create reverse mapping for efficiency
        self.condition_to_meds = defaultdict(set)
        for med, conditions in self.known_mappings.items():
            for condition in conditions:
                self.condition_to_meds[condition.lower()].add(med.lower())
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        if not text or pd.isna(text):
            return ""
        
        text = str(text).lower().strip()
        # Remove common medical abbreviations and normalize
        text = re.sub(r'\b(htn|dm|cad|chf|copd|gerd|uti|mi)\b', lambda m: {
            'htn': 'hypertension',
            'dm': 'diabetes',
            'cad': 'coronary artery disease', 
            'chf': 'heart failure',
            'copd': 'chronic obstructive pulmonary disease',
            'gerd': 'gastroesophageal reflux',
            'uti': 'urinary tract infection',
            'mi': 'myocardial infarction'
        }.get(m.group(), m.group()), text)
        
        # Remove extra whitespace and punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_medication_name(self, med_string: str) -> str:
        """Extract base medication name from string"""
        if not med_string:
            return ""
        
        # Remove dosages and common suffixes
        med = str(med_string).lower().strip()
        med = re.sub(r'\b\d+\s*(mg|mcg|g|ml|units?|%)\b.*$', '', med)
        med = re.sub(r'\s+(tablet|capsule|injection|cream|gel|solution).*$', '', med)
        med = re.sub(r'\s+', ' ', med).strip()
        
        # Take first word if compound
        if ' ' in med:
            med = med.split()[0]
        
        return med
    
    def find_medication_matches(self, medication: str) -> List[str]:
        """Find expected conditions for a medication"""
        normalized_med = self.normalize_text(self.extract_medication_name(medication))
        
        # Direct lookup
        if normalized_med in self.known_mappings:
            return self.known_mappings[normalized_med]
        
        # Partial matching for common variations
        for known_med in self.known_mappings:
            if (normalized_med in known_med or known_med in normalized_med) and len(normalized_med) > 3:
                return self.known_mappings[known_med]
        
        return []
    
    def validate_prediction(self, medication: str, predicted_conditions: List[str], patient_conditions: List[str]) -> Dict:
        """Validate a single medication prediction"""
        expected_conditions = self.find_medication_matches(medication)
        
        if not expected_conditions:
            return {
                'medication': medication,
                'validation_possible': False,
                'reason': 'Medication not in reference database',
                'expected_conditions': [],
                'predicted_conditions': predicted_conditions,
                'patient_conditions': patient_conditions
            }
        
        # Normalize all condition lists
        expected_norm = [self.normalize_text(c) for c in expected_conditions]
        predicted_norm = [self.normalize_text(c) for c in predicted_conditions]
        patient_norm = [self.normalize_text(c) for c in patient_conditions]
        
        # Check if predicted conditions match expected conditions
        correct_predictions = []
        incorrect_predictions = []
        
        for pred_condition in predicted_norm:
            if not pred_condition:
                continue
                
            # Check if this predicted condition matches any expected condition
            is_correct = False
            for exp_condition in expected_norm:
                if (pred_condition in exp_condition or exp_condition in pred_condition) and len(pred_condition) > 3:
                    is_correct = True
                    correct_predictions.append(pred_condition)
                    break
            
            if not is_correct:
                incorrect_predictions.append(pred_condition)
        
        # Check if any expected conditions were in patient list but missed
        missed_opportunities = []
        for exp_condition in expected_norm:
            # Is this expected condition in the patient's condition list?
            in_patient_list = any((exp_condition in pat_cond or pat_cond in exp_condition) and len(exp_condition) > 3 
                                for pat_cond in patient_norm)
            # Was it predicted?
            was_predicted = any((exp_condition in pred_cond or pred_cond in exp_condition) and len(exp_condition) > 3 
                              for pred_cond in predicted_norm)
            
            if in_patient_list and not was_predicted:
                missed_opportunities.append(exp_condition)
        
        return {
            'medication': medication,
            'validation_possible': True,
            'expected_conditions': expected_conditions,
            'predicted_conditions': predicted_conditions,
            'patient_conditions': patient_conditions,
            'correct_predictions': correct_predictions,
            'incorrect_predictions': incorrect_predictions,
            'missed_opportunities': missed_opportunities,
            'precision': len(correct_predictions) / len(predicted_conditions) if predicted_conditions else 1.0,
            'recall': len(correct_predictions) / max(len(correct_predictions) + len(missed_opportunities), 1),
            'has_errors': bool(incorrect_predictions or missed_opportunities)
        }
    
    def validate_results_file(self, results_file: str) -> Dict:
        """Validate entire results file"""
        try:
            df = pd.read_csv(results_file)
        except Exception as e:
            print(f"Error loading results file: {e}")
            return {}
        
        all_validations = []
        validation_stats = {
            'total_medications': 0,
            'validatable_medications': 0,
            'perfect_predictions': 0,
            'predictions_with_errors': 0,
            'total_correct_predictions': 0,
            'total_incorrect_predictions': 0,
            'total_missed_opportunities': 0
        }
        
        for _, row in df.iterrows():
            patient_id = row['Patient_ID']
            
            # Parse detailed results
            try:
                if isinstance(row['Detailed_Results'], str):
                    detailed_results = json.loads(row['Detailed_Results'])
                else:
                    detailed_results = row['Detailed_Results']
            except:
                continue
            
            # Get patient conditions
            patient_conditions = []
            if 'Problems' in row and not pd.isna(row['Problems']):
                patient_conditions = [c.strip() for c in str(row['Problems']).split(',') if c.strip()]
            
            for medication, med_details in detailed_results.items():
                validation_stats['total_medications'] += 1
                
                predicted_conditions = med_details.get('matched_problems', [])
                
                validation = self.validate_prediction(
                    medication, 
                    predicted_conditions, 
                    patient_conditions
                )
                validation['patient_id'] = patient_id
                
                if validation['validation_possible']:
                    validation_stats['validatable_medications'] += 1
                    validation_stats['total_correct_predictions'] += len(validation['correct_predictions'])
                    validation_stats['total_incorrect_predictions'] += len(validation['incorrect_predictions'])
                    validation_stats['total_missed_opportunities'] += len(validation['missed_opportunities'])
                    
                    if not validation['has_errors']:
                        validation_stats['perfect_predictions'] += 1
                    else:
                        validation_stats['predictions_with_errors'] += 1
                
                all_validations.append(validation)
        
        # Calculate aggregate metrics
        if validation_stats['validatable_medications'] > 0:
            validation_stats['validation_coverage'] = validation_stats['validatable_medications'] / validation_stats['total_medications']
            validation_stats['accuracy_rate'] = validation_stats['perfect_predictions'] / validation_stats['validatable_medications']
            validation_stats['overall_precision'] = (
                validation_stats['total_correct_predictions'] / 
                max(validation_stats['total_correct_predictions'] + validation_stats['total_incorrect_predictions'], 1)
            )
            validation_stats['overall_recall'] = (
                validation_stats['total_correct_predictions'] / 
                max(validation_stats['total_correct_predictions'] + validation_stats['total_missed_opportunities'], 1)
            )
        
        return {
            'validations': all_validations,
            'stats': validation_stats
        }
    
    def generate_validation_report(self, results_file: str, output_file: str = 'validation_report.html'):
        """Generate comprehensive validation report"""
        validation_results = self.validate_results_file(results_file)
        
        if not validation_results:
            print("No validation results generated")
            return
        
        validations = validation_results['validations']
        stats = validation_results['stats']
        
        # Create visualizations
        self.create_validation_plots(validations, stats)
        
        # Analyze error patterns
        error_analysis = self.analyze_validation_errors(validations)
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Automated RAG Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .metric {{ background-color: #f0f0f0; padding: 15px; margin: 10px 0; border-radius: 8px; }}
                .success {{ background-color: #d4edda; border: 1px solid #c3e6cb; }}
                .warning {{ background-color: #fff3cd; border: 1px solid #ffeaa7; }}
                .error {{ background-color: #f8d7da; border: 1px solid #f5c6cb; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                .center {{ text-align: center; }}
                h1, h2, h3 {{ color: #333; }}
                ul {{ line-height: 1.6; }}
            </style>
        </head>
        <body>
            <h1>🔍 Automated RAG Validation Report</h1>
            <p><em>Generated using known medication-condition mappings database</em></p>
            
            <h2>📊 Overall Statistics</h2>
            
            <div class="metric">
                <strong>Total Medications Analyzed:</strong> {stats['total_medications']}
            </div>
            
            <div class="metric {'success' if stats.get('validation_coverage', 0) > 0.5 else 'warning'}">
                <strong>Validation Coverage:</strong> {stats.get('validation_coverage', 0):.1%} 
                ({stats['validatable_medications']} out of {stats['total_medications']} medications could be validated)
            </div>
            
            <div class="metric {'success' if stats.get('accuracy_rate', 0) > 0.7 else 'error'}">
                <strong>Accuracy Rate:</strong> {stats.get('accuracy_rate', 0):.1%}
                ({stats['perfect_predictions']} perfect predictions out of {stats['validatable_medications']} validatable)
            </div>
            
            <div class="metric">
                <strong>Overall Precision:</strong> {stats.get('overall_precision', 0):.1%}
                ({stats['total_correct_predictions']} correct out of {stats['total_correct_predictions'] + stats['total_incorrect_predictions']} total predictions)
            </div>
            
            <div class="metric">
                <strong>Overall Recall:</strong> {stats.get('overall_recall', 0):.1%}
                ({stats['total_correct_predictions']} found out of {stats['total_correct_predictions'] + stats['total_missed_opportunities']} possible)
            </div>
            
            <div class="metric {'error' if stats['predictions_with_errors'] > stats['perfect_predictions'] else 'success'}">
                <strong>Error Rate:</strong> {(stats['predictions_with_errors'] / max(stats['validatable_medications'], 1)):.1%}
                ({stats['predictions_with_errors']} medications had prediction errors)
            </div>
            
            <h2>🔍 Error Analysis</h2>
            
            <h3>Most Common Incorrect Predictions (False Positives)</h3>
            <ul>
        """
        
        for error, count in error_analysis['common_incorrect'][:10]:
            html_content += f"<li><strong>{error}:</strong> {count} times</li>"
        
        html_content += """
            </ul>
            
            <h3>Most Common Missed Opportunities (False Negatives)</h3>
            <ul>
        """
        
        for error, count in error_analysis['common_missed'][:10]:
            html_content += f"<li><strong>{error}:</strong> {count} times</li>"
        
        html_content += """
            </ul>
            
            <h3>Medications with Highest Error Rates</h3>
            <ul>
        """
        
        for med, error_rate in error_analysis['problematic_medications'][:10]:
            html_content += f"<li><strong>{med}:</strong> {error_rate:.1%} error rate</li>"
        
        html_content += f"""
            </ul>
            
            <h2>📋 Detailed Validation Results</h2>
            <p><em>Showing validatable medications only. Red rows indicate errors.</em></p>
            
            <table>
                <tr>
                    <th>Patient ID</th>
                    <th>Medication</th>
                    <th>Expected Conditions</th>
                    <th>Predicted Conditions</th>
                    <th>✅ Correct</th>
                    <th>❌ Incorrect</th>
                    <th>😞 Missed</th>
                    <th class="center">Status</th>
                </tr>
        """
        
        # Show first 100 validatable results
        validatable_results = [v for v in validations if v['validation_possible']][:100]
        
        for validation in validatable_results:
            row_class = 'error' if validation['has_errors'] else 'success'
            status = '❌ Errors' if validation['has_errors'] else '✅ Perfect'
            
            html_content += f"""
                <tr class="{row_class}">
                    <td>{validation['patient_id']}</td>
                    <td>{validation['medication']}</td>
                    <td>{'; '.join(validation['expected_conditions'][:3])}{'...' if len(validation['expected_conditions']) > 3 else ''}</td>
                    <td>{'; '.join(validation['predicted_conditions']) if validation['predicted_conditions'] else 'None'}</td>
                    <td>{'; '.join(validation['correct_predictions']) if validation['correct_predictions'] else 'None'}</td>
                    <td>{'; '.join(validation['incorrect_predictions']) if validation['incorrect_predictions'] else 'None'}</td>
                    <td>{'; '.join(validation['missed_opportunities']) if validation['missed_opportunities'] else 'None'}</td>
                    <td class="center">{status}</td>
                </tr>
            """
        
        html_content += f"""
            </table>
            
            <p><em>Showing first 100 validatable results out of {len(validatable_results)} total.</em></p>
            
            <h2>📝 Methodology Notes</h2>
            <ul>
                <li><strong>Known Mappings Database:</strong> {len(self.known_mappings)} medications with {sum(len(conditions) for conditions in self.known_mappings.values())} total condition associations</li>
                <li><strong>Validation Coverage:</strong> Only medications in our reference database can be validated</li>
                <li><strong>Matching Logic:</strong> Uses fuzzy text matching to handle variations in condition names</li>
                <li><strong>Precision:</strong> Correct predictions / Total predictions</li>
                <li><strong>Recall:</strong> Correct predictions / (Correct predictions + Missed opportunities)</li>
            </ul>
            
            <h2>💡 Recommendations</h2>
        """
        
        # Generate recommendations based on results
        recommendations = []
        
        if stats.get('validation_coverage', 0) < 0.5:
            recommendations.append("🔍 <strong>Low validation coverage:</strong> Consider expanding the known mappings database to include more medications from your dataset")
        
        if stats.get('accuracy_rate', 0) < 0.6:
            recommendations.append("⚠️ <strong>Low accuracy:</strong> Review RAG system prompts and knowledge base quality")
        
        if stats.get('overall_precision', 0) < 0.7:
            recommendations.append("🎯 <strong>Low precision:</strong> System is making too many incorrect predictions - consider more restrictive matching")
        
        if stats.get('overall_recall', 0) < 0.7:
            recommendations.append("📈 <strong>Low recall:</strong> System is missing valid medication-condition matches - consider more permissive matching")
        
        if error_analysis['common_incorrect']:
            recommendations.append(f"🚫 <strong>Common false positives:</strong> '{error_analysis['common_incorrect'][0][0]}' is frequently incorrectly predicted")
        
        if not recommendations:
            recommendations.append("✅ <strong>Good performance:</strong> The RAG system shows solid accuracy across validated medications")
        
        for rec in recommendations:
            html_content += f"<li>{rec}</li>"
        
        html_content += """
            </ul>
            
            <p><em>Report generated automatically using known medication-condition mappings. 
            For more comprehensive evaluation, consider manual annotation of a sample dataset.</em></p>
        </body>
        </html>
        """
        
        # Save HTML report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Save detailed CSV
        validation_df = pd.DataFrame(validations)
        validation_df.to_csv('detailed_validation_results.csv', index=False)
        
        print(f"✅ Validation report saved to: {output_file}")
        print(f"📊 Detailed results saved to: detailed_validation_results.csv")
        print(f"📈 Validation plots saved to: validation_plots.png")
        
        # Print summary to console
        print(f"\n{'='*60}")
        print(f"VALIDATION SUMMARY")
        print(f"{'='*60}")
        print(f"Validation Coverage: {stats.get('validation_coverage', 0):.1%}")
        print(f"Accuracy Rate: {stats.get('accuracy_rate', 0):.1%}")
        print(f"Overall Precision: {stats.get('overall_precision', 0):.1%}")
        print(f"Overall Recall: {stats.get('overall_recall', 0):.1%}")
        print(f"Total Medications: {stats['total_medications']}")
        print(f"Validatable: {stats['validatable_medications']}")
        print(f"Perfect Predictions: {stats['perfect_predictions']}")
        print(f"With Errors: {stats['predictions_with_errors']}")
        
        return validation_results
    
    def analyze_validation_errors(self, validations: List[Dict]) -> Dict:
        """Analyze patterns in validation errors"""
        common_incorrect = Counter()
        common_missed = Counter()
        medication_errors = defaultdict(list)
        
        for validation in validations:
            if not validation['validation_possible']:
                continue
                
            med = validation['medication']
            
            for incorrect in validation['incorrect_predictions']:
                common_incorrect[incorrect] += 1
                medication_errors[med].append('incorrect')
            
            for missed in validation['missed_opportunities']:
                common_missed[missed] += 1
                medication_errors[med].append('missed')
        
        # Calculate error rates by medication
        problematic_medications = []
        for med, errors in medication_errors.items():
            error_rate = len(errors) / max(len(errors), 1)  # Simple error rate calculation
            problematic_medications.append((med, error_rate))
        
        problematic_medications.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'common_incorrect': common_incorrect.most_common(),
            'common_missed': common_missed.most_common(),
            'problematic_medications': problematic_medications
        }
    
    def create_validation_plots(self, validations: List[Dict], stats: Dict):
        """Create validation visualization plots"""
        validatable = [v for v in validations if v['validation_possible']]
        
        if not validatable:
            print("No validatable medications found for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Automated RAG Validation Analysis', fontsize=16, fontweight='bold')
        
        # 1. Overall metrics bar chart
        metrics = {
            'Coverage': stats.get('validation_coverage', 0),
            'Accuracy': stats.get('accuracy_rate', 0),
            'Precision': stats.get('overall_precision', 0),
            'Recall': stats.get('overall_recall', 0)
        }
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        bars = axes[0,0].bar(metrics.keys(), [v*100 for v in metrics.values()], color=colors)
        axes[0,0].set_title('Overall Validation Metrics (%)', fontweight='bold')
        axes[0,0].set_ylabel('Percentage')
        axes[0,0].set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, value in zip(bars, metrics.values()):
            height = bar.get_height()
            axes[0,0].text(bar.get_x() + bar.get_width()/2., height + 1,
                         f'{value:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Precision-Recall scatter plot
        precisions = [v['precision'] for v in validatable if 'precision' in v]
        recalls = [v['recall'] for v in validatable if 'recall' in v]
        
        scatter = axes[0,1].scatter(recalls, precisions, alpha=0.6, s=50, c='#2E86AB')
        axes[0,1].set_xlabel('Recall')
        axes[0,1].set_ylabel('Precision')
        axes[0,1].set_title('Precision vs Recall by Medication', fontweight='bold')
        axes[0,1].set_xlim(0, 1)
        axes[0,1].set_ylim(0, 1)
        axes[0,1].plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Perfect Balance')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Error distribution
        perfect = sum(1 for v in validatable if not v['has_errors'])
        with_errors = len(validatable) - perfect
        not_validatable = len(validations) - len(validatable)
        
        sizes = [perfect, with_errors, not_validatable]
        labels = ['Perfect Predictions', 'With Errors', 'Not Validatable']
        colors = ['#2E8B57', '#DC143C', '#D3D3D3']
        
        wedges, texts, autotexts = axes[1,0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1,0].set_title('Prediction Quality Distribution', fontweight='bold')
        
        # 4. Top problematic medications
        med_error_counts = Counter()
        for v in validatable:
            if v['has_errors']:
                med_error_counts[v['medication']] += 1
        
        if med_error_counts:
            top_problematic = med_error_counts.most_common(10)
            meds, counts = zip(*top_problematic)
            
            bars = axes[1,1].barh(range(len(meds)), counts, color='#DC143C', alpha=0.7)
            axes[1,1].set_yticks(range(len(meds)))
            axes[1,1].set_yticklabels([med[:15] + '...' if len(med) > 15 else med for med in meds])
            axes[1,1].set_xlabel('Number of Errors')
            axes[1,1].set_title('Top 10 Medications with Most Errors', fontweight='bold')
            axes[1,1].grid(True, alpha=0.3, axis='x')
            
            # Add value labels on bars
            for i, (bar, count) in enumerate(zip(bars, counts)):
                axes[1,1].text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                             str(count), va='center', fontweight='bold')
        else:
            axes[1,1].text(0.5, 0.5, 'No errors found!\n🎉', ha='center', va='center', 
                          transform=axes[1,1].transAxes, fontsize=16, fontweight='bold')
            axes[1,1].set_title('Medication Error Analysis', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('validation_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Validation plots saved as: validation_plots.png")

def main():
    """Main function to run automated validation"""
    import sys
    
    print("🚀 Starting Automated RAG Accuracy Validation")
    print("=" * 60)
    
    # Initialize validator
    validator = AutomatedAccuracyValidator()
    
    # Check if results file provided as argument
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
    else:
        # Default file path - adjust as needed
        results_file = "Mapping/coverage_improved_results.csv"
    
    # Check if file exists
    import os
    if not os.path.exists(results_file):
        print(f"❌ Error: Results file not found: {results_file}")
        print(f"Please ensure your RAG results CSV file exists at the specified path.")
        print(f"Expected columns: Patient_ID, Detailed_Results, Problems (optional)")
        return
    
    print(f"📁 Loading results from: {results_file}")
    
    try:
        # Run validation
        validation_results = validator.generate_validation_report(results_file)
        
        if validation_results:
            stats = validation_results['stats']
            print(f"\n🎯 VALIDATION COMPLETE!")
            print(f"📊 {stats['validatable_medications']}/{stats['total_medications']} medications validated")
            print(f"✅ {stats['perfect_predictions']} perfect predictions")
            print(f"❌ {stats['predictions_with_errors']} with errors")
            print(f"📈 Overall accuracy: {stats.get('accuracy_rate', 0):.1%}")
            
            # Quick recommendations
            if stats.get('accuracy_rate', 0) > 0.8:
                print(f"🌟 Excellent performance! Your RAG system is working well.")
            elif stats.get('accuracy_rate', 0) > 0.6:
                print(f"⚠️  Moderate performance. Consider reviewing error patterns in the report.")
            else:
                print(f"🚨 Performance needs improvement. Check the detailed report for insights.")
        
    except Exception as e:
        print(f"❌ Error during validation: {str(e)}")
        print(f"Please check your results file format and try again.")

if __name__ == "__main__":
    main()