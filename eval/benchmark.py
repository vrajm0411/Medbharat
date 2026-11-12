"""MedBharat Benchmark — First Indian Healthcare LLM Evaluation.

Tests models across 6 categories:
  1. Indian Drug Knowledge (NLEM, schedules, dosages)
  2. Tropical Disease (dengue, malaria, TB — India context)
  3. AYUSH / Traditional Medicine (Ayurveda)
  4. Indian Medical Exams (AIIMS/NEET style MCQ)
  5. Clinical Reasoning (diagnosis, treatment plans)
  6. Hindi Medical (bilingual medical questions)
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class BenchmarkResult:
    category: str
    total: int
    correct: int
    accuracy: float
    avg_confidence: float
    time_taken: float


# Built-in benchmark questions (representative sample)
# Full benchmark would be 1000+ questions loaded from file
BENCHMARK_QUESTIONS = [
    # Category 1: Indian Drug Knowledge
    {
        "category": "indian_drugs",
        "question": "What is the schedule classification of Paracetamol in India's NLEM?",
        "options": ["Schedule H", "Schedule X", "OTC", "Schedule H1"],
        "answer": 2,
        "explanation": "Paracetamol is available Over-The-Counter (OTC) in India.",
    },
    {
        "category": "indian_drugs",
        "question": "Which of the following is NOT in India's National List of Essential Medicines?",
        "options": ["Metformin", "Atorvastatin", "Botox", "Amoxicillin"],
        "answer": 2,
        "explanation": "Botox (Botulinum toxin) is not in the NLEM.",
    },
    {
        "category": "indian_drugs",
        "question": "The DOTS regimen for tuberculosis in India includes all EXCEPT:",
        "options": ["Isoniazid", "Rifampicin", "Azithromycin", "Pyrazinamide"],
        "answer": 2,
        "explanation": "DOTS uses HRZE: Isoniazid, Rifampicin, Pyrazinamide, Ethambutol.",
    },
    {
        "category": "indian_drugs",
        "question": "Which antimalarial is recommended as first-line for P. falciparum in India?",
        "options": ["Chloroquine", "Artesunate + Lumefantrine", "Quinine", "Primaquine only"],
        "answer": 1,
        "explanation": "ACT (Artemisinin Combination Therapy) is first-line for P. falciparum in India.",
    },
    {
        "category": "indian_drugs",
        "question": "Schedule X drugs in India include:",
        "options": ["Paracetamol", "Morphine and Ketamine", "Amoxicillin", "Cetirizine"],
        "answer": 1,
        "explanation": "Schedule X covers narcotics and psychotropic substances.",
    },
    # Category 2: Tropical Disease
    {
        "category": "tropical_disease",
        "question": "The most common serotype causing severe dengue in India is:",
        "options": ["DEN-1", "DEN-2", "DEN-3", "DEN-4"],
        "answer": 1,
        "explanation": "DEN-2 is most commonly associated with severe dengue and DHF in India.",
    },
    {
        "category": "tropical_disease",
        "question": "Kala-azar (Visceral Leishmaniasis) in India is most prevalent in:",
        "options": ["Kerala", "Bihar", "Tamil Nadu", "Punjab"],
        "answer": 1,
        "explanation": "Bihar accounts for >80% of India's VL cases.",
    },
    {
        "category": "tropical_disease",
        "question": "The vector for Japanese Encephalitis in India is:",
        "options": ["Aedes aegypti", "Anopheles", "Culex tritaeniorhynchus", "Phlebotomus"],
        "answer": 2,
        "explanation": "Culex tritaeniorhynchus is the primary vector for JE in India.",
    },
    {
        "category": "tropical_disease",
        "question": "Widal test is used for diagnosis of:",
        "options": ["Dengue", "Malaria", "Typhoid", "Tuberculosis"],
        "answer": 2,
        "explanation": "Widal test detects antibodies against Salmonella typhi.",
    },
    {
        "category": "tropical_disease",
        "question": "Which snake is responsible for most snakebite deaths in India?",
        "options": ["King Cobra", "Russell's Viper", "Krait", "Saw-scaled Viper"],
        "answer": 1,
        "explanation": "Russell's Viper causes the most snakebite fatalities in India.",
    },
    # Category 3: AYUSH / Traditional Medicine
    {
        "category": "ayush",
        "question": "In Ayurveda, the three doshas are:",
        "options": ["Vata, Pitta, Kapha", "Sattva, Rajas, Tamas", "Rasa, Rakta, Mamsa", "Agni, Soma, Vayu"],
        "answer": 0,
        "explanation": "The three doshas in Ayurveda are Vata, Pitta, and Kapha.",
    },
    {
        "category": "ayush",
        "question": "Ashwagandha (Withania somnifera) is classified in Ayurveda as:",
        "options": ["Rasayana (rejuvenative)", "Vishahara (anti-poison)", "Vedanasthapana (analgesic)", "Raktashodhaka (blood purifier)"],
        "answer": 0,
        "explanation": "Ashwagandha is primarily classified as a Rasayana herb.",
    },
    {
        "category": "ayush",
        "question": "Panchakarma therapy does NOT include:",
        "options": ["Vamana (emesis)", "Virechana (purgation)", "Dialysis", "Basti (enema)"],
        "answer": 2,
        "explanation": "Panchakarma includes Vamana, Virechana, Basti, Nasya, and Raktamokshana.",
    },
    {
        "category": "ayush",
        "question": "AYUSH Ministry in India regulates which systems of medicine?",
        "options": [
            "Only Ayurveda",
            "Ayurveda, Yoga, Unani, Siddha, Homeopathy",
            "Only Homeopathy",
            "Ayurveda and Allopathy",
        ],
        "answer": 1,
        "explanation": "AYUSH stands for Ayurveda, Yoga & Naturopathy, Unani, Siddha, Homeopathy.",
    },
    # Category 4: Indian Medical Exam Style (AIIMS/NEET)
    {
        "category": "indian_exam",
        "question": "A 45-year-old man from rural Bihar presents with fever, hepatosplenomegaly, and pancytopenia. Most likely diagnosis:",
        "options": ["Malaria", "Kala-azar", "Lymphoma", "Typhoid"],
        "answer": 1,
        "explanation": "Classic presentation of Visceral Leishmaniasis (Kala-azar) endemic in Bihar.",
    },
    {
        "category": "indian_exam",
        "question": "Under the National TB Elimination Programme, the target year for TB elimination in India is:",
        "options": ["2025", "2030", "2035", "2040"],
        "answer": 0,
        "explanation": "India targets TB elimination by 2025, five years ahead of the global SDG target.",
    },
    {
        "category": "indian_exam",
        "question": "ASHA workers in India are part of which health mission?",
        "options": ["NRHM/NHM", "PMJAY", "ICDS", "RSBY"],
        "answer": 0,
        "explanation": "ASHA (Accredited Social Health Activist) workers serve under the National Health Mission.",
    },
    {
        "category": "indian_exam",
        "question": "Pulse Polio Programme in India uses which vaccine?",
        "options": ["IPV only", "OPV", "BCG", "MMR"],
        "answer": 1,
        "explanation": "Pulse Polio uses Oral Polio Vaccine (OPV) for mass immunization.",
    },
    # Category 5: Clinical Reasoning
    {
        "category": "clinical",
        "question": "First-line treatment for diabetic ketoacidosis includes:",
        "options": ["Oral metformin", "IV insulin + IV fluids + K+ monitoring", "Sulfonylurea", "Diet control"],
        "answer": 1,
        "explanation": "DKA requires IV insulin, aggressive fluid resuscitation, and potassium monitoring.",
    },
    {
        "category": "clinical",
        "question": "A patient on INH prophylaxis should also receive which supplement?",
        "options": ["Vitamin C", "Vitamin B6 (Pyridoxine)", "Vitamin D", "Iron"],
        "answer": 1,
        "explanation": "Isoniazid can cause peripheral neuropathy; Pyridoxine (B6) is given prophylactically.",
    },
    # Category 6: Hindi Medical (bilingual)
    {
        "category": "hindi_medical",
        "question": "Dengue bukhar ka sabse aam lakshan kya hai? (What is the most common symptom of dengue fever?)",
        "options": ["Khansi (Cough)", "Tez bukhar aur jodo mein dard (High fever and joint pain)", "Pet dard (Stomach pain)", "Sujan (Swelling)"],
        "answer": 1,
        "explanation": "Dengue presents with high fever, severe headache, and joint/muscle pain (break-bone fever).",
    },
    {
        "category": "hindi_medical",
        "question": "ORS ghol mein kya hota hai? (What does ORS solution contain?)",
        "options": [
            "Sirf namak (Only salt)",
            "Namak, cheeni, aur paani (Salt, sugar, and water)",
            "Sirf cheeni (Only sugar)",
            "Doodh aur paani (Milk and water)",
        ],
        "answer": 1,
        "explanation": "ORS contains sodium chloride, glucose/sucrose, potassium chloride, and sodium citrate in water.",
    },
]


class MedBharatBenchmark:
    """Run the MedBharat Benchmark on a model."""

    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device) if isinstance(device, str) else device
        self.questions = BENCHMARK_QUESTIONS

    def _score_option(self, question_text: str, option_text: str) -> float:
        """Score how likely the model considers this option as continuation."""
        prompt = f"Medical Question: {question_text}\nAnswer: {option_text}"
        encoded = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([encoded.ids], dtype=torch.long).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs["logits"]

        # Average log-probability of answer tokens
        # logits[t] predicts token at position t+1, so to score answer tokens
        # we use logits from positions [answer_start-1, ..., end-1] to predict
        # tokens at positions [answer_start, ..., end]
        prompt_ids = self.tokenizer.encode(f"Medical Question: {question_text}\nAnswer: ").ids
        answer_start = len(prompt_ids)
        if answer_start >= logits.shape[1]:
            return 0.0

        # logits at position t predicts token at t+1
        predict_logits = logits[0, answer_start - 1 : -1, :]
        target_tokens = input_ids[0, answer_start:]

        n_tokens = min(predict_logits.shape[0], target_tokens.shape[0])
        if n_tokens == 0:
            return 0.0

        log_probs = F.log_softmax(predict_logits[:n_tokens], dim=-1)
        token_log_probs = log_probs.gather(1, target_tokens[:n_tokens].unsqueeze(1)).squeeze(1)
        return token_log_probs.mean().item()

    def evaluate(self, questions: list = None) -> dict:
        """Run benchmark and return results by category."""
        self.model.eval()
        questions = questions or self.questions

        category_results = {}

        start = time.time()
        for q in questions:
            cat = q["category"]
            if cat not in category_results:
                category_results[cat] = []

            # Score each option
            scores = []
            for option in q["options"]:
                score = self._score_option(q["question"], option)
                scores.append(score)

            predicted = max(range(len(scores)), key=lambda i: scores[i])
            correct = predicted == q["answer"]
            confidence = torch.softmax(torch.tensor(scores), dim=0)[predicted].item()

            category_results[cat].append({
                "correct": correct,
                "confidence": confidence,
                "predicted": predicted,
                "expected": q["answer"],
            })

        elapsed = time.time() - start

        # Aggregate results
        results = {}
        total_correct = 0
        total_questions = 0

        for cat, items in category_results.items():
            correct = sum(1 for x in items if x["correct"])
            total = len(items)
            total_correct += correct
            total_questions += total

            results[cat] = BenchmarkResult(
                category=cat,
                total=total,
                correct=correct,
                accuracy=correct / total if total > 0 else 0,
                avg_confidence=sum(x["confidence"] for x in items) / total,
                time_taken=elapsed / len(questions) * total,
            )

        results["overall"] = BenchmarkResult(
            category="overall",
            total=total_questions,
            correct=total_correct,
            accuracy=total_correct / total_questions if total_questions > 0 else 0,
            avg_confidence=sum(r.avg_confidence * r.total for r in results.values()) / total_questions,
            time_taken=elapsed,
        )

        return results

    def print_results(self, results: dict):
        """Pretty-print benchmark results."""
        print("\n" + "=" * 70)
        print("  MEDBHARAT BENCHMARK RESULTS")
        print("=" * 70)
        print(f"{'Category':<25} {'Correct':>8} {'Total':>8} {'Accuracy':>10} {'Confidence':>12}")
        print("-" * 70)

        for cat, result in sorted(results.items()):
            if cat == "overall":
                continue
            print(
                f"{cat:<25} {result.correct:>8} {result.total:>8} "
                f"{result.accuracy:>9.1%} {result.avg_confidence:>11.3f}"
            )

        overall = results["overall"]
        print("-" * 70)
        print(
            f"{'OVERALL':<25} {overall.correct:>8} {overall.total:>8} "
            f"{overall.accuracy:>9.1%} {overall.avg_confidence:>11.3f}"
        )
        print(f"\nTime: {overall.time_taken:.1f}s")
        print("=" * 70)

    def save_results(self, results: dict, path: str):
        """Save results to JSON."""
        data = {}
        for cat, result in results.items():
            data[cat] = {
                "category": result.category,
                "total": result.total,
                "correct": result.correct,
                "accuracy": result.accuracy,
                "avg_confidence": result.avg_confidence,
                "time_taken": result.time_taken,
            }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to {path}")
