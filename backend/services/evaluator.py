# backend/services/evaluator.py
import os
import re
from typing import Dict, Any, List
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
genai.configure(api_key=GEMINI_API_KEY)

DEFAULT_MODEL = "gemini-1.5-flash"


class RAGEvaluator:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        if GEMINI_API_KEY:
            self.model = genai.GenerativeModel(model_name)
            print(f"Evaluator ready: {model_name}")
        else:
            self.model = None
            print("GEMINI_API_KEY not set — evaluator in fallback mode")

    def evaluate_response(
        self,
        query: str,
        generated_answer: str,
        expected_answer: str = None,
        context_chunks: List[str] = None
    ) -> Dict[str, Any]:
        if not self.model:
            return self._fallback_evaluation(query, generated_answer, expected_answer)

        prompt = self._build_evaluation_prompt(
            query, generated_answer, expected_answer, context_chunks
        )

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=0.1)
            )
            response_text = response.text
            scores = self._parse_evaluation(response_text)
            return {
                "scores": scores,
                "feedback": response_text,
                "evaluator_model": self.model_name
            }
        except Exception as e:
            print(f"Evaluation error: {e}")
            return self._fallback_evaluation(query, generated_answer, expected_answer)

    def _build_evaluation_prompt(
        self,
        query: str,
        generated_answer: str,
        expected_answer: str = None,
        context_chunks: List[str] = None
    ) -> str:
        context_section = ""
        if context_chunks:
            context = "\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(context_chunks)])
            context_section = f"\nRETRIEVED CONTEXT:\n{context}\n"

        expected_section = ""
        if expected_answer:
            expected_section = f"\nEXPECTED ANSWER:\n{expected_answer}\n"

        return f"""You are an expert evaluator for RAG systems.

QUERY: {query}

GENERATED ANSWER: {generated_answer}
{expected_section}{context_section}

Score each metric from 0-100:

1. RELEVANCE: Does the answer address the query?
2. ACCURACY: Is the information factually correct?
3. COMPLETENESS: Are all important aspects covered?
4. COHERENCE: Is it well-structured and clear?
5. FAITHFULNESS: Does it stay true to context (no hallucinations)?

Respond in this exact format:
RELEVANCE: [score]/100
ACCURACY: [score]/100
COMPLETENESS: [score]/100
COHERENCE: [score]/100
FAITHFULNESS: [score]/100
OVERALL: [average]/100

FEEDBACK:
[Brief explanation]
"""

    def _parse_evaluation(self, text: str) -> Dict[str, float]:
        scores = {
            "relevance": 0.0, "accuracy": 0.0, "completeness": 0.0,
            "coherence": 0.0, "faithfulness": 0.0, "overall": 0.0
        }
        for line in text.split('\n'):
            line = line.strip().upper()
            if ':' in line:
                metric, value = line.split(':', 1)
                metric = metric.strip().lower()
                numbers = re.findall(r'\d+', value)
                if numbers and metric in scores:
                    scores[metric] = float(numbers[0])

        if scores["overall"] == 0.0:
            scores["overall"] = sum([
                scores["relevance"], scores["accuracy"], scores["completeness"],
                scores["coherence"], scores["faithfulness"]
            ]) / 5.0
        return scores

    def _fallback_evaluation(self, query, generated_answer, expected_answer=None):
        base = 60.0 if len(generated_answer.split()) > 10 else 30.0
        scores = {k: base for k in ["relevance", "accuracy", "completeness", "coherence", "faithfulness", "overall"]}
        return {
            "scores": scores,
            "feedback": "Set GEMINI_API_KEY for AI-powered evaluation.",
            "evaluator_model": "fallback"
        }

    def compare_pipelines(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {"error": "No results to compare"}
        comparisons = []
        for result in results:
            scores = result.get("scores", {})
            comparisons.append({
                "pipeline_config": result.get("config", {}),
                "overall_score": scores.get("overall", 0),
                **{k: scores.get(k, 0) for k in ["relevance", "accuracy", "completeness", "coherence", "faithfulness"]}
            })
        comparisons.sort(key=lambda x: x["overall_score"], reverse=True)
        return {
            "winner": comparisons[0] if comparisons else None,
            "all_results": comparisons,
            "total_pipelines": len(comparisons)
        }


_evaluator = None


def get_evaluator(model_name: str = DEFAULT_MODEL) -> RAGEvaluator:
    global _evaluator
    if _evaluator is None:
        _evaluator = RAGEvaluator(model_name)
    return _evaluator
