# backend/services/generator.py
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
genai.configure(api_key=GEMINI_API_KEY)

DEFAULT_MODEL = "gemini-1.5-flash"


class LLMGenerator:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        if GEMINI_API_KEY:
            self.model = genai.GenerativeModel(model_name)
            print(f"Generator ready: {model_name}")
        else:
            self.model = None
            print("GEMINI_API_KEY not set")

    def generate_answer(
        self,
        query: str,
        context_chunks: List[str],
        model_name: str = None,
        max_tokens: int = 2048,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        model_to_use = model_name or self.model_name

        if model_name and model_name != self.model_name and GEMINI_API_KEY:
            active_model = genai.GenerativeModel(model_name)
        else:
            active_model = self.model

        context = "\n\n".join(
            [f"[Source {i+1}] {chunk}" for i, chunk in enumerate(context_chunks)]
        )
        prompt = self._build_rag_prompt(query, context)

        if active_model:
            try:
                response = active_model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=max_tokens,
                        temperature=temperature,
                    )
                )
                return {
                    "answer": response.text,
                    "model": model_to_use,
                    "usage": {},
                    "context_used": len(context_chunks)
                }
            except Exception as e:
                print(f"Generation error: {e}")
                return self._fallback_answer(query, context_chunks)
        else:
            return self._fallback_answer(query, context_chunks)

    def _build_rag_prompt(self, query: str, context: str) -> str:
        return f"""You are a helpful assistant that answers questions using the provided context.

CONTEXT:
{context}

QUESTION:
{query}

INSTRUCTIONS:
- Answer using ONLY the information in the context above
- After every sentence that uses a source, add a citation tag like [Source 1] or [Source 2]
- If the context doesn't have enough information, say so clearly
- Be concise and accurate

ANSWER:"""

    def _fallback_answer(self, query: str, context_chunks: List[str]) -> Dict[str, Any]:
        return {
            "answer": f"Based on {len(context_chunks)} retrieved chunks: "
                      + (context_chunks[0][:200] + "..." if context_chunks else "No context available."),
            "model": "fallback",
            "usage": {},
            "context_used": len(context_chunks),
            "note": "Set GEMINI_API_KEY for full functionality"
        }

    def generate_test_questions(
        self,
        document_text: str,
        num_questions: int = 5
    ) -> List[Dict[str, str]]:
        if not self.model:
            return self._fallback_questions(num_questions)

        prompt = f"""Based on this document, generate {num_questions} diverse test questions with answers.

DOCUMENT:
{document_text[:2000]}

Format:
Q1: [question]
A1: [answer]

Q2: [question]
A2: [answer]
"""
        try:
            response = self.model.generate_content(prompt)
            text = response.text
            questions = []
            lines = text.split('\n')
            current_q, current_a = None, None
            for line in lines:
                line = line.strip()
                if line.startswith('Q') and ':' in line:
                    if current_q and current_a:
                        questions.append({"question": current_q, "expected_answer": current_a})
                    current_q = line.split(':', 1)[1].strip()
                    current_a = None
                elif line.startswith('A') and ':' in line and current_q:
                    current_a = line.split(':', 1)[1].strip()
            if current_q and current_a:
                questions.append({"question": current_q, "expected_answer": current_a})
            return questions[:num_questions]
        except Exception as e:
            print(f"Question generation error: {e}")
            return self._fallback_questions(num_questions)

    def _fallback_questions(self, num: int) -> List[Dict[str, str]]:
        return [
            {"question": f"Sample question {i+1}", "expected_answer": "Set GEMINI_API_KEY"}
            for i in range(num)
        ]


_generator = None


def get_generator(model_name: str = DEFAULT_MODEL) -> LLMGenerator:
    global _generator
    if _generator is None:
        _generator = LLMGenerator(model_name)
    return _generator
