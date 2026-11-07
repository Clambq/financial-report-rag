"""Deterministic local Qwen generation for controlled FinQA experiments."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Any

from finqa_rag.data.models import EvidenceUnit
from finqa_rag.retrieval.types import ScoredEvidence


@dataclass(frozen=True, slots=True)
class GenerationResult:
    raw_text: str
    parsed_text: str
    latency_ms: float


_NUMBER_FRAGMENT = re.compile(
    r"(?<![A-Za-z0-9_])(?:[$€£¥]\s*)?\(?[+-]?(?:[0-9][0-9,]*(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?%?\)?"
)
_PROGRAM_OPERATION = (
    r"(?:add|subtract|multiply|divide|exp|greater|table_sum|table_average|"
    r"table_max|table_min)"
)
_PROGRAM_SEQUENCE = re.compile(
    rf"\b{_PROGRAM_OPERATION}\([^()\r\n]*\)"
    rf"(?:\s*,\s*{_PROGRAM_OPERATION}\([^()\r\n]*\))*"
)


def parse_numeric_response(raw_text: str) -> str:
    """Extract the last yes/no or financial-number token from model output."""

    text = raw_text.strip().replace("```", "")
    lowered = text.lower()
    decisions = re.findall(r"\b(?:yes|no)\b", lowered)
    numbers = _NUMBER_FRAGMENT.findall(text)
    if decisions and not numbers:
        return decisions[-1]
    if numbers:
        return numbers[-1].strip()
    return text.splitlines()[-1].strip() if text else ""


def parse_program_response(raw_text: str) -> str:
    """Extract the longest fixed-grammar program and discard surrounding prose."""

    matches = _PROGRAM_SEQUENCE.findall(raw_text.replace("```", ""))
    return max(matches, key=len).strip() if matches else raw_text.strip()


class QwenGenerator:
    """One shared Qwen model with prompts for arithmetic and RAG service routes."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-4B-Instruct-2507",
        *,
        max_new_tokens: int = 64,
        do_sample: bool = False,
        tokenizer: Any | None = None,
        model: Any | None = None,
    ) -> None:
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        if tokenizer is None or model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype="auto",
                device_map="auto",
            )
        self._tokenizer = tokenizer
        self._model = model

    def _generate(self, system: str, user: str) -> GenerationResult:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self._tokenizer([prompt], return_tensors="pt")
        if hasattr(model_inputs, "to") and hasattr(self._model, "device"):
            model_inputs = model_inputs.to(self._model.device)
        input_length = model_inputs["input_ids"].shape[1]
        started = perf_counter_ns()
        generated = self._model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
        )
        latency_ms = (perf_counter_ns() - started) / 1_000_000
        output_tokens = generated[0][input_length:]
        raw_text = self._tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
        return GenerationResult(raw_text, parse_numeric_response(raw_text), latency_ms)

    def generate_prompt(self, system: str, user: str) -> GenerationResult:
        """Generate from a generic system/user prompt for agent adapters."""

        return self._generate(system, user)

    def calculate_direct(
        self,
        question: str,
        evidence: Sequence[EvidenceUnit],
        program: str,
    ) -> GenerationResult:
        context = "\n".join(f"[{item.evidence_id}] {item.text}" for item in evidence)
        return self._generate(
            "You are a precise financial calculator. Follow the supplied symbolic "
            "program exactly. Return only the final numeric answer or yes/no.",
            f"Evidence:\n{context}\n\nQuestion: {question}\n"
            f"Program: {program}\n\nReturn only the final numeric answer.",
        )

    def answer_text(
        self,
        question: str,
        evidence: Sequence[ScoredEvidence],
    ) -> GenerationResult:
        context = "\n".join(
            f"[{item.evidence.evidence_id}] {item.evidence.text}" for item in evidence
        )
        return self._generate(
            "Answer only from the supplied evidence. Cite supporting evidence IDs "
            "in square brackets. If evidence is insufficient, say so.",
            f"Evidence:\n{context}\n\nQuestion: {question}",
        )

    def generate_program(
        self,
        question: str,
        evidence: Sequence[ScoredEvidence],
    ) -> GenerationResult:
        context = "\n".join(
            f"[{item.evidence.evidence_id}] {item.evidence.text}" for item in evidence
        )
        result = self._generate(
            "Translate the financial question into exactly one FinQA program and "
            "return only that program, with no prose or code fence. Operation meanings: "
            "add(a,b), subtract(a,b), multiply(a,b), divide(a,b), exp(a,b), and "
            "greater(a,b) only for yes/no comparisons. Use table_sum, table_average, "
            "table_max, or table_min only for a named table row. Use numeric literals, "
            "const_* tokens, and backward # references. Never return Python.",
            "Example:\nEvidence: [text_0] Technology valued at $100 million is "
            "amortized over 5 years.\nQuestion: How much is amortized each year?\n"
            "Program: divide(100, 5)\n\n"
            f"Evidence:\n{context}\n\nQuestion: {question}\nProgram:",
        )
        parsed = parse_program_response(result.raw_text)
        return GenerationResult(result.raw_text, parsed, result.latency_ms)
