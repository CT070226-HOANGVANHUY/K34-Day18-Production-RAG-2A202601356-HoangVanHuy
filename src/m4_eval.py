from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json, math, re
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    if not (len(questions) == len(answers) == len(contexts) == len(ground_truths)):
        raise ValueError("questions, answers, contexts and ground_truths must have equal length")

    # RAGAS uses an LLM for all four metrics. Use it when a real key is set.
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if api_key and api_key != "sk-...":
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import (
                answer_relevancy, context_precision, context_recall, faithfulness,
            )

            dataset = Dataset.from_dict({
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            })
            result = evaluate(dataset, metrics=[
                faithfulness, answer_relevancy, context_precision, context_recall,
            ])
            frame = result.to_pandas()
            per_question = []
            for index, row in frame.iterrows():
                values = {
                    name: _safe_float(row.get(name, 0.0)) for name in metric_names
                }
                per_question.append(EvalResult(
                    question=questions[index], answer=answers[index],
                    contexts=contexts[index], ground_truth=ground_truths[index], **values,
                ))
            return {
                **{name: _mean(getattr(item, name) for item in per_question) for name in metric_names},
                "per_question": per_question,
            }
        except Exception as exc:
            print(f"  ⚠️  RAGAS evaluation failed, using local fallback: {exc}")

    per_question = [_heuristic_eval(q, a, c, gt) for q, a, c, gt in zip(
        questions, answers, contexts, ground_truths
    )]
    return {
        **{name: _mean(getattr(item, name) for item in per_question) for name in metric_names},
        "per_question": per_question,
    }


def _safe_float(value) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _mean(values) -> float:
    values = list(values)
    return round(sum(values) / len(values), 6) if values else 0.0


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[\wÀ-ỹ]+", (text or "").lower()))


def _overlap(left: str, right: str) -> float:
    a, b = _terms(left), _terms(right)
    return len(a & b) / max(len(a), 1)


def _heuristic_eval(question: str, answer: str, contexts: list[str], ground_truth: str) -> EvalResult:
    context_text = " ".join(contexts)
    answer_terms = _terms(answer)
    context_terms = _terms(context_text)
    gt_terms = _terms(ground_truth)
    answer_relevancy = _overlap(question, answer)
    faithfulness = len(answer_terms & context_terms) / max(len(answer_terms), 1)
    context_recall = len(gt_terms & context_terms) / max(len(gt_terms), 1)
    context_precision = (
        sum(_overlap(context, question) for context in contexts) / len(contexts)
        if contexts else 0.0
    )
    return EvalResult(
        question, answer, contexts, ground_truth,
        round(min(faithfulness, 1.0), 6),
        round(min(answer_relevancy, 1.0), 6),
        round(min(context_precision, 1.0), 6),
        round(min(context_recall, 1.0), 6),
    )


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("LLM có dấu hiệu trả lời ngoài context", "Siết prompt và chỉ cho phép trích dẫn từ context"),
        "context_recall": ("Thiếu chunk chứa thông tin cần thiết", "Cải thiện chunking hoặc bổ sung BM25 vào hybrid search"),
        "context_precision": ("Context chứa quá nhiều nội dung không liên quan", "Dùng reranking hoặc lọc metadata"),
        "answer_relevancy": ("Câu trả lời chưa bám sát câu hỏi", "Cải thiện prompt trả lời và kiểm tra query rewriting"),
    }
    metric_names = list(diagnostic_tree)
    ranked = []
    for item in eval_results:
        values = {name: _safe_float(getattr(item, name, 0.0)) for name in metric_names}
        average = _mean(values.values())
        worst_metric = min(metric_names, key=lambda name: values[name])
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        ranked.append({
            "question": item.question,
            "worst_metric": worst_metric,
            "score": values[worst_metric],
            "average_score": average,
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })
    ranked.sort(key=lambda item: item["average_score"])
    return ranked[:max(bottom_n, 0)]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
