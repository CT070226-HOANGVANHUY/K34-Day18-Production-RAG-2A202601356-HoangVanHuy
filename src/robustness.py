from __future__ import annotations

"""Query planning and evidence guardrails for adversarial RAG cases.

These rules are intentionally deterministic. They complement, rather than
replace, the semantic encoder and LLM: retrieval gets more evidence for
multi-hop questions and the final context is biased toward the right version,
numbers, and negation markers.
"""

import re
from dataclasses import dataclass


NEGATION_MARKERS = (
    "không", "chưa", "tuyệt đối không", "không được", "cấm", "ngoại trừ",
)
CURRENT_MARKERS = ("hiện hành", "mới nhất", "đang áp dụng", "v2024", "v2.0", "v2")
OLD_MARKERS = ("cũ", "đã bị thay thế", "v2023", "v1.0", "v1")
OUT_OF_SCOPE_MARKERS = ("cổ phiếu", "giá thị trường", "thời tiết", "bitcoin", "tỷ số bóng đá")


@dataclass(frozen=True)
class QueryPlan:
    original: str
    queries: tuple[str, ...]
    is_multi_hop: bool
    has_negation: bool
    prefer_current_version: bool
    numbers: tuple[str, ...]


def _normalized(text: str) -> str:
    return " ".join((text or "").lower().split())


def extract_numbers(text: str) -> tuple[str, ...]:
    """Return normalized numeric expressions, preserving units."""
    values = []
    for match in re.finditer(r"\b(\d+(?:[.,]\d+)?)\s*(triệu|ngày|năm|%|vnđ|đồng)?", text.lower()):
        number = match.group(1).replace(",", ".")
        unit = match.group(2) or ""
        values.append(f"{number}{unit}")
    return tuple(values)


def _is_multi_hop(query: str) -> bool:
    q = _normalized(query)
    return bool(
        re.search(r"\bbao nhiêu.*\bvà\b", q)
        or re.search(r"\bai phê duyệt.*\bvà\b", q)
        or re.search(r"\bcần gì.*\bvà\b", q)
        or "đồng thời" in q
        or ("nghỉ" in q and "lương" in q)
    )


def build_query_plan(query: str) -> QueryPlan:
    """Create retrieval variants for multi-hop and ambiguous policy queries."""
    original = " ".join((query or "").strip().split())
    q = _normalized(original)
    multi_hop = _is_multi_hop(original)
    has_negation = any(marker in q for marker in NEGATION_MARKERS)
    explicit_current = any(marker in q for marker in CURRENT_MARKERS)
    explicit_old = any(marker in q for marker in OLD_MARKERS)
    # A bare policy question should prefer the current version. Explicit old
    # wording wins unless the same query explicitly asks for the current value.
    prefer_current = explicit_current or not explicit_old
    numbers = extract_numbers(original)

    variants = [original] if original else []
    if multi_hop:
        if "nghỉ" in q and "lương" in q:
            variants.extend([
                f"{original} chính sách nghỉ phép thâm niên",
                f"{original} bảng lương cấp bậc",
            ])
        else:
            clauses = [part.strip(" ,") for part in re.split(r"\s+(?:và|đồng thời)\s+", original, flags=re.I)]
            variants.extend(part for part in clauses if len(part) >= 12)

    # Preserve monetary thresholds in a form used by policy documents.
    for match in re.finditer(r"(\d+(?:[.,]\d+)?)\s*triệu", q):
        amount = float(match.group(1).replace(",", "."))
        variants.append(f"{int(amount * 1_000_000):,}".replace(",", "."))
        if amount > 50:
            variants.append("trên 50.000.000 CEO phê duyệt")
        elif 5 <= amount <= 50:
            variants.append("5.000.000 đến 50.000.000 Director phê duyệt")

    # De-duplicate while preserving the original query first.
    unique = tuple(dict.fromkeys(v for v in variants if v))
    return QueryPlan(original, unique, multi_hop, has_negation, prefer_current, numbers)


def _source(result) -> str:
    metadata = getattr(result, "metadata", None)
    if metadata is None and isinstance(result, dict):
        metadata = result.get("metadata", {})
    return str((metadata or {}).get("source", "")).lower()


def evidence_score(query: str, text: str, result=None) -> float:
    """Score evidence signals that generic semantic ranking can miss."""
    plan = build_query_plan(query)
    haystack = _normalized(text)
    score = 0.0

    if plan.has_negation and any(marker in haystack for marker in NEGATION_MARKERS):
        score += 3.0
    if plan.prefer_current_version:
        source = _source(result)
        if re.search(r"v(?:2024|2)(?:\D|$)", source):
            score += 2.5
        if re.search(r"v(?:2023|1)(?:\D|$)", source):
            score -= 2.5

    query_terms = set(re.findall(r"[\wÀ-ỹ]+", _normalized(query)))
    text_terms = set(re.findall(r"[\wÀ-ỹ]+", haystack))
    score += min(len(query_terms & text_terms) / max(len(query_terms), 1), 1.0)

    for value in plan.numbers:
        if value in haystack or value.replace("triệu", ".000.000") in haystack:
            score += 1.5
    return score


def prioritize_results(query: str, results: list) -> list:
    """Stable evidence-aware ordering for SearchResult-like objects."""
    def text_of(result) -> str:
        return getattr(result, "text", result.get("text", "") if isinstance(result, dict) else "")

    def score_of(result) -> float:
        return getattr(result, "score", result.get("score", 0.0) if isinstance(result, dict) else 0.0)

    return sorted(
        results,
        key=lambda result: (evidence_score(query, text_of(result), result), score_of(result)),
        reverse=True,
    )


def should_abstain(query: str, contexts: list[str]) -> bool:
    """Prevent confident answers when the corpus cannot support the query."""
    q = _normalized(query)
    joined = _normalized(" ".join(contexts))
    if not contexts:
        return True
    if any(marker in q for marker in OUT_OF_SCOPE_MARKERS):
        return True
    # The corpus contains these PDFs only as scans. Do not answer from an
    # unrelated chunk when the expected source text is absent.
    if "nghị định" in q and "nghị định" not in joined:
        return True
    return False
