"""Regression tests for the known RAG failure modes.

These tests are deliberately model-free. They validate the deterministic
guardrails before spending time or API budget on a full retrieval evaluation.
"""

import json
from pathlib import Path

from src.robustness import build_query_plan, extract_numbers, prioritize_results, should_abstain


ROOT = Path(__file__).resolve().parents[1]


def load_adversarial_cases():
    with open(ROOT / "adversarial_test_set.json", encoding="utf-8") as handle:
        return json.load(handle)


def test_adversarial_dataset_has_all_required_fields():
    cases = load_adversarial_cases()
    assert len(cases) >= 15
    assert {case["category"] for case in cases} >= {
        "multi_hop", "negation", "version_conflict", "numeric_threshold",
        "out_of_scope", "ocr_gap",
    }
    for case in cases:
        assert case["id"] and case["question"]
        assert case["expected_signals"]
        assert case["failure_target"]


def test_multi_hop_query_is_decomposed():
    plan = build_query_plan(
        "Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?"
    )
    assert plan.is_multi_hop
    assert len(plan.queries) >= 3
    assert any("nghỉ phép" in query for query in plan.queries)
    assert any("lương" in query for query in plan.queries)


def test_numeric_threshold_query_preserves_amount_and_boundary():
    plan = build_query_plan("Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?")
    assert "55triệu" in plan.numbers
    assert any("55.000.000" in query for query in plan.queries)
    assert any("trên 50.000.000" in query for query in plan.queries)


def test_numeric_extraction_keeps_days_and_money_separate():
    assert extract_numbers("Tạm ứng 15 triệu, thanh toán sau 20 ngày") == (
        "15triệu", "20ngày"
    )


def test_negation_evidence_is_prioritized():
    query = "Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không?"
    results = [
        {"text": "Nhân viên chính thức được hưởng bảo hiểm sức khỏe PVI.", "score": 0.9, "metadata": {}},
        {"text": "KHÔNG. Nhân viên thử việc chưa được hưởng gói PVI.", "score": 0.7, "metadata": {}},
    ]
    ranked = prioritize_results(query, results)
    assert ranked[0]["text"].startswith("KHÔNG")


def test_current_policy_beats_superseded_version():
    query = "Bao lâu phải đổi mật khẩu một lần?"
    results = [
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.95, "metadata": {"source": "mat_khau_v1.md"}},
        {"text": "Mật khẩu thay đổi mỗi 120 ngày và cần MFA.", "score": 0.75, "metadata": {"source": "mat_khau_v2.md"}},
    ]
    ranked = prioritize_results(query, results)
    assert "120 ngày" in ranked[0]["text"]


def test_explicit_old_version_does_not_force_current_policy():
    plan = build_query_plan("Chính sách mật khẩu cũ v1.0 yêu cầu bao lâu?")
    assert not plan.prefer_current_version


def test_out_of_scope_query_is_not_expanded_into_fake_evidence():
    plan = build_query_plan("Giá cổ phiếu của công ty hôm nay là bao nhiêu?")
    assert not plan.is_multi_hop
    assert plan.queries == (plan.original,)


def test_out_of_scope_and_scan_gap_abstain():
    assert should_abstain("Giá cổ phiếu của công ty hôm nay là bao nhiêu?", ["Quy định nghỉ phép năm."])
    assert should_abstain("Nghị định 13 quy định mức phạt nào?", ["Quy định nghỉ phép năm."])
    assert not should_abstain("Bao lâu phải đổi mật khẩu?", ["Mật khẩu phải đổi mỗi 120 ngày."])


def test_corpus_contains_expected_signals_for_adversarial_cases():
    corpus = " ".join(
        path.read_text(encoding="utf-8", errors="ignore").lower()
        for path in (ROOT / "data").glob("*.md")
    )
    assert "mật khẩu" in corpus
    assert "mfa" in corpus
    assert "không được" in corpus
    assert "50.000.000" in corpus
