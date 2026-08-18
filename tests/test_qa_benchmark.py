from __future__ import annotations

from tools.phase7_benchmark import evaluate


def test_benchmark_reports_unavailable_without_manual_labels() -> None:
    result = evaluate([{"query_id": "q1", "question_type": "OCR", "expected_video_ids": [], "expected_answers": []}], {})
    assert result["metrics"]["availability"] == "unavailable"
    assert result["metrics"]["answer_exact_match"] is None


def test_benchmark_calculates_labelled_fixture_without_claiming_semantic_metric() -> None:
    queries = [{"query_id":"q1","question_type":"OCR","expected_video_ids":["L21_V001"],"expected_frame_range":[4,6],"expected_answers":["Samsung"],"accepted_evidence":["ocr-1"]}]
    observations = {"q1":{"confirmed":True,"review_decision":"confirmed","video_id":"L21_V001","frame_id":5,"answer":"samsung","evidence_ids":["ocr-1"]}}
    result = evaluate(queries, observations)
    assert result["metrics"]["video_accuracy"] == 1.0
    assert result["metrics"]["frame_accuracy"] == 1.0
    assert result["metrics"]["answer_exact_match"] == 0.0
    assert result["metrics"]["normalized_answer_exact_match"] == 1.0
    assert result["metrics"]["unsupported_answer_rate"] == 0.0
