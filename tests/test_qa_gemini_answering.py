from __future__ import annotations

from aic_retrieval.qa_evidence import build_evidence_pack
from aic_retrieval.qa_gemini_answering import GeminiEvidenceAnswerer, gemini_qa_payload, parse_gemini_qa_response
from aic_retrieval.qa_question_router import RuleBasedQuestionRouter
from aic_retrieval.qa_schema import AvailabilityStatus, ConfidenceState, EvidenceModality, QaRequest
from aic_retrieval.translation import TranslationConfig


def _pack():
    response = {
        "video_results": [
            {
                "video_id": "L21_V001",
                "frames": [
                    {
                        "video_id": "L21_V001",
                        "keyframe_id": 5,
                        "frame_idx": 125,
                        "pts_time": 5.0,
                        "evidence": {"asr": [{"matched_text": "Người dẫn chương trình đang nói về tin tức buổi sáng."}], "ocr": [], "object": [], "attribute": []},
                    }
                ],
            }
        ]
    }
    return build_evidence_pack(
        QaRequest("q1", "news presenter", "Người đó đang làm gì?"),
        response,
        modality_availability={EvidenceModality.ASR: AvailabilityStatus.AVAILABLE},
    )


def test_gemini_qa_payload_requests_json_from_selected_evidence_only() -> None:
    pack = _pack()
    route = RuleBasedQuestionRouter().route("Người đó đang làm gì?")
    payload = gemini_qa_payload(pack, route, [pack.evidence_refs[0]])
    prompt = payload["contents"][0]["parts"][0]["text"]
    assert payload["generationConfig"]["responseMimeType"] == "application/json"
    assert "Answer using only the selected evidence" in prompt
    assert pack.evidence_refs[0].evidence_id in prompt


def test_parse_gemini_qa_response_accepts_strict_json() -> None:
    payload = {"candidates": [{"content": {"parts": [{"text": '{"answer":"Đang dẫn chương trình","evidence_ids":["e1"],"warnings":[]}'}]}}]}
    assert parse_gemini_qa_response(payload)["answer"] == "Đang dẫn chương trình"


def test_gemini_answerer_drafts_reviewable_answer_from_selected_evidence(monkeypatch) -> None:
    pack = _pack()
    route = RuleBasedQuestionRouter().route("Người đó đang làm gì?")
    evidence_id = pack.evidence_refs[0].evidence_id

    def fake_call(config, payload):
        return {"candidates": [{"content": {"parts": [{"text": f'{{"answer":"Đang dẫn chương trình","evidence_ids":["{evidence_id}"],"warnings":["HUMAN_CONFIRMATION_REQUIRED"]}}'}]}}]}

    monkeypatch.setattr("aic_retrieval.qa_gemini_answering._call_gemini", fake_call)
    draft = GeminiEvidenceAnswerer(TranslationConfig(api_key="test", provider="gemini")).draft(pack, route, (evidence_id,))
    assert draft.raw_answer == "Đang dẫn chương trình"
    assert draft.confidence_state is ConfidenceState.REVIEW_REQUIRED
    assert draft.generation_method == "gemini_vlm_evidence"
    assert draft.evidence_refs == (evidence_id,)


def test_gemini_payload_attaches_related_keyframe_image_for_text_evidence(tmp_path) -> None:
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"\xff\xd8\xff\xd9")
    response = {
        "video_results": [
            {
                "video_id": "L21_V001",
                "frames": [
                    {
                        "video_id": "L21_V001",
                        "keyframe_id": 5,
                        "frame_idx": 125,
                        "pts_time": 5.0,
                        "keyframe_path": str(image_path),
                        "evidence": {"asr": [{"matched_text": "A presenter is speaking."}], "ocr": [], "object": [], "attribute": []},
                    }
                ],
            }
        ]
    }
    pack = build_evidence_pack(QaRequest("q-img", "presenter", "Người đó đang làm gì?"), response, modality_availability={EvidenceModality.ASR: AvailabilityStatus.AVAILABLE})
    route = RuleBasedQuestionRouter().route("Người đó đang làm gì?")
    selected = [ref for ref in pack.evidence_refs if ref.modality is EvidenceModality.ASR]
    payload = gemini_qa_payload(pack, route, selected)
    parts = payload["contents"][0]["parts"]
    assert any("inline_data" in part for part in parts)
    assert "image(s) are attached" in parts[0]["text"]
