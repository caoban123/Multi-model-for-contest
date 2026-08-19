from __future__ import annotations

from aic_retrieval.trake_candidates import TrakeRetrievalAdapter, group_candidates_by_video, rank_normalized_score
from aic_retrieval.trake_schema import Availability, TrakeEvent, TrakeRequest


def event(order: int, required: bool = True, modalities: tuple[str, ...] = ("clip",)) -> TrakeEvent:
    return TrakeEvent(f"e{order}", order, f"event {order}", f"event {order}", modalities=modalities, required=required)


def result(video: str, keyframe: int, score: float = 100.0) -> dict[str, object]:
    return {"video_id": video, "keyframe_id": keyframe, "frame_idx": keyframe * 10, "pts_time": float(keyframe), "score": score, "keyframe_path": f"{video}/{keyframe}.jpg"}


def test_rank_normalization_does_not_use_raw_score_scale() -> None:
    raw = [result("L21_V001", 1, 0.1), result("L21_V002", 2, 99999.0)]
    pool = TrakeRetrievalAdapter(lambda _event, _size: raw).retrieve_event(event(1), pool_size=30)
    assert pool.candidates[0].local_score == 1.0
    assert pool.candidates[1].local_score == 0.0
    assert pool.candidates[1].raw_scores["score"] == 99999.0


def test_event_pool_limits_frames_per_video_and_preserves_provenance() -> None:
    raw = [result("L21_V001", index) for index in range(12)]
    pool = TrakeRetrievalAdapter(lambda _event, _size: raw).retrieve_event(event(1), pool_size=30, max_per_video=5)
    assert len(pool.candidates) == 5
    assert pool.candidates[0].provenance == ("clip",)


def test_unavailable_modalities_are_explicit_not_negative() -> None:
    pool = TrakeRetrievalAdapter(lambda _event, _size: []).retrieve_event(
        event(1, modalities=("clip", "ocr", "asr")),
        pool_size=30,
        availability={"ocr": Availability.UNAVAILABLE, "asr": Availability.UNAVAILABLE},
    )
    assert "OCR_UNAVAILABLE" in pool.warnings
    assert pool.modality_availability["ocr"] is Availability.UNAVAILABLE


def test_video_grouping_keeps_incomplete_candidates_for_debug() -> None:
    request = TrakeRequest("q", "two", (event(1), event(2)))
    data = {"e1": [result("L21_V001", 1), result("L21_V002", 1)], "e2": [result("L21_V002", 2)]}
    pools = TrakeRetrievalAdapter(lambda ev, _size: data[ev.event_id]).retrieve_request(request, pool_size=30)
    videos = group_candidates_by_video(request, pools, video_pool_size=10)
    assert videos[0].video_id == "L21_V002" and videos[0].complete
    assert videos[1].video_id == "L21_V001" and not videos[1].complete
    assert videos[1].warnings == ("REQUIRED_EVENT_MISSING",)


def test_optional_event_does_not_make_video_incomplete() -> None:
    request = TrakeRequest("q", "optional", (event(1), event(2, required=False)))
    pools = TrakeRetrievalAdapter(lambda ev, _size: [result("L21_V001", 1)] if ev.event_id == "e1" else []).retrieve_request(request, pool_size=30)
    video = group_candidates_by_video(request, pools, video_pool_size=10)[0]
    assert video.complete
    assert video.optional_coverage == 0
