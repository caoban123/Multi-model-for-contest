from __future__ import annotations

from pathlib import Path

import numpy as np

from aic_retrieval.search import FrameRef, search_numpy_index, search_numpy_index_batch
from aic_retrieval.trake_candidates import EventCandidatePool, TrakeRetrievalAdapter, select_anchor_event, with_distinctiveness
from aic_retrieval.trake_config import load_trake_config
from aic_retrieval.trake_schema import Availability, TrakeEvent, TrakeRequest
from aic_retrieval.trake_workflow import TrakeWorkflow


ROOT = Path(__file__).parents[1]


def raw(video: str, frame: int, score: float, *, event_id: str = "e1") -> dict[str, object]:
    return {
        "event_id": event_id,
        "video_id": video,
        "keyframe_id": frame,
        "frame_idx": frame * 25,
        "pts_time": float(frame),
        "score": score,
        "raw_modality_scores": {"clip_variant_0": score},
        "raw_modality_ranks": {"clip_variant_0": frame},
        "provenance": ["clip"],
    }


def test_batch_numpy_search_matches_individual_search() -> None:
    index = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]], dtype="float32")
    index /= np.linalg.norm(index, axis=1, keepdims=True)
    refs = [FrameRef("L21_V001", "L21", index + 1, (index + 1) * 25, float(index), 25.0, None) for index in range(3)]
    queries = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="float32")

    batched = search_numpy_index_batch(index, refs, queries, 2)

    assert [[row.keyframe_id for row in rows] for rows in batched] == [
        [row.keyframe_id for row in search_numpy_index(index, refs, query, 2)] for query in queries
    ]


def test_adaptive_topk_uses_max_for_generic_and_min_for_distinctive_structured_event() -> None:
    config = load_trake_config(ROOT / "configs" / "phase8_trake_v2.json")
    requested_limits: list[int] = []

    def batch(request: TrakeRequest, limit: int) -> dict[str, list[dict[str, object]]]:
        requested_limits.append(limit)
        return {
            "e1": [raw("L21_V001", 1, 1.0), raw("L21_V002", 2, 0.99)],
            "e2": [raw("L21_V001", 3, 1.0, event_id="e2"), raw("L21_V002", 4, 0.5, event_id="e2")],
        }

    workflow = TrakeWorkflow(lambda _event, _limit: [], runtime_config=config, request_retriever=batch)
    planned = workflow.plan("q", "person then a distinctive red ball on the football stadium field")
    searched = workflow.search(planned["state"]["session_id"])
    diagnostics = searched["state"]["diagnostics"]["retrieval"]

    assert requested_limits == [config.retrieval.topk_max]
    assert diagnostics["topk_by_event"]["e1"] == config.retrieval.topk_max
    assert diagnostics["topk_by_event"]["e2"] == config.retrieval.topk_min
    assert searched["state"]["pools"][0]["candidates"][0]["raw_scores"]["clip_variant_0"] == 1.0
    assert searched["state"]["pools"][0]["candidates"][0]["raw_ranks"]["clip_variant_0"] == 1


def test_distinctiveness_selects_evidence_driven_anchor_not_fixed_position() -> None:
    config = load_trake_config(ROOT / "configs" / "phase8_trake_v2.json")
    events = (
        TrakeEvent("e1", 1, "person", "person"),
        TrakeEvent("e2", 2, "walking", "walking"),
        TrakeEvent("e3", 3, "player holding red trophy", "player holding red trophy", modalities=("clip", "object", "attribute"), object_constraints=({"label": "trophy"},), attribute_constraints=({"color": "red"},)),
    )
    request = TrakeRequest("q", "three", events)
    adapter_data = {
        "e1": [raw("L21_V001", 1, 1.0), raw("L21_V002", 2, 0.99)],
        "e2": [raw("L21_V001", 3, 1.0, event_id="e2"), raw("L21_V002", 4, 0.99, event_id="e2")],
        "e3": [raw("L21_V003", 5, 1.0, event_id="e3"), raw("L21_V003", 6, 0.2, event_id="e3")],
    }
    adapter = TrakeRetrievalAdapter(lambda event, _limit: adapter_data[event.event_id])
    pools = tuple(
        with_distinctiveness(pool, config.distinctiveness)
        for pool in adapter.retrieve_request(request, pool_size=30, max_per_video=5, availability={"clip": Availability.AVAILABLE})
    )

    anchor, diagnostics = select_anchor_event(request, pools)

    assert anchor == "e3"
    assert diagnostics["scores"]["e3"] > diagnostics["scores"]["e2"]
    assert all(isinstance(pool, EventCandidatePool) for pool in pools)
