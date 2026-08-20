from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from aic_retrieval.trake_config import RefinementSettings, load_trake_config
from aic_retrieval.trake_refinement import DenseWindowExpander, RefinementConfig, SampledFrame
from aic_retrieval.trake_schema import TrakeRequest
from aic_retrieval.trake_store import TrakeStore
from aic_retrieval.trake_workflow import TrakeWorkflow, validate_chain


ROOT = Path(__file__).parents[1]


class FakeSeekDecoder:
    def sample(self, _path: Path, times: list[float]) -> list[SampledFrame]:
        return [SampledFrame(timestamp + 0.01, timestamp, requested_pts=timestamp, frame_index=None) for timestamp in times]


def test_controlled_v2_e2e_persists_dense_dante_review_and_export(tmp_path: Path) -> None:
    base = load_trake_config(ROOT / "configs" / "phase8_trake_v2.json")
    config = replace(
        base,
        refinement=RefinementSettings(True, 3.0, 2.0, 12.0, 1.0, 24, 1),
    )
    video = tmp_path / "data" / "video" / "L21_V001.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"fixture-video")
    expander = DenseWindowExpander(
        tmp_path,
        {"L21_V001": {"video_path": "data/video/L21_V001.mp4"}},
        tmp_path / "artifacts" / "cache" / "trake",
        lambda text, images: [-abs(float(image) - (2.25 if "first" in text else 8.25)) for image in images],
        decoder=FakeSeekDecoder(),
        config=RefinementConfig(2.0, 3.0, 1.0, 12.0, 24),
        model_fingerprint="clip-fixture",
        config_fingerprint=config.fingerprint,
        index_fingerprint="index-fixture",
        candidates_per_event=8,
    )

    def batch(request: TrakeRequest, _limit: int) -> dict[str, list[dict[str, object]]]:
        return {
            event.event_id: [{
                "video_id": "L21_V001",
                "keyframe_id": event.order,
                "frame_idx": (2 if event.order == 1 else 8) * 25,
                "pts_time": 2.0 if event.order == 1 else 8.0,
                "keyframe_path": f"keyframes/{event.order}.jpg",
                "score": 1.0,
                "evidence": {"fps": 25.0, "metadata": [{"title": "fixture sequence"}]},
                "raw_modality_ranks": {"clip_variant_0": 1},
                "provenance": ["clip", "metadata"],
            }]
            for event in request.events
        }

    store = TrakeStore(tmp_path / "artifacts" / "trake" / "phase8.sqlite3")
    workflow = TrakeWorkflow(
        lambda _event, _limit: [],
        store=store,
        runtime_config=config,
        request_retriever=batch,
        window_expander=expander,
        session_provenance={"index_fingerprint": "index-fixture", "clip_model_fingerprint": "clip-fixture"},
    )

    session = workflow.plan("e2e", "first then second")["state"]["session_id"]
    searched = workflow.search(session, event_pool_size=30, max_per_video=5, video_pool_size=10)
    aligned = workflow.align(session, top_k_per_video=3, top_videos=3)
    chain_payload = aligned["state"]["alignments"][0]["chains"][0]
    chain = workflow.sessions[session].alignments[0].chains[0]

    assert searched["state"]["temporal_feasibility"][0]["feasible"] is True
    assert searched["state"]["candidate_windows"]
    assert {item["sampling"]["stage"] for item in aligned["state"]["dense_expansions"]} == {"coarse", "fine"}
    assert chain.version == "dante-inspired-coarse-fine-v1"
    assert all(entry.candidate.evidence["dense_refinement"] for entry in chain.events if entry.candidate)
    assert validate_chain(workflow.sessions[session].request, chain)[0] is True
    assert chain_payload["score_components"]["final_rerank_version"] == "trake-final-rerank-v1"
    assert aligned["state"]["retrieved_video_id"] == "L21_V001"
    assert aligned["state"]["trake_answer"]["video_id"] == "L21_V001"
    assert len(aligned["state"]["trake_answer"]["frame_ids"]) == 2
    assert len(aligned["state"]["alignments"]) == len(aligned["state"]["alignments"][0]["chains"]) == 1

    reviewed = workflow.review(session, chain.chain_id, "confirmed", "fixture-reviewer")
    exported = workflow.export(session, reviewed["review"]["review_id"])["export"]

    assert exported["human_confirmed"] is True and exported["same_video"] is True and exported["temporally_ordered"] is True
    assert all(event["mapping"]["source"] == "dense_video_seek" for event in exported["events"])
    assert all(event["mapping"]["frame_index_is_estimate"] is True for event in exported["events"])

    reopened = TrakeWorkflow(lambda _event, _limit: [], store=TrakeStore(store.path), runtime_config=config)
    persisted = reopened.get(session)["state"]
    assert persisted["config_fingerprint"] == config.fingerprint
    assert persisted["dense_expansions"] and persisted["diagnostics"]["alignment"]["paths"]
    assert len(reopened.store.reviews(session)) == 1
