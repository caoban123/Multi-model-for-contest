from __future__ import annotations

from dataclasses import dataclass

from aic_retrieval.hybrid_engine import HybridRetrievalEngine
from aic_retrieval.hybrid_query_planner import HybridQueryPlan
from aic_retrieval.retrievers import RetrievalHit, RetrievalRequest, RetrieverHealth
from aic_retrieval.rrf_fusion import RrfFusionConfig


@dataclass
class FixtureRetriever:
    name: str
    video_id: str
    seen_query: str = ""
    seen_filters: dict | None = None

    def health(self) -> RetrieverHealth:
        return RetrieverHealth(self.name, "READY")

    def search(self, request: RetrievalRequest) -> list[RetrievalHit]:
        self.seen_query = request.query_text
        self.seen_filters = dict(request.filters)
        return [RetrievalHit(self.name, 1, 1.0, self.video_id, self.name, document_id=f"{self.name}:1")]


def test_engine_uses_per_channel_queries_and_lazy_factories() -> None:
    created = {}

    def factory(name: str, video_id: str):
        def make():
            created[name] = FixtureRetriever(name, video_id)
            return created[name]
        return make

    engine = HybridRetrievalEngine(
        {"clip": factory("clip", "V1"), "bm25": factory("bm25", "V2")},
        RrfFusionConfig(candidate_pool=10),
    )
    plan = HybridQueryPlan(
        "áo đỏ có chữ HTV", "a red shirt with HTV text", "áo đỏ có chữ HTV", "HTV",
        "mixed", ("clip", "bm25"), "clip_bm25_rrf", "rrf", ("mixed evidence",),
        source_filters={"bm25": ("metadata",)},
    )
    response = engine.search("q1", plan, top_k=5)
    assert created["clip"].seen_query == "a red shirt with HTV text"
    assert created["bm25"].seen_query == "HTV"
    assert created["bm25"].seen_filters == {"source_types": ("metadata",)}
    assert response["profile"] == "clip_bm25_rrf"
    assert response["channel_hit_counts"] == {"clip": 1, "bm25": 1}
    assert len(response["results"]) == 2


def test_engine_fail_open_records_missing_factory() -> None:
    plan = HybridQueryPlan(
        "query", "query", "query", "query", "semantic_text", ("bge",), "bge", "none", (),
    )
    response = HybridRetrievalEngine({}).search("q", plan)
    assert response["results"] == []
    assert "bge" in response["failures"]


def test_engine_fuses_multiple_clip_queries_before_outer_fusion() -> None:
    class VariantClipRetriever:
        name = "clip"

        def __init__(self) -> None:
            self.queries: list[str] = []

        def health(self) -> RetrieverHealth:
            return RetrieverHealth("clip", "READY")

        def search(self, request: RetrievalRequest) -> list[RetrievalHit]:
            self.queries.append(request.query_text)
            video_id = "L21_V001" if "inside" in request.query_text else "L21_V002"
            return [RetrievalHit("clip", 1, 0.9, video_id, "clip", document_id=f"clip:{video_id}:1")]

    retriever = VariantClipRetriever()
    engine = HybridRetrievalEngine({"clip": lambda: retriever}, RrfFusionConfig(candidate_pool=10))
    plan = HybridQueryPlan(
        "long query",
        "view from inside a self-driving car",
        "long query",
        "long query",
        "visual",
        ("clip",),
        "clip",
        "none",
        ("long visual narrative",),
        visual_clip_queries_en=(
            "view from inside a self-driving car",
            "a white car turning left",
        ),
    )

    response = engine.search("q-multi", plan, top_k=5)

    assert retriever.queries == [
        "view from inside a self-driving car",
        "a white car turning left",
    ]
    assert response["fusion_method"] == "multi_query_rrf_with_variant_rescue"
    assert response["channel_postprocessing"]["clip"]["variant_hit_counts"] == {
        "clip:q1": 1,
        "clip:q2": 1,
    }
    assert response["channel_postprocessing"]["clip"]["variant_rescue"]["enabled"] is True
    assert {result["video_id"] for result in response["results"]} == {"L21_V001", "L21_V002"}
    clip_evidence = response["results"][0]["provenance"]["evidence"]["clip"]
    assert clip_evidence["provenance"]["multi_query"] is True


def test_multi_query_review_window_rescues_top_video_from_each_lane() -> None:
    class LaneClipRetriever:
        name = "clip"

        def health(self) -> RetrieverHealth:
            return RetrieverHealth("clip", "READY")

        def search(self, request: RetrievalRequest) -> list[RetrievalHit]:
            prefix = "A" if request.query_text == "event one" else "B"
            rows = [
                RetrievalHit("clip", 1, 0.99, f"{prefix}_ONLY", "clip", document_id=f"{prefix}:only")
            ]
            rows.extend(
                RetrievalHit("clip", rank, 0.9 - rank / 100, f"SHARED_{rank}", "clip", document_id=f"{prefix}:{rank}")
                for rank in range(2, 8)
            )
            return rows

    engine = HybridRetrievalEngine({"clip": LaneClipRetriever}, RrfFusionConfig(candidate_pool=20))
    plan = HybridQueryPlan(
        "two events", "event one", "two events", "two events", "visual", ("clip",), "clip", "none", (),
        visual_clip_queries_en=("event one", "event two"),
    )

    response = engine.search("q-lanes", plan, top_k=6)
    videos = [row["video_id"] for row in response["results"]]
    trace = response["channel_postprocessing"]["clip"]["variant_rescue"]

    assert "A_ONLY" in videos
    assert "B_ONLY" in videos
    assert trace["policy"] == "rrf-head-sequence-then-round-robin-v2"
    assert trace["lanes"][0]["video_ids"][0] == "A_ONLY"
    assert trace["lanes"][1]["video_ids"][0] == "B_ONLY"


def test_temporal_sequence_rescue_prefers_ordered_events_over_higher_reverse_hits() -> None:
    class TemporalClipRetriever:
        name = "clip"

        def health(self) -> RetrieverHealth:
            return RetrieverHealth("clip", "READY")

        def search(self, request: RetrievalRequest) -> list[RetrievalHit]:
            if request.query_text == "broad scene":
                return [
                    RetrievalHit("clip", rank, 1.0 - rank / 100, video_id, "clip", f"broad:{video_id}", pts_time=float(rank))
                    for rank, video_id in enumerate(("SHARED_A", "SHARED_B", "SHARED_C", "TARGET", "REVERSE"), 1)
                ]
            event_index = {"event one": 0, "event two": 1, "event three": 2}[request.query_text]
            target_times = (10.0, 20.0, 30.0)
            reverse_times = (30.0, 20.0, 10.0)
            shared_times = (90.0, 60.0, 30.0)
            return [
                RetrievalHit("clip", 1, 0.99, "REVERSE", "clip", f"reverse:{event_index}", keyframe_id=event_index + 1, pts_time=reverse_times[event_index]),
                RetrievalHit("clip", 2, 0.98, "SHARED_A", "clip", f"shared-a:{event_index}", pts_time=shared_times[event_index]),
                RetrievalHit("clip", 3, 0.97, "SHARED_B", "clip", f"shared-b:{event_index}", pts_time=shared_times[event_index]),
                RetrievalHit("clip", 4, 0.96, "TARGET", "clip", f"target:{event_index}", keyframe_id=event_index + 10, pts_time=target_times[event_index]),
            ]

    engine = HybridRetrievalEngine({"clip": TemporalClipRetriever}, RrfFusionConfig(candidate_pool=20))
    plan = HybridQueryPlan(
        "ordered story", "broad scene", "ordered story", "ordered story", "visual", ("clip",), "clip", "none", (),
        visual_clip_queries_en=("event one", "event two", "event three"),
    )

    response = engine.search("q-sequence", plan, top_k=6)
    trace = response["channel_postprocessing"]["clip"]["variant_rescue"]["temporal_sequence"]

    assert trace["enabled"] is True
    assert trace["event_channels"] == ["clip:q2", "clip:q3", "clip:q4"]
    assert trace["candidates"][0]["video_id"] == "TARGET"
    assert trace["candidates"][0]["matched_events"] == 3
    assert [frame["pts_time"] for frame in trace["candidates"][0]["frames"]] == [10.0, 20.0, 30.0]
    assert "REVERSE" not in {candidate["video_id"] for candidate in trace["candidates"]}
    target = next(result for result in response["results"] if result["video_id"] == "TARGET")
    sequence = target["provenance"]["evidence"]["clip"]["provenance"]["temporal_sequence"]
    assert sequence["matched_events"] == 3


def test_engine_suppresses_repeated_ocr_before_video_fusion() -> None:
    class RepeatedOcrRetriever:
        name = "bm25"

        def health(self) -> RetrieverHealth:
            return RetrieverHealth("bm25", "READY")

        def search(self, request: RetrievalRequest) -> list[RetrievalHit]:
            rows = [
                RetrievalHit("bm25", rank, 20.0 - rank, "L21_V001", "ocr", f"ocr:logo:{rank}", matched_text="HTV")
                for rank in range(1, 8)
            ]
            rows.append(RetrievalHit("bm25", 8, 11.0, "L21_V002", "ocr", "ocr:answer", matched_text="123 đường A"))
            return rows

    engine = HybridRetrievalEngine({"bm25": RepeatedOcrRetriever}, RrfFusionConfig(candidate_pool=10))
    plan = HybridQueryPlan(
        "địa chỉ",
        "address sign",
        "địa chỉ",
        "địa chỉ",
        "lexical_text",
        ("bm25",),
        "bm25",
        "none",
        ("exact address",),
        source_filters={"bm25": ("ocr",)},
    )

    response = engine.search("q-ocr", plan, top_k=5)

    assert {result["video_id"] for result in response["results"]} == {"L21_V001", "L21_V002"}
    stats = response["channel_postprocessing"]["bm25"]
    assert stats["duplicate_ocr_removed"] == 6


def test_candidate_local_text_search_keeps_distinct_ocr_frames() -> None:
    class LocalOcrRetriever:
        name = "bm25"

        def health(self) -> RetrieverHealth:
            return RetrieverHealth("bm25", "READY")

        def search(self, request: RetrievalRequest) -> list[RetrievalHit]:
            return [
                RetrievalHit(
                    "bm25",
                    rank,
                    20.0 - rank,
                    "L21_V001",
                    "ocr",
                    f"ocr:{rank}",
                    matched_text=f"distinct text {rank}",
                )
                for rank in range(1, 9)
            ]

    engine = HybridRetrievalEngine({"bm25": LocalOcrRetriever}, RrfFusionConfig(candidate_pool=10))
    hits = engine.search_channel(
        "bm25",
        RetrievalRequest(
            "q-local",
            "address",
            top_k=8,
            filters={"source_types": ("ocr",), "video_ids": ("L21_V001",)},
        ),
    )

    assert len(hits) == 8
