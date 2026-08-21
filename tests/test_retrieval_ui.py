import json
from pathlib import Path

import numpy as np
import pytest

from aic_retrieval.hybrid_query_planner import HybridQueryPlan, HybridQueryPlanner
from aic_retrieval.metadata_search import load_metadata_documents
from aic_retrieval.retrievers import RetrievalHit
from aic_retrieval.retrieval_ui import RetrievalUiConfig, RetrievalUiService, parse_int
from aic_retrieval.search import FrameRef, SearchResult, aggregate_results_by_video
from aic_retrieval.qa_workflow import QaWorkflow
from aic_retrieval.trake_config import TrakeRuntimeConfig
from aic_retrieval.trake_schema import TrakeEvent, TrakeRequest
from aic_retrieval.translation import TranslationConfig


def make_service(tmp_path: Path) -> RetrievalUiService:
    media_dir = tmp_path / "data" / "media-info"
    media_dir.mkdir(parents=True)
    video_dir = tmp_path / "data" / "videos"
    video_dir.mkdir(parents=True)
    (video_dir / "L21_V001.mp4").write_bytes(b"fake video")
    (media_dir / "L21_V001.json").write_text(
        json.dumps(
            {
                "title": "60 Giây Sáng",
                "author": "60 Giây Official",
                "publish_date": "01/08/2024",
                "keywords": ["tin tức", "HTV"],
                "watch_url": "https://example.com",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    service = RetrievalUiService.__new__(RetrievalUiService)
    service.config = RetrievalUiConfig(
        repo_root=tmp_path,
        registry_path=tmp_path / "registry.json",
        index_dir=tmp_path / "index",
        metadata_dir=tmp_path / "data" / "media-info",
        static_dir=tmp_path / "web",
        groups={"L21"},
    )
    service.assets_by_video = {
        "L21_V001": {
            "media_info_path": "data/media-info/L21_V001.json",
            "video_path": "data/videos/L21_V001.mp4",
        }
    }
    service.refs_by_video = {
        "L21_V001": [
            FrameRef("L21_V001", "L21", 1, 0, 0.0, 30.0, "data/keyframes/L21_V001/001.jpg"),
            FrameRef("L21_V001", "L21", 2, 30, 1.0, 30.0, "data/keyframes/L21_V001/002.jpg"),
            FrameRef("L21_V001", "L21", 3, 60, 2.0, 30.0, "data/keyframes/L21_V001/003.jpg"),
            FrameRef("L21_V001", "L21", 4, 90, 3.0, 30.0, "data/keyframes/L21_V001/004.jpg"),
        ]
    }
    service.metadata_by_video = {}
    return service


def test_enrich_result_adds_metadata_and_image_url(tmp_path: Path) -> None:
    service = make_service(tmp_path)

    result = service.enrich_result(
        {
            "rank": 1,
            "score": 1.0,
            "video_id": "L21_V001",
            "keyframe_id": 1,
            "keyframe_path": "data/keyframes/L21_V001/001.jpg",
        }
    )

    assert result["image_url"] == "/keyframe?path=data/keyframes/L21_V001/001.jpg"
    assert result["video_url"] == "/video?video_id=L21_V001"
    assert result["metadata"]["title"] == "60 Giây Sáng"
    assert result["metadata"]["keywords"] == ["tin tức", "HTV"]


def test_resolve_keyframe_path_rejects_outside_repo(tmp_path: Path) -> None:
    service = make_service(tmp_path)

    with pytest.raises(PermissionError):
        service.resolve_keyframe_path("../outside.jpg")


def test_resolve_keyframe_path_accepts_existing_repo_file(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    image = tmp_path / "data" / "keyframes" / "L21_V001" / "001.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"fake")

    assert service.resolve_keyframe_path("data/keyframes/L21_V001/001.jpg") == image.resolve()


def test_resolve_media_accepts_configured_external_data_root(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    external_data = tmp_path / "external-data"
    repo_root.mkdir()
    image = external_data / "keyframes" / "L21_V001" / "001.jpg"
    video = external_data / "videos" / "L21_V001.mp4"
    image.parent.mkdir(parents=True)
    video.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    video.write_bytes(b"video")
    service = make_service(repo_root)
    service.registry = {"data_root": str(external_data)}
    service.assets_by_video["L21_V001"]["video_path"] = str(video)

    assert service.resolve_keyframe_path(str(image)) == image.resolve()
    assert service.resolve_video_path("L21_V001") == video.resolve()


def test_resolve_video_path_accepts_registered_repo_video(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    video = tmp_path / "data" / "videos" / "L21_V001.mp4"

    assert service.video_url("L21_V001") == "/video?video_id=L21_V001"
    assert service.resolve_video_path("L21_V001") == video.resolve()


def test_resolve_video_path_rejects_missing_or_outside_video(tmp_path: Path) -> None:
    service = make_service(tmp_path)

    with pytest.raises(FileNotFoundError):
        service.resolve_video_path("L21_V999")

    service.assets_by_video["L21_BAD"] = {"video_path": "../outside.mp4"}
    with pytest.raises(PermissionError):
        service.resolve_video_path("L21_BAD")


def test_parse_int_falls_back_for_bad_values() -> None:
    assert parse_int("12", 1) == 12
    assert parse_int("bad", 7) == 7


def test_service_translate_uses_configured_translator(tmp_path: Path) -> None:
    service = make_service(tmp_path)

    class FakeTranslator:
        config = type("Config", (), {"model": "fake-model", "provider": "gemini"})()

        def translate_vi_to_en(self, text: str) -> str:
            assert text == "người cầm điện thoại"
            return "a person holding a phone"

    service.translator = FakeTranslator()

    assert service.translate(" người cầm điện thoại ") == {
        "source_text": "người cầm điện thoại",
        "translated_text": "a person holding a phone",
        "provider": "gemini",
        "model": "fake-model",
    }


def test_qa_prepare_uses_retrieval_query_but_keeps_original_event_query(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    seen_queries: list[str] = []

    def fake_search(query: str, **kwargs):
        seen_queries.append(query)
        return {
            "mode": "visual",
            "query": query,
            "groups": ["L21"],
            "top_k": 1,
            "candidate_pool": 1,
            "video_results": [{"video_id": "L21_V001", "rank": 1, "frames": []}],
        }

    service.search = fake_search
    service.qa_workflow = QaWorkflow()
    service.index_metadata = {}
    service.reranker_config = type("Config", (), {})()
    service.phase5_service = None
    service.refs = []
    service.index = np.zeros((1, 2))
    service.object_service = None
    service.metadata_docs = []
    service.attribute_service = type("Attributes", (), {"available": False})()

    payload = service.qa_prepare("q1", "người mặc áo đỏ", "đang làm gì?", retrieval_query="a person wearing a red shirt")

    assert seen_queries == ["a person wearing a red shirt"]
    assert payload["session"]["request"]["event_query"] == "người mặc áo đỏ"
    assert payload["evidence_pack"]["retrieval_context"]["qa_retrieval_query"] == "a person wearing a red shirt"


def test_qa_prepare_can_use_opt_in_agent_hybrid_retrieval(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    seen: list[tuple[str, str, int, bool]] = []

    def fake_agent_search(query_id: str, query: str, *, top_k: int, use_gemini: bool, strict: bool = False):
        seen.append((query_id, query, top_k, use_gemini))
        return {
            "mode": "agent_hybrid",
            "profile": "clip_bm25",
            "query_plan": {"visual_clip_query_en": "a person wearing a red shirt"},
            "video_results": [{"video_id": "L21_V001", "rank": 1, "keyframe_id": 1}],
        }

    service.agent_search = fake_agent_search
    service.qa_workflow = QaWorkflow()
    service.index_metadata = {}
    service.reranker_config = type("Config", (), {})()
    service.phase5_service = None
    service.refs = []
    service.index = np.zeros((1, 2))
    service.object_service = None
    service.metadata_docs = []
    service.attribute_service = type("Attributes", (), {"available": False})()

    payload = service.qa_prepare(
        "q-agent",
        "nguoi mac ao do",
        "nguoi do dang lam gi?",
        use_hybrid_retrieval=True,
        use_gemini_planner=False,
    )

    assert seen == [("q-agent", "nguoi mac ao do", 12, False)]
    context = payload["evidence_pack"]["retrieval_context"]
    assert context["qa_hybrid_retrieval"] is True
    assert context["qa_retrieval_query"] == "a person wearing a red shirt"
    assert payload["session"]["request"]["event_query"] == "nguoi mac ao do"


def test_agent_search_returns_workspace_trace_without_changing_result_contract(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    service.hybrid_query_planner = HybridQueryPlanner(TranslationConfig(provider="gemini"))

    class Engine:
        def search(self, query_id, plan, **_kwargs):
            return {
                "schema_version": "hybrid-retrieval-response-v1",
                "query_id": query_id,
                "query_plan": plan.to_dict(),
                "profile": plan.profile,
                "fusion_method": "single_channel_rank",
                "health": {},
                "failures": {},
                "channel_hit_counts": {"clip": 1},
                "latency_ms": {"clip": 1.0, "total": 1.2},
                "results": [{
                    "retriever": "rrf",
                    "rank": 1,
                    "raw_score": 0.1,
                    "video_id": "L21_V001",
                    "source_type": "hybrid",
                    "document_id": "clip:L21_V001:1",
                    "keyframe_id": 1,
                    "frame_idx": 0,
                    "pts_time": 0.0,
                    "matched_text": None,
                    "provenance": {"retriever_ranks": {"clip": 1}},
                }],
            }

    service.hybrid_engine = Engine()

    payload = service.agent_search("query-1-kis", "a person walking outdoors", use_gemini=True)

    assert payload["agent_workspace_version"] == "agent-workspace-v2"
    assert payload["agent_trace"]["status"] == "UNAVAILABLE"
    assert payload["agent_trace"]["validated_plan"] == payload["query_plan"]
    assert payload["results"][0]["video_id"] == "L21_V001"


def test_build_hybrid_engine_returns_configured_lazy_retriever_factories(tmp_path: Path) -> None:
    bge_dir = tmp_path / "bge"
    bm25_dir = tmp_path / "bm25"
    bge_dir.mkdir()
    bm25_dir.mkdir()
    (bge_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (bm25_dir / "manifest.json").write_text("{}", encoding="utf-8")
    service = RetrievalUiService.__new__(RetrievalUiService)
    service.config = RetrievalUiConfig(
        repo_root=tmp_path,
        registry_path=tmp_path / "registry.json",
        index_dir=tmp_path / "clip",
        metadata_dir=tmp_path / "metadata",
        static_dir=tmp_path / "web",
        groups={"L21"},
        hybrid_enabled=True,
        bge_index_dir=bge_dir,
        bm25_index_dir=bm25_dir,
        bge_model_path=tmp_path / "model",
    )
    service._encoder = lambda: object()

    engine = service._build_hybrid_engine()

    assert set(engine.factories) == {"clip", "bge", "bm25"}


def test_trake_request_can_opt_in_to_frame_level_hybrid_channels(tmp_path: Path) -> None:
    service = make_service(tmp_path)

    class Planner:
        def plan(self, query: str, *, use_gemini: bool):
            assert query == "noi ve mua lon"
            assert use_gemini is False
            return HybridQueryPlan(
                original_query=query,
                visual_clip_query_en="a report about heavy rain",
                semantic_text_query=query,
                lexical_text_query="mua lon",
                intent="lexical_text",
                enabled_retrievers=("bm25",),
                profile="bm25",
                fusion_method="single_channel",
                reasons=("ASR phrase",),
                source_filters={"bm25": ("asr",)},
            )

    class Engine:
        def search_channel(self, name, request):
            assert name == "bm25"
            assert request.query_text == "mua lon"
            assert request.groups == ("L21",)
            return [RetrievalHit("bm25", 1, 4.2, "L21_V001", "asr", "asr:1", 2, 30, 1.0, "mua lon")]

    service.hybrid_query_planner = Planner()
    service.hybrid_engine = Engine()
    service.trake_runtime_config = TrakeRuntimeConfig.legacy()
    service.structured_generator = type("Generator", (), {"ref_by_key": {
        (ref.video_id, ref.keyframe_id): ref for ref in service.refs_by_video["L21_V001"]
    }})()
    service.phase5_service = None
    service.object_service = None
    service.attribute_service = type("Attributes", (), {"available": False})()
    request = TrakeRequest(
        "q-trake",
        "noi ve mua lon",
        (TrakeEvent("e1", 1, "noi ve mua lon", "noi ve mua lon"),),
        constraints={"hybrid_retrieval": True, "hybrid_use_gemini": False},
    )

    results = service._trake_retrieve_request(request, 10)

    assert results["e1"][0]["retrieval_method"] == "rank_fusion_hybrid_event_v1"
    assert results["e1"][0]["provenance"] == ["bm25"]
    assert results.query_plans["e1"]["profile"] == "bm25"
    assert results.hybrid_failures == {}


def test_service_metadata_search_returns_video_results(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    service.metadata_docs = load_metadata_documents(tmp_path / "data" / "media-info", groups={"L21"})

    payload = service.metadata_search("60 giay sang", top_k=5)

    assert payload["total_documents"] == 1
    assert payload["results"][0]["video_id"] == "L21_V001"
    assert payload["results"][0]["video_url"] == "/video?video_id=L21_V001"


def test_service_keyframe_neighborhood_returns_surrounding_frames(tmp_path: Path) -> None:
    service = make_service(tmp_path)

    payload = service.keyframe_neighborhood("L21_V001", keyframe_id=2, radius=1)

    assert payload["video_id"] == "L21_V001"
    assert payload["total_frames"] == 4
    assert [frame["keyframe_id"] for frame in payload["frames"]] == [1, 2, 3]
    assert [frame["is_center"] for frame in payload["frames"]] == [False, True, False]
    assert payload["frames"][1]["image_url"] == "/keyframe?path=data/keyframes/L21_V001/002.jpg"
    assert payload["ordering"] == "pts_time_ascending"


def test_service_keyframe_neighborhood_handles_boundaries(tmp_path: Path) -> None:
    service = make_service(tmp_path)

    start = service.keyframe_neighborhood("L21_V001", keyframe_id=1, radius=3)
    end = service.keyframe_neighborhood("L21_V001", keyframe_id=4, radius=3)

    assert [frame["keyframe_id"] for frame in start["frames"]] == [1, 2, 3, 4]
    assert [frame["keyframe_id"] for frame in end["frames"]] == [1, 2, 3, 4]
    assert sum(frame["is_center"] for frame in start["frames"]) == 1
    assert sum(frame["is_center"] for frame in end["frames"]) == 1


def test_service_search_exposes_raw_and_video_results_from_candidate_pool(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    service.index = np.array(
        [
            [1.0, 0.0],
            [0.999, 0.001],
            [0.95, 0.05],
            [0.80, 0.20],
        ],
        dtype=np.float32,
    )
    service.index /= np.linalg.norm(service.index, axis=1, keepdims=True)
    service.refs = [
        FrameRef("L21_V001", "L21", 1, 0, 0.0, 30.0, None),
        FrameRef("L21_V001", "L21", 2, 30, 1.0, 30.0, None),
        FrameRef("L21_V002", "L21", 1, 0, 0.0, 30.0, None),
        FrameRef("L21_V003", "L21", 1, 0, 0.0, 30.0, None),
    ]
    service.assets_by_video.update({"L21_V002": {}, "L21_V003": {}})

    class FakeEncoder:
        def encode_text(self, text):
            assert text == "person"
            return np.array([1.0, 0.0], dtype=np.float32)

    service.encoder = FakeEncoder()

    payload = service.search(
        "person",
        top_k=2,
        candidate_pool=4,
        max_frames_per_video=1,
        matched_frames_per_video=2,
    )

    assert payload["aggregation_method"] == "max"
    assert payload["candidate_pool_size"] == 4
    assert len(payload["raw_results"]) == 4
    assert [item["rank"] for item in payload["raw_results"]] == [1, 2, 3, 4]
    assert [item["video_id"] for item in payload["video_results"]] == ["L21_V001", "L21_V002"]
    assert payload["video_results"][0]["frame_count"] == 2
    assert payload["video_results"][0]["matched_frame_count"] == 2
    assert len(payload["video_results"][0]["frames"]) == 2
    assert len(payload["results"]) == 2
    assert payload["aggregation_ms"] >= 0


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("top_k", 0),
        ("candidate_pool", 0),
        ("max_frames_per_video", 0),
        ("matched_frames_per_video", 0),
        ("mean_top_n", 0),
    ],
)
def test_service_search_rejects_non_positive_configuration(tmp_path: Path, argument: str, value: int) -> None:
    service = make_service(tmp_path)
    kwargs = {argument: value}

    with pytest.raises(ValueError, match=argument):
        service.search("person", **kwargs)


def test_enrich_video_result_preserves_non_l21_frame_without_image(tmp_path: Path) -> None:
    service = make_service(tmp_path)
    service.assets_by_video["L23_V001"] = {}
    raw = [
        SearchResult(1, 0.9, "L23_V001", "L23", 7, 31, 1.25, 25.0, None),
    ]
    video = aggregate_results_by_video(raw, max_frames_per_video=5)[0]

    payload = service.enrich_video_result(__import__("dataclasses").asdict(video))

    assert payload["best_keyframe_path"] is None
    assert payload["image_url"] is None
    assert payload["frames"][0]["keyframe_path"] is None
    assert payload["frames"][0]["image_url"] is None
