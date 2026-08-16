import json
import sqlite3
from pathlib import Path

import numpy as np

from aic_retrieval.hybrid_candidates import StructuredCandidateGenerator
from aic_retrieval.metadata_search import MetadataConstraints, document_from_payload
from aic_retrieval.object_search import ObjectSearchService
from aic_retrieval.object_store import create_schema, load_alias_dictionary
from aic_retrieval.search import FrameRef
from aic_retrieval.structured_query import ObjectConstraint, StructuredQuery


def fixture_generator(tmp_path: Path):
    refs = [FrameRef("L21_V001","L21",1,10,1.0,30,None), FrameRef("L21_V001","L21",2,20,2.0,30,None), FrameRef("L21_V002","L21",1,10,1.0,30,None)]
    index = np.array([[1,0],[0.8,0.2],[0,1]], dtype=np.float32)
    db = tmp_path / "objects.sqlite"; connection = sqlite3.connect(db); create_schema(connection)
    connection.executemany("INSERT INTO frames VALUES (?,?,?,?,?,?)", [("L21_V001",1,10,1,1,"AVAILABLE"),("L21_V001",2,20,2,0,"UNKNOWN"),("L21_V002",1,10,1,1,"AVAILABLE")])
    connection.execute("INSERT INTO detections(video_id,keyframe_id,label_id,label_raw,label_normalized,confidence,bbox_y1,bbox_x1,bbox_y2,bbox_x2,center_x,center_y,area,source,schema_version) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", ("L21_V002",1,"1","Mobile phone","phone",0.9,0,0,0.2,0.2,0.1,0.1,0.04,"test","v1"))
    connection.commit(); connection.close()
    aliases = load_alias_dictionary(Path(__file__).resolve().parents[1] / "config" / "object_aliases_v1.json")
    objects = ObjectSearchService(db, aliases)
    docs = [document_from_payload("L21_V001", {"title":"Morning news", "author":"A", "length":10}), document_from_payload("L21_V002", {"title":"Sports", "author":"B", "length":20})]
    return StructuredCandidateGenerator(index, refs, objects, docs)


def test_three_channel_union_preserves_provenance_and_no_fake_metadata_score(tmp_path: Path) -> None:
    generator = fixture_generator(tmp_path)
    query = StructuredQuery("phone", enable_clip=True, enable_objects=True, enable_metadata=True,
                            object_constraints=(ObjectConstraint(("điện thoại",), filter_mode="soft"),),
                            metadata_constraints=MetadataConstraints(title_phrase="Morning"), metadata_mode="soft", clip_candidate_pool=1)
    payload = generator.generate(query, np.array([1,0], dtype=np.float32))
    assert payload["channel_counts"] == {"clip_frames":1,"object_frames":1,"attribute_frames":0,"metadata_videos":1}
    assert {item["video_id"] for item in payload["candidates"]} == {"L21_V001","L21_V002"}
    metadata = next(item for item in payload["candidates"] if "metadata" in item["provenance"])
    assert metadata["metadata_score"] is None
    assert metadata["clip_score"] is not None
    assert set(metadata["provenance"]) >= {"clip", "metadata"}
    assert "modality-specific" in payload["score_contract"]


def test_metadata_outside_clip_pool_gets_real_best_clip_representative(tmp_path: Path) -> None:
    generator = fixture_generator(tmp_path)
    query = StructuredQuery("phone", enable_clip=True, enable_metadata=True, metadata_constraints=MetadataConstraints(title_phrase="Sports"), metadata_mode="soft", clip_candidate_pool=1)
    payload = generator.generate(query, np.array([1,0], dtype=np.float32))
    item = next(candidate for candidate in payload["candidates"] if candidate["video_id"] == "L21_V002")
    assert item["keyframe_id"] == 1
    assert item["clip_rank"] is None
    assert item["clip_score"] == 0.0
    assert "clip_representative_only" in item["provenance"]


def test_hard_object_mode_filters_union_and_disabled_mode_excludes_channel(tmp_path: Path) -> None:
    generator = fixture_generator(tmp_path)
    hard = StructuredQuery("phone", enable_clip=True, enable_objects=True, object_constraints=(ObjectConstraint(("phone",), filter_mode="hard"),), clip_candidate_pool=3)
    payload = generator.generate(hard, np.array([1,0], dtype=np.float32))
    assert [(item["video_id"],item["keyframe_id"]) for item in payload["candidates"]] == [("L21_V002",1)]
    disabled = StructuredQuery("phone", enable_clip=True, enable_objects=True, object_constraints=(ObjectConstraint(("phone",), filter_mode="disabled"),), clip_candidate_pool=1)
    payload = generator.generate(disabled, np.array([1,0], dtype=np.float32))
    assert payload["channel_counts"]["object_frames"] == 0
    assert payload["unknown_object_frame_count"] == 0


def test_hard_clip_mode_excludes_object_only_candidates(tmp_path: Path) -> None:
    generator = fixture_generator(tmp_path)
    query = StructuredQuery("phone", enable_clip=True, enable_objects=True, clip_mode="hard",
                            object_constraints=(ObjectConstraint(("phone",), filter_mode="soft"),), clip_candidate_pool=1)
    payload = generator.generate(query, np.array([1,0], dtype=np.float32))
    assert [(item["video_id"], item["keyframe_id"]) for item in payload["candidates"]] == [("L21_V001",1)]
