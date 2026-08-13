import json

from aic_retrieval.phase1_data import load_phase1_manifest, preflight_phase1_data


def test_preflight_reports_missing_data_without_crashing(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_schema_version": "1",
                "registry_schema_version": "0.1",
                "required_directories": ["clip-features-32", "map-keyframes", "media-info", "objects"],
                "groups": {"L21": {"expected_videos": 1, "expected_keyframe_images": 1, "has_keyframe_images": True}},
            }
        ),
        encoding="utf-8",
    )

    payload = preflight_phase1_data(tmp_path / "data", load_phase1_manifest(manifest_path))

    assert payload["overall_status"] == "MISSING"
    assert any(item["status"] == "MISSING" for item in payload["checks"])


def test_preflight_accepts_expected_missing_non_visual_group(tmp_path) -> None:
    data = tmp_path / "data"
    for name in ("clip-features-32", "map-keyframes", "media-info", "objects"):
        (data / name).mkdir(parents=True)
    manifest = {
        "manifest_schema_version": "1",
        "registry_schema_version": "0.1",
        "required_directories": ["clip-features-32", "map-keyframes", "media-info", "objects"],
        "groups": {"L22": {"expected_videos": 0, "has_keyframe_images": False}},
    }

    payload = preflight_phase1_data(data, manifest)

    assert payload["overall_status"] == "MISSING"  # no assets means registry validation is not integration-ready
    assert not any(item["name"] == "group:L22:keyframe_images" and item["status"] == "MISSING" for item in payload["checks"])


def test_preflight_counts_partial_images_as_warning_and_keeps_baseline_ready(tmp_path) -> None:
    data = tmp_path / "data"
    for name in ("clip-features-32", "map-keyframes", "media-info", "objects", "keyframes"):
        (data / name).mkdir(parents=True)
    import numpy as np

    np.save(data / "clip-features-32" / "L21_V001.npy", np.ones((2, 512), dtype=np.float16))
    (data / "map-keyframes" / "L21_V001.csv").write_text(
        "n,pts_time,fps,frame_idx\n1,0,30,0\n2,1,30,30\n", encoding="utf-8"
    )
    (data / "media-info" / "L21_V001.json").write_text("{}", encoding="utf-8")
    image_dir = data / "keyframes" / "L21_V001"
    image_dir.mkdir()
    (image_dir / "001.jpg").write_bytes(b"fixture")
    manifest = {
        "manifest_schema_version": "1",
        "registry_schema_version": "0.1",
        "required_directories": ["clip-features-32", "map-keyframes", "media-info", "objects"],
        "groups": {"L21": {"expected_videos": 1, "expected_keyframe_images": 2, "has_keyframe_images": True}},
    }

    payload = preflight_phase1_data(data, manifest, groups={"L21"})

    image_check = next(item for item in payload["checks"] if item["name"] == "group:L21:keyframe_images")
    assert image_check["status"] == "WARNING"
    assert "observed=1" in image_check["detail"]
    assert payload["overall_status"] == "WARNING"
