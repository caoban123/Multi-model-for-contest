from aic_retrieval.provenance import fingerprint_paths


def test_fingerprint_paths_is_deterministic_across_input_order(tmp_path) -> None:
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")

    assert fingerprint_paths([second, first], tmp_path) == fingerprint_paths([first, second], tmp_path)
