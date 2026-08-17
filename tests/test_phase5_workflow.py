from pathlib import Path

import pytest

from aic_retrieval.phase5_store import create_store
from aic_retrieval.phase5_workflow import Phase5PreparationError,require_numpy_index
import tools.prepare_phase5 as prepare_phase5


def paths(root: Path):
    return (
        root/"data",root/"artifacts/registry/data_registry.json",root/"artifacts/registry/validation_report.json",
        root/"artifacts/indexes/l21_numpy",root/"ocr.jsonl",root/"asr.jsonl",root/"artifacts/phase5/manual/phase5.sqlite3",
    )


def source_data(data: Path):
    features=data/"clip-features-32"; mappings=data/"map-keyframes"; features.mkdir(parents=True); mappings.mkdir(parents=True); (features/"L21_V001.npy").write_bytes(b"source"); (mappings/"L21_V001.csv").write_text("n,pts_time,fps,frame_idx\n",encoding="utf-8")


def fake_runner(calls):
    def run(command,cwd=prepare_phase5.ROOT):
        calls.append(command)
        if "tools/data_registry.py" in command:
            output=Path(command[command.index("--output")+1]); output.parent.mkdir(parents=True,exist_ok=True); output.write_text("{}",encoding="utf-8")
        elif "tools/build_numpy_index.py" in command:
            output=Path(command[command.index("--output-dir")+1]); output.mkdir(parents=True,exist_ok=True); (output/"vectors.npy").write_bytes(b"real-builder-placeholder"); (output/"refs.json").write_text("[]",encoding="utf-8")
    return run


def test_prepare_rebuilds_registry_and_index_when_artifacts_does_not_exist(tmp_path,monkeypatch):
    data,registry,validation,index,ocr,asr,output=paths(tmp_path); source_data(data); calls=[]; monkeypatch.setattr(prepare_phase5,"run",fake_runner(calls))
    prepare_phase5.prepare(data,registry,validation,"L21",index,None,None,output,False,True,False)
    assert [Path(call[1]).name for call in calls]==["data_registry.py","build_numpy_index.py"]
    assert (index/"vectors.npy").is_file() and (index/"refs.json").is_file()


def test_prepare_reports_exact_missing_index_source_prerequisites(tmp_path):
    data,registry,validation,index,ocr,asr,output=paths(tmp_path); data.mkdir()
    with pytest.raises(Phase5PreparationError) as error:
        prepare_phase5.prepare(data,registry,validation,"L21",index,None,None,output,False,True,False)
    assert "clip-features-32" in str(error.value) and "map-keyframes" in str(error.value)


def test_prepare_rebuilds_missing_index_directory_but_reuses_registry(tmp_path,monkeypatch):
    data,registry,validation,index,ocr,asr,output=paths(tmp_path); source_data(data); registry.parent.mkdir(parents=True); registry.write_text("{}",encoding="utf-8"); calls=[]; monkeypatch.setattr(prepare_phase5,"run",fake_runner(calls))
    prepare_phase5.prepare(data,registry,validation,"L21",index,None,None,output,False,True,False)
    assert len(calls)==1 and "tools/build_numpy_index.py" in calls[0]


@pytest.mark.parametrize("present,missing",[("refs.json","vectors.npy"),("vectors.npy","refs.json")])
def test_index_diagnostic_names_each_missing_required_file(tmp_path,present,missing):
    index=tmp_path/"artifacts/indexes/l21_numpy"; index.mkdir(parents=True); (index/present).write_text("x",encoding="utf-8")
    with pytest.raises(Phase5PreparationError) as error: require_numpy_index(index,tmp_path/"artifacts/registry/data_registry.json","L21")
    assert str(index/missing) in str(error.value)
    assert "python tools/build_numpy_index.py" in str(error.value)


def test_store_creation_creates_missing_output_parents(tmp_path):
    output=tmp_path/"artifacts/phase5/manual/phase5.sqlite3"; connection=create_store(output); connection.close()
    assert output.is_file()


def test_normal_prepare_with_valid_index_runs_only_store_builder(tmp_path,monkeypatch):
    data,registry,validation,index,ocr,asr,output=paths(tmp_path); source_data(data); registry.parent.mkdir(parents=True); registry.write_text("{}",encoding="utf-8"); index.mkdir(parents=True); (index/"vectors.npy").write_bytes(b"x"); (index/"refs.json").write_text("[]",encoding="utf-8"); ocr.write_text("{}\n",encoding="utf-8"); calls=[]; monkeypatch.setattr(prepare_phase5,"run",fake_runner(calls))
    prepare_phase5.prepare(data,registry,validation,"L21",index,ocr,None,output,True,False,False)
    assert len(calls)==1 and "tools/build_phase5_store.py" in calls[0]
