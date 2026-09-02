from __future__ import annotations

from pathlib import Path

from aic_retrieval.trake_refinement import DenseRefiner, RefinementConfig, SampledFrame


class FakeDecoder:
    def __init__(self): self.calls=[]
    def sample(self,_path,times): self.calls.append(times); return [SampledFrame(value,{"time":value}) for value in times]
    def write(self,path,_image): path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(b'image')


def test_refinement_is_bounded_two_stage_and_cached(tmp_path: Path) -> None:
    video=tmp_path/'data/video/L21_V001.mp4'; video.parent.mkdir(parents=True); video.write_bytes(b'video')
    decoder=FakeDecoder(); scorer=lambda _text,images:[-abs(item['time']-10.25) for item in images]
    refiner=DenseRefiner(tmp_path,{"L21_V001":{"video_path":"data/video/L21_V001.mp4"}},tmp_path/'artifacts/trake/refinement',scorer,decoder=decoder,config=RefinementConfig(max_frames_per_refinement=50))
    result=refiner.refine(video_id='L21_V001',event_id='e1',event_text='event',source_keyframe_id=1,source_frame_idx=250,source_pts_time=10.0,fps=25.0)
    assert result.status=='AVAILABLE' and len(decoder.calls)==2
    assert sum(map(len,decoder.calls))<=50
    assert result.refined_image_path and (tmp_path/result.refined_image_path).is_file()
    cached=refiner.refine(video_id='L21_V001',event_id='e1',event_text='event',source_keyframe_id=1,source_frame_idx=250,source_pts_time=10.0,fps=25.0)
    assert cached.refined_pts_time==result.refined_pts_time and len(decoder.calls)==2


def test_missing_video_or_scorer_returns_keyframe_safe_unavailable(tmp_path: Path) -> None:
    missing=DenseRefiner(tmp_path,{},tmp_path/'cache',lambda _t,_i:[],decoder=FakeDecoder()).refine(video_id='L21_V999',event_id='e1',event_text='x',source_keyframe_id=1,source_frame_idx=1,source_pts_time=1,fps=25)
    assert missing.status=='REFINEMENT_UNAVAILABLE'
    video=tmp_path/'v.mp4'; video.write_bytes(b'x')
    no_scorer=DenseRefiner(tmp_path,{"L21_V001":{"video_path":"v.mp4"}},tmp_path/'cache',None,decoder=FakeDecoder()).refine(video_id='L21_V001',event_id='e1',event_text='x',source_keyframe_id=1,source_frame_idx=1,source_pts_time=1,fps=25)
    assert 'keyframe chain remains valid' in no_scorer.reason


def test_refinement_accepts_registry_video_resolver_outside_repo_root(tmp_path: Path) -> None:
    external = tmp_path.parent / f"{tmp_path.name}-external" / "L21_V001.mp4"
    external.parent.mkdir(parents=True, exist_ok=True)
    external.write_bytes(b"video")
    decoder = FakeDecoder()
    refiner = DenseRefiner(
        tmp_path,
        {"L21_V001": {"video_path": "data/videos/L21_V001.mp4"}},
        tmp_path / "cache",
        lambda _text, images: [-abs(item["time"] - 2.0) for item in images],
        decoder=decoder,
        video_resolver=lambda _video_id: external,
    )

    result = refiner.refine(
        video_id="L21_V001",
        event_id="e1",
        event_text="event",
        source_keyframe_id=1,
        source_frame_idx=50,
        source_pts_time=2.0,
        fps=25.0,
    )

    assert result.status == "AVAILABLE"
    assert result.provenance["source_video"] == str(external.resolve())
