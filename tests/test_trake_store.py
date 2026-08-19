from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from aic_retrieval.trake_store import TrakeStore
from aic_retrieval.trake_workflow import TrakeWorkflow


def rows(event_id: str) -> list[dict[str, object]]:
    frame=1 if event_id=='e1' else 2
    return [{"video_id":"L21_V001","keyframe_id":frame,"frame_idx":frame*10,"pts_time":float(frame),"score":1.0}]


def ready(store: TrakeStore) -> tuple[TrakeWorkflow,str,str]:
    workflow=TrakeWorkflow(lambda event,_pool:rows(event.event_id),store=store)
    session=workflow.plan('q','first then second')['state']['session_id']; workflow.search(session,event_pool_size=30,max_per_video=5,video_pool_size=10)
    payload=workflow.align(session); chain=payload['state']['alignments'][0]['chains'][0]['chain_id']
    return workflow,session,chain


def test_store_has_separate_phase8_tables_and_reopens_session(tmp_path: Path) -> None:
    store=TrakeStore(tmp_path/'trake.sqlite3'); workflow,session,_=ready(store)
    reopened=TrakeWorkflow(lambda event,_pool:rows(event.event_id),store=TrakeStore(store.path))
    assert reopened.get(session)['state']['request']['query_id']=='q'
    with sqlite3.connect(store.path) as connection:
        tables={row[0] for row in connection.execute("select name from sqlite_master where type='table'")}
    assert {'sessions','events','candidates','video_candidates','chains','chain_events','reviews','refinements','exports'} <= tables


def test_manual_replacement_is_revalidated(tmp_path: Path) -> None:
    store=TrakeStore(tmp_path/'trake.sqlite3')
    def many(event,_pool):
        if event.event_id=='e1': return [{"video_id":"L21_V001","keyframe_id":1,"frame_idx":10,"pts_time":1.0,"score":1.0},{"video_id":"L21_V001","keyframe_id":9,"frame_idx":90,"pts_time":9.0,"score":.8}]
        return [{"video_id":"L21_V001","keyframe_id":2,"frame_idx":20,"pts_time":2.0,"score":1.0}]
    workflow=TrakeWorkflow(many,store=store); session=workflow.plan('q','first then second')['state']['session_id']; workflow.search(session,event_pool_size=30,max_per_video=5,video_pool_size=10); aligned=workflow.align(session)
    chain=aligned['state']['alignments'][0]['chains'][0]
    bad=next(c['candidate_id'] for p in aligned['state']['pools'] if p['event']['event_id']=='e1' for c in p['candidates'] if c['keyframe_id']==9)
    with pytest.raises(ValueError,match='invariants'):
        workflow.replace_candidate(session,chain['chain_id'],'e1',bad)


def test_review_is_append_only_and_export_requires_confirmation(tmp_path: Path) -> None:
    store=TrakeStore(tmp_path/'trake.sqlite3'); workflow,session,chain=ready(store)
    rejected=workflow.review(session,chain,'rejected','tester')['review']
    with pytest.raises(ValueError,match='confirmed'): workflow.export(session,rejected['review_id'])
    confirmed=workflow.review(session,chain,'confirmed','tester')['review']; record=workflow.export(session,confirmed['review_id'])['export']
    assert len(store.reviews(session))==2
    assert record['not_final_btc_format'] is True and record['video_id']=='L21_V001'
    assert record['same_video'] is True and record['temporally_ordered'] is True
    assert record['confirmed_by']=='tester' and record['human_confirmed'] is True
    assert record['events'][0]['mapping']['source']=='validated_index_ref'
    assert 'algorithm_selection' in record['events'][0] and 'evidence' in record['events'][0]
