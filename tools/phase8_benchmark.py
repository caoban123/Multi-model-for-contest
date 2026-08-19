from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT/'src') not in sys.path: sys.path.insert(0,str(ROOT/'src'))


METRIC_KEYS=("video_top1_accuracy","video_topk_recall","required_event_coverage","temporal_order_accuracy","mean_frame_distance","frame_tolerance_accuracy","chain_success_rate","manual_correction_rate","latency_ms_median","latency_ms_p95")


def evaluate(queries_path: Path, store_path: Path, *, top_k: int=5, frame_tolerance: int=3) -> dict[str,Any]:
    spec=json.loads(queries_path.read_text(encoding='utf-8')); queries=spec.get('queries',[])
    labelled=[item for item in queries if item.get('expected_video_id') and item.get('events')]
    output={"benchmark_version":"phase8-benchmark-v1","query_version":spec.get('version'),"split":spec.get('split'),"query_count":len(queries),"labelled_query_count":len(labelled),"availability":"unavailable" if not labelled else "available","metrics":{key:None for key in METRIC_KEYS},"ablations":{"A_clip_keyframe_only":None,"B_structured_evidence":None,"C_dense_refinement":None},"notes":[]}
    if not labelled:
        output['notes'].append('Metrics are null/unavailable because expected labels are absent; null is not zero.')
        return output
    if not store_path.is_file():
        output['availability']='unavailable'; output['notes'].append('TRAKE SQLite store is missing.'); return output
    sessions=_load_sessions(store_path); rows=[]
    for query in labelled:
        matching=[state for state in sessions if state['request']['query_id']==query['query_id']]
        if not matching: continue
        state=matching[-1]; ranked=[chain for result in state.get('alignments',[]) for chain in result.get('chains',[])]; ranked.sort(key=lambda chain:-chain['score']['final_score'])
        expected_video=query['expected_video_id']; expected_frames=query.get('expected_event_frames') or []
        best=ranked[0] if ranked else None; selected=[entry['candidate'] for entry in best.get('events',[]) if entry.get('candidate')] if best else []
        distances=[abs(int(actual['keyframe_id'])-int(expected)) for actual,expected in zip(selected,expected_frames)] if expected_frames else []
        rows.append({"top1":bool(best and best['video_id']==expected_video),"topk":any(chain['video_id']==expected_video for chain in ranked[:top_k]),"coverage":len(selected)/max(1,len(query['events'])),"order":all(a['pts_time']<b['pts_time'] for a,b in zip(selected,selected[1:])),"distance":statistics.mean(distances) if distances else None,"tolerance":sum(value<=frame_tolerance for value in distances)/len(distances) if distances else None,"success":bool(best),"manual":bool(state.get('manual_chain')),"latency":sum(state.get('stage_timings_ms',{}).values())})
    if not rows:
        output['availability']='unavailable'; output['notes'].append('No matching persisted sessions for labelled queries.'); return output
    values=lambda key:[row[key] for row in rows if row[key] is not None]
    output['metrics'].update({"video_top1_accuracy":statistics.mean(values('top1')),"video_topk_recall":statistics.mean(values('topk')),"required_event_coverage":statistics.mean(values('coverage')),"temporal_order_accuracy":statistics.mean(values('order')),"mean_frame_distance":statistics.mean(values('distance')) if values('distance') else None,"frame_tolerance_accuracy":statistics.mean(values('tolerance')) if values('tolerance') else None,"chain_success_rate":statistics.mean(values('success')),"manual_correction_rate":statistics.mean(values('manual')),"latency_ms_median":statistics.median(values('latency')),"latency_ms_p95":_percentile(values('latency'),.95)})
    output['evaluated_query_count']=len(rows); return output


def _load_sessions(path: Path) -> list[dict[str,Any]]:
    with sqlite3.connect(path) as connection: rows=connection.execute('SELECT state_json FROM sessions ORDER BY updated_at').fetchall()
    return [json.loads(row[0]) for row in rows]


def _percentile(values:list[float],p:float)->float:
    ordered=sorted(values); return ordered[min(len(ordered)-1,max(0,int(round((len(ordered)-1)*p))))]


def main()->int:
    parser=argparse.ArgumentParser(); parser.add_argument('--queries',type=Path,required=True); parser.add_argument('--trake-store',type=Path,default=ROOT/'artifacts/trake/phase8_trake.sqlite3'); parser.add_argument('--output',type=Path); args=parser.parse_args()
    payload=evaluate(args.queries,args.trake_store); text=json.dumps(payload,ensure_ascii=False,indent=2)
    if args.output: args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(text+'\n',encoding='utf-8')
    print(text); return 0


if __name__=='__main__': raise SystemExit(main())
