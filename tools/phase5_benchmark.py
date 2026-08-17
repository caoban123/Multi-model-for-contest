from __future__ import annotations
import argparse,json,statistics,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from aic_retrieval.phase5_store import Phase5SearchService

def reciprocal_rank(results,expected):
    keys={(x["video_id"],x.get("keyframe_id")) for x in expected}
    return next((1/rank for rank,item in enumerate(results,1) if (item["video_id"],item.get("keyframe_id")) in keys),0.0)

def main()->int:
    p=argparse.ArgumentParser(description="Reproducible OCR/ASR lexical benchmark; null metrics when labels are absent."); p.add_argument("--queries",type=Path,required=True); p.add_argument("--store",type=Path,required=True); p.add_argument("--split",choices=["development","holdout"],required=True); p.add_argument("--top-k",type=int,default=10); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    payload=json.loads(a.queries.read_text(encoding="utf-8")); queries=[x for x in payload["queries"] if x["split"]==a.split]; service=Phase5SearchService(a.store); runs=[]
    for query in queries:
        started=time.perf_counter(); results=service.search_ocr(query["text"],a.top_k) if query["modality"]=="ocr" else service.search_asr(query["text"],a.top_k); elapsed=(time.perf_counter()-started)*1000; expected=query.get("expected",[])
        runs.append({"query_id":query["query_id"],"modality":query["modality"],"latency_ms":elapsed,"results":results,"reciprocal_rank":reciprocal_rank(results,expected) if expected else None})
    judged=[x["reciprocal_rank"] for x in runs if x["reciprocal_rank"] is not None]; latencies=[x["latency_ms"] for x in runs]
    output={"version":payload["version"],"split":a.split,"query_count":len(runs),"judged_query_count":len(judged),"metrics":{"mrr":statistics.mean(judged) if judged else None,"availability":"available" if judged else "unavailable"},"latency_ms":{"median":statistics.median(latencies) if latencies else None,"max":max(latencies) if latencies else None},"runs":runs}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(output["metrics"])); return 0
if __name__=="__main__": raise SystemExit(main())
