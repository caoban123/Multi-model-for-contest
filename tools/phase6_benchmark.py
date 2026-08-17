from __future__ import annotations

import argparse,json,statistics,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"src"))
from aic_retrieval.query_planner import RuleBasedQueryPlanner
from aic_retrieval.reranking import RerankerConfig,load_reranker_config,rerank_video_results,with_top_n
from tools.phase6_tune import fuse


def reciprocal_rank(results,expected):
    return next((1/rank for rank,item in enumerate(results,1) if item["video_id"] in expected),0.0)


def main()->int:
    p=argparse.ArgumentParser(description="Offline Phase 6 before/after reranking benchmark."); p.add_argument("--input",type=Path,required=True); p.add_argument("--split",choices=["development","holdout"],required=True); p.add_argument("--reranker-top-n",type=int); p.add_argument("--config",type=Path,default=ROOT/"configs"/"phase6_reranker_v1.json"); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    source=json.loads(a.input.read_text(encoding="utf-8")); config_payload=json.loads(a.config.read_text(encoding="utf-8")); planner=RuleBasedQueryPlanner(); config=load_reranker_config(a.config); config=with_top_n(config,a.reranker_top_n) if a.reranker_top_n else config; runs=[]
    for query in (x for x in source["queries"] if x["split"]==a.split):
        baseline=query.get("baseline_results",[]); expected=set(query.get("expected_video_ids",[])); started=time.perf_counter(); plan=planner.plan(query["text"]); fused=fuse(baseline,config_payload.get("fusion",{"rrf_k":60})); reranked=rerank_video_results(fused,plan,config); elapsed=(time.perf_counter()-started)*1000
        runs.append({"query_id":query["query_id"],"plan":plan.to_dict(),"baseline_rr":reciprocal_rank(baseline,expected) if expected else None,"reranked_rr":reciprocal_rank(reranked,expected) if expected else None,"reranker_ms":elapsed,"baseline_results":baseline,"reranked_results":reranked})
    judged=[x for x in runs if x["baseline_rr"] is not None]; metrics={"availability":"available" if judged else "unavailable","baseline_top1":statistics.mean(x["baseline_rr"]==1 for x in judged) if judged else None,"reranked_top1":statistics.mean(x["reranked_rr"]==1 for x in judged) if judged else None,"baseline_mrr":statistics.mean(x["baseline_rr"] for x in judged) if judged else None,"reranked_mrr":statistics.mean(x["reranked_rr"] for x in judged) if judged else None}
    if judged:
        metrics["top1_delta"]=metrics["reranked_top1"]-metrics["baseline_top1"]
        metrics["mrr_delta"]=metrics["reranked_mrr"]-metrics["baseline_mrr"]
    output={"version":source["version"],"split":a.split,"query_count":len(runs),"judged_query_count":len(judged),"fusion_config":config_payload.get("fusion"),"reranker_config":config.__dict__,"metrics":metrics,"reranker_latency_ms":{"median":statistics.median(x["reranker_ms"] for x in runs) if runs else None,"max":max((x["reranker_ms"] for x in runs),default=None)},"runs":runs}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(metrics)); return 0
if __name__=="__main__": raise SystemExit(main())
