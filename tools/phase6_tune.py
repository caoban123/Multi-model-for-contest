from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from aic_retrieval.query_planner import RuleBasedQueryPlanner
from aic_retrieval.reranking import RERANKER_VERSION,RerankerConfig,rerank_video_results


def reciprocal_rank(results, expected):
    return next((1/rank for rank,item in enumerate(results,1) if item["video_id"] in expected),0.0)


def fuse(results, fusion):
    output=[]; k=fusion["rrf_k"]
    for item in results:
        score=sum(fusion.get(f"weight_{name}",0)/(k+rank) for name,rank in item.get("modality_ranks",{}).items() if rank is not None)
        output.append({**item,"fusion_score":score,"video_score":score})
    return [{**item,"rank":rank} for rank,item in enumerate(sorted(output,key=lambda x:(-x["fusion_score"],x.get("rank",10**9),x["video_id"])),1)]


def evaluate(queries, config, fusion=None):
    planner=RuleBasedQueryPlanner(); scores=[]
    fusion=fusion or {"rrf_k":60,"weight_clip":1,"weight_object":.5,"weight_attribute":.35,"weight_metadata":.3,"weight_ocr":.6,"weight_asr":.55}
    for query in queries:
        reranked=rerank_video_results(fuse(query["baseline_results"],fusion),planner.plan(query["text"]),config)
        scores.append(reciprocal_rank(reranked,set(query["expected_video_ids"])))
    return {"mrr":statistics.mean(scores),"top1":statistics.mean(score==1 for score in scores)}


def main()->int:
    p=argparse.ArgumentParser(description="Tune the lightweight reranker on development labels only."); p.add_argument("--input",type=Path,required=True); p.add_argument("--output-config",type=Path,required=True); a=p.parse_args()
    payload=json.loads(a.input.read_text(encoding="utf-8")); queries=[q for q in payload["queries"] if q["split"]=="development"]
    if not queries or any(not q.get("expected_video_ids") or not q.get("baseline_results") for q in queries): p.error("every development query must have expected_video_ids and frozen baseline_results")
    fusion_profiles=[{"rrf_k":60,"weight_clip":1,"weight_object":.5,"weight_attribute":.35,"weight_metadata":.3,"weight_ocr":.6,"weight_asr":.55},{"rrf_k":60,"weight_clip":1,"weight_object":.65,"weight_attribute":.45,"weight_metadata":.4,"weight_ocr":.75,"weight_asr":.7},{"rrf_k":60,"weight_clip":1,"weight_object":.4,"weight_attribute":.3,"weight_metadata":.25,"weight_ocr":.5,"weight_asr":.45}]
    candidates=[]
    for fusion,(top_n,modality,lexical,alignment) in itertools.product(fusion_profiles,itertools.product((10,20,50),(.04,.08,.12),(.06,.12,.18),(.03,.06,.09))):
        config=RerankerConfig(top_n=top_n,modality_match_weight=modality,lexical_evidence_weight=lexical,planner_alignment_weight=alignment); metrics=evaluate(queries,config,fusion); candidates.append((metrics["mrr"],metrics["top1"],-top_n,config,fusion,metrics))
    _,_,_,best,best_fusion,metrics=max(candidates,key=lambda item:item[:3]); output={"version":RERANKER_VERSION,"status":"tuned_on_development_not_holdout","source_version":payload["version"],"development_query_count":len(queries),"objective":{"primary":"mrr","secondary":"top1","tie_break":"smaller_top_n"},"metrics":metrics,"fusion":best_fusion,"config":asdict(best)}
    a.output_config.parent.mkdir(parents=True,exist_ok=True); a.output_config.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps({"metrics":metrics,"fusion":best_fusion,"config":asdict(best)})); return 0
if __name__=="__main__": raise SystemExit(main())
