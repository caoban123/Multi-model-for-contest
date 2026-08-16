from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))

from aic_retrieval.hybrid_candidates import StructuredCandidateGenerator
from aic_retrieval.hybrid_ranking import RrfConfig, rank_video_candidates
from aic_retrieval.metadata_search import MetadataConstraints, load_metadata_documents
from aic_retrieval.object_search import ObjectSearchService
from aic_retrieval.object_store import load_alias_dictionary
from aic_retrieval.provenance import sha256_file
from aic_retrieval.search import load_numpy_index
from aic_retrieval.structured_query import ObjectConstraint, StructuredQuery
from aic_retrieval.text_encoder import DEFAULT_CLIP_MODEL_ID, ClipTextEncoder
from tools.phase4_baseline import load_phase4_queries
from tools.text_query_benchmark import latency_summary

ABLATIONS = {
    "A_clip_only": (True, False, False),
    "B_object_only": (False, True, False),
    "C_metadata_only": (False, False, True),
    "D_clip_object": (True, True, False),
    "E_clip_metadata": (True, False, True),
    "F_clip_object_metadata": (True, True, True),
}
THRESHOLDS = (0.1, 0.2, 0.3, 0.5)


def unavailable_quality_metrics() -> dict[str, Any]:
    return {"status":"unavailable", "reason":"No verified relevance ground truth/manual judgements.",
            "Recall@1":None,"Recall@5":None,"Recall@20":None,"Recall@50":None,"Recall@100":None,
            "MRR":None,"NDCG":None,"Top-1_good":None,"Top-1_bad":None}


def constraints_from_query(raw: dict[str, Any], min_confidence: float = 0.3) -> tuple[tuple[ObjectConstraint, ...], MetadataConstraints]:
    structured = raw.get("structured_constraints") or {}
    objects = []
    for item in structured.get("objects") or []:
        position = item.get("position", "any")
        horizontal = position if position in {"left","center","right"} else "any"
        vertical = position if position in {"top","middle","bottom"} else "any"
        objects.append(ObjectConstraint((str(item["label"]),), item.get("count_operator", ">="), int(item.get("min_count", item.get("count", 1))), horizontal, vertical, min_confidence, 0.5, item.get("mode", "soft")))
    interaction = structured.get("interaction")
    if interaction and interaction.get("object"):
        objects.append(ObjectConstraint((str(interaction["object"]),), ">=", 1, "any", "any", min_confidence, 0.5, interaction.get("mode", "soft")))
    metadata = dict(structured.get("metadata") or {})
    metadata.pop("mode", None)
    mapped = MetadataConstraints(
        groups=tuple(metadata.get("groups") or ()), video_ids=tuple(metadata.get("video_ids") or ()),
        channel=metadata.get("channel") or metadata.get("author"), publish_date=metadata.get("publish_date") or metadata.get("date"),
        date_from=metadata.get("date_from"), date_to=metadata.get("date_to"), duration_min=metadata.get("duration_min"), duration_max=metadata.get("duration_max"),
        title_phrase=metadata.get("title") or metadata.get("title_phrase"), keywords=tuple(metadata.get("keywords") or ()),
    )
    return tuple(objects), mapped


def metadata_mode(raw: dict[str, Any]) -> str:
    return str(((raw.get("structured_constraints") or {}).get("metadata") or {}).get("mode", "soft"))


def compact_video_result(item: dict[str, Any]) -> dict[str, Any]:
    return {key:item.get(key) for key in ("rank","video_id","fusion_score","best_keyframe_id","best_frame_idx","best_pts_time","frame_count","matched_frame_count","modality_ranks","representative_rule")}


def run_one(generator: StructuredCandidateGenerator, raw_query: dict[str, Any], vector: Any, ablation: str, enable: tuple[bool,bool,bool], threshold: float, top_k: int, pool: int, config: RrfConfig) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    enable_clip, enable_objects, enable_metadata = enable
    object_constraints, metadata_constraints = constraints_from_query(raw_query, threshold)
    query = StructuredQuery(raw_query["query_text_en"], enable_clip, enable_objects, enable_metadata,
                            "soft" if enable_clip else "disabled", object_constraints,
                            metadata_constraints, metadata_mode(raw_query) if enable_metadata else "disabled", pool, "rrf")
    started=time.perf_counter(); candidate_started=time.perf_counter()
    candidates=generator.generate(query,vector)
    candidate_ms=(time.perf_counter()-candidate_started)*1000
    fusion_started=time.perf_counter(); ranked=rank_video_candidates(candidates,query,config,top_k,5); fusion_ms=(time.perf_counter()-fusion_started)*1000
    elapsed_ms=(time.perf_counter()-started)*1000
    object_applicable=bool(object_constraints) and enable_objects
    metadata_applicable=any(vars(metadata_constraints).values()) and enable_metadata
    frame_total=len(generator.refs)
    metadata_video_ids={item.video_id for item in generator.metadata_docs}; index_video_ids=set(generator.positions_by_video)
    run={"ablation":ablation,"query_id":raw_query["query_id"],"query_type":raw_query["query_type"],"split":raw_query["split"],
         "enabled":{"clip":enable_clip,"objects":enable_objects,"metadata":enable_metadata},"object_applicable":object_applicable,"metadata_applicable":metadata_applicable,
         "min_confidence":threshold,"candidate_count":candidates["candidate_count"],"candidate_coverage":candidates["candidate_count"]/frame_total if frame_total else 0,
         "result_video_count":len(ranked["video_results"]),"missing_modality_rate":{"objects":candidates["unknown_object_frame_count"]/frame_total if object_applicable and frame_total else None,
         "metadata":len(index_video_ids-metadata_video_ids)/len(index_video_ids) if metadata_applicable and index_video_ids else None},
         "latency_ms":{"candidate_generation":candidate_ms,"fusion":fusion_ms,"total":elapsed_ms},"quality_metrics":unavailable_quality_metrics(),
         "video_results":[compact_video_result(item) for item in ranked["video_results"]]}
    rows=[]
    for result in ranked["video_results"]:
        representative=next((frame for frame in result["frames"] if frame["is_representative"]),{})
        object_evidence=representative.get("evidence",{}).get("object",[])
        matched_objects=sorted({label for match in object_evidence for label in match.get("matched_labels",[])})
        metadata_evidence=representative.get("evidence",{}).get("metadata") or {}
        matched_metadata_fields=sorted({item["field"] for item in metadata_evidence.get("evidence",[])})
        rows.append({"ablation":ablation,"query_id":raw_query["query_id"],"split":raw_query["split"],"query_type":raw_query["query_type"],
                     "query_text_vi":raw_query["query_text_vi"],"query_text_en":raw_query["query_text_en"],"min_confidence":threshold,
                     "rank":result["rank"],"video_id":result["video_id"],"keyframe_id":result["best_keyframe_id"],
                     "clip_score":representative.get("clip_score"),"clip_rank":result["modality_ranks"]["clip"],
                     "object_score":representative.get("object_score"),"object_rank":result["modality_ranks"]["object"],
                     "matched_objects":"|".join(matched_objects),"metadata_score":"","metadata_rank":result["modality_ranks"]["metadata"],
                     "matched_metadata_fields":"|".join(matched_metadata_fields),"fusion_score":result["fusion_score"],"fusion_rank":result["rank"],"fusion_method":"rrf",
                     "manual_judgement":"","manual_notes":""})
    return run,rows


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs: return {"query_count":0}
    return {"query_count":len(runs),"candidate_count_mean":sum(item["candidate_count"] for item in runs)/len(runs),
            "object_applicable_query_count":sum(bool(item.get("object_applicable")) for item in runs),"metadata_applicable_query_count":sum(bool(item.get("metadata_applicable")) for item in runs),
            "candidate_coverage_mean":sum(item["candidate_coverage"] for item in runs)/len(runs),
            "latency_ms":{stage:latency_summary([item["latency_ms"][stage] for item in runs]) for stage in ("candidate_generation","fusion","total")},
            "object_missing_rate_mean":(lambda values: sum(values)/len(values) if values else None)([item["missing_modality_rate"]["objects"] for item in runs if item["missing_modality_rate"]["objects"] is not None]),
            "metadata_missing_rate_mean":(lambda values: sum(values)/len(values) if values else None)([item["missing_modality_rate"].get("metadata") for item in runs if item["missing_modality_rate"].get("metadata") is not None]),
            "quality_metrics":unavailable_quality_metrics()}


def main() -> int:
    parser=argparse.ArgumentParser(description="Run Phase 4 six-way ablation and development-only confidence sweep.")
    parser.add_argument("--queries",default="benchmarks/phase4_queries_v1.json"); parser.add_argument("--index-dir",default="artifacts/indexes/l21_numpy")
    parser.add_argument("--object-store",default="artifacts/structured/l21_objects.sqlite"); parser.add_argument("--aliases",default="config/object_aliases_v1.json")
    parser.add_argument("--metadata-dir",default="data/media-info"); parser.add_argument("--groups",default="L21"); parser.add_argument("--candidate-pool",type=int,default=100)
    parser.add_argument("--top-k",type=int,default=100); parser.add_argument("--clip-model-id",default=os.environ.get("AIC_CLIP_MODEL_ID",DEFAULT_CLIP_MODEL_ID))
    parser.add_argument("--clip-cache-dir",default=os.environ.get("AIC_CLIP_CACHE_DIR")); parser.add_argument("--clip-local-files-only",action="store_true")
    parser.add_argument("--output",default="artifacts/benchmarks/phase4/ablation_v1.json"); parser.add_argument("--csv-output",default="artifacts/benchmarks/phase4/ablation_v1_review.csv")
    args=parser.parse_args(); version,queries=load_phase4_queries(Path(args.queries)); groups={item.strip() for item in args.groups.split(",") if item.strip()}
    index,refs,index_metadata=load_numpy_index(Path(args.index_dir)); aliases=load_alias_dictionary(Path(args.aliases)); objects=ObjectSearchService(Path(args.object_store),aliases)
    generator=StructuredCandidateGenerator(index,refs,objects,load_metadata_documents(Path(args.metadata_dir),groups))
    model_started=time.perf_counter(); encoder=ClipTextEncoder(args.clip_model_id,Path(args.clip_cache_dir) if args.clip_cache_dir else None,args.clip_local_files_only); model_load_ms=(time.perf_counter()-model_started)*1000
    vectors={}; encoding_ms={}
    for query in queries:
        started=time.perf_counter(); vectors[query["query_id"]]=encoder.encode_text(query["query_text_en"]); encoding_ms[query["query_id"]]=(time.perf_counter()-started)*1000
    config=RrfConfig(); runs=[]; rows=[]
    for ablation,enabled in ABLATIONS.items():
        for query in queries:
            run,review=run_one(generator,query,vectors[query["query_id"]],ablation,enabled,0.3,args.top_k,args.candidate_pool,config); run["latency_ms"]["encoding"]=encoding_ms[query["query_id"]]; runs.append(run); rows.extend(review)
    threshold_runs=[]
    sweep_configs={name:ABLATIONS[name] for name in ("B_object_only","D_clip_object","F_clip_object_metadata")}
    for threshold in THRESHOLDS:
        for ablation,enabled in sweep_configs.items():
            for query in (item for item in queries if item["split"]=="development" and constraints_from_query(item,threshold)[0]):
                run,_=run_one(generator,query,vectors[query["query_id"]],ablation,enabled,threshold,args.top_k,args.candidate_pool,config); threshold_runs.append(run)
    by_config={name:summarize_runs([item for item in runs if item["ablation"]==name]) for name in ABLATIONS}
    by_type={name:{query_type:summarize_runs([item for item in runs if item["ablation"]==name and item["query_type"]==query_type]) for query_type in sorted({q["query_type"] for q in queries})} for name in ABLATIONS}
    commit=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,check=True,capture_output=True,text=True).stdout.strip(); branch=subprocess.run(["git","branch","--show-current"],cwd=ROOT,check=True,capture_output=True,text=True).stdout.strip()
    payload={"benchmark_version":"phase4-ablation-1.0","query_set_version":version,"timestamp":datetime.now(timezone.utc).isoformat(),"branch":branch,"commit":commit,
             "query_set_fingerprint":sha256_file(Path(args.queries)),"index_manifest":index_metadata,"model":args.clip_model_id,"machine":{"platform":platform.platform(),"python":platform.python_version(),"device":getattr(encoder,"device","unknown")},
             "benchmark_config":{"candidate_pool":args.candidate_pool,"top_k":args.top_k,"rrf":vars(config),"fixed_nms_iou_threshold":0.5,"ablation_threshold":0.3},
             "split_discipline":{"development_count":sum(q["split"]=="development" for q in queries),"holdout_count":sum(q["split"]=="holdout" for q in queries),
                                  "threshold_sweep_split":"development only","holdout_used_for_tuning":False,"selection_policy":"No threshold or weight selected because relevance judgements are unavailable."},
             "model_load_ms":model_load_ms,"encoding_latency_ms":latency_summary(list(encoding_ms.values())),"ablations":runs,"summary_by_configuration":by_config,"summary_by_query_type":by_type,
             "threshold_sweep":{"thresholds":list(THRESHOLDS),"configurations":list(sweep_configs),"runs":threshold_runs,
                                "summary":{str(t):{name:summarize_runs([item for item in threshold_runs if item["min_confidence"]==t and item["ablation"]==name]) for name in sweep_configs} for t in THRESHOLDS}},
             "quality_metrics":unavailable_quality_metrics()}
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    csv_path=Path(args.csv_output); csv_path.parent.mkdir(parents=True,exist_ok=True); fields=list(rows[0]) if rows else []
    with csv_path.open("w",encoding="utf-8-sig",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"output":str(output),"csv_output":str(csv_path),"ablation_runs":len(runs),"threshold_runs":len(threshold_runs),"review_rows":len(rows),"summary_by_configuration":by_config,"quality_metrics":payload["quality_metrics"]},ensure_ascii=False,indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
