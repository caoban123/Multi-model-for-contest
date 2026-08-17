from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from aic_retrieval.phase5_store import Phase5SearchService

def main()->int:
    p=argparse.ArgumentParser(); p.add_argument("--store",type=Path,required=True); p.add_argument("--modality",choices=["ocr","asr"],required=True); p.add_argument("--query",required=True); p.add_argument("--top-k",type=int,default=10); p.add_argument("--min-confidence",type=float,default=0); p.add_argument("--output",type=Path); a=p.parse_args(); service=Phase5SearchService(a.store)
    if not a.store.is_file():
        p.error(
            f"Phase 5 store not found: {a.store}\n\nBuild it from real OCR/ASR JSONL with:\n"
            "  python tools/prepare_phase5.py --data-root data --groups L21 "
            "--ocr-jsonl <path-to-ocr.jsonl> --asr-jsonl <path-to-asr.jsonl> "
            f"--output {a.store} --overwrite"
        )
    result=service.search_ocr(a.query,a.top_k,a.min_confidence) if a.modality=="ocr" else service.search_asr(a.query,a.top_k); payload={"modality":a.modality,"query":a.query,"results":result}
    text=json.dumps(payload,ensure_ascii=False,indent=2); print(text)
    if a.output: a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(text,encoding="utf-8")
    return 0
if __name__=="__main__": raise SystemExit(main())
