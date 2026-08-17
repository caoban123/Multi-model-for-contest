from __future__ import annotations
import argparse,importlib,importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main()->int:
    p=argparse.ArgumentParser(description="Explicit, bounded Phase 5 model pilot; never runs a whole group implicitly."); p.add_argument("--modality",choices=["ocr","asr"],required=True); p.add_argument("--inputs",type=Path,nargs="+",required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--model",required=True); a=p.parse_args()
    if len(a.inputs)>10: raise ValueError("pilot is limited to 10 explicit inputs")
    records=[]
    if a.modality=="ocr":
        if importlib.util.find_spec("paddleocr") is None: raise RuntimeError("paddleocr is not installed; approve and install the selected OCR profile first")
        PaddleOCR=importlib.import_module("paddleocr").PaddleOCR; engine=PaddleOCR(lang="vi")
        for path in a.inputs: records.append({"input":str(path),"model":a.model,"raw_result":engine.predict(str(path))})
    else:
        if importlib.util.find_spec("faster_whisper") is None: raise RuntimeError("faster-whisper is not installed; approve and install the selected ASR profile first")
        model=importlib.import_module("faster_whisper").WhisperModel(a.model)
        for path in a.inputs:
            segments,info=model.transcribe(str(path),word_timestamps=True); records.append({"input":str(path),"model":a.model,"language":info.language,"segments":[{"start":x.start,"end":x.end,"text":x.text} for x in segments]})
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(records,ensure_ascii=False,indent=2,default=str),encoding="utf-8"); return 0
if __name__=="__main__": raise SystemExit(main())
