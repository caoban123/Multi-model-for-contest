from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main() -> int:
    parser=argparse.ArgumentParser(description="Audit inputs for Phase 5 without modifying contest data.")
    parser.add_argument("--data-root",type=Path,default=ROOT/"data"); parser.add_argument("--groups",default="L21"); parser.add_argument("--probe-audio",action="store_true"); parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(); groups={x.strip() for x in args.groups.split(",") if x.strip()}
    video_ids=set()
    for base,pattern in ((args.data_root/"keyframes","*"),(args.data_root/"map-keyframes","*.csv"),(args.data_root/"videos","*")):
        if base.exists():
            for path in base.rglob(pattern):
                video_id=path.stem if path.is_file() else path.name
                if video_id.split("_",1)[0] in groups: video_ids.add(video_id)
    rows=[]
    video_ext={".mp4",".mkv",".avi",".mov",".webm"}
    for video_id in sorted(video_ids):
        images=list((args.data_root/"keyframes"/video_id).glob("*.jpg"))
        videos=sorted(path for path in (args.data_root/"videos").rglob(f"{video_id}.*") if path.suffix.lower() in video_ext) if (args.data_root/"videos").exists() else []
        video=videos[0] if videos else None
        row={"video_id":video_id,"keyframes":len(images),"has_mapping":(args.data_root/"map-keyframes"/f"{video_id}.csv").is_file(),"has_video":video is not None,"video_size_bytes":video.stat().st_size if video else None,"audio_status":"NOT_PROBED"}
        if args.probe_audio and video:
            row["audio_status"],row["audio_probe"]=probe_audio(video)
        rows.append(row)
    payload={"schema_version":"phase5-audit-v1","data_root":str(args.data_root),"groups":sorted(groups),"video_count":len(rows),"keyframe_count":sum(x["keyframes"] for x in rows),"videos_with_files":sum(x["has_video"] for x in rows),"records":rows,"blocking":["data root is missing"] if not args.data_root.exists() else []}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({key:payload[key] for key in ("video_count","keyframe_count","videos_with_files","blocking")},ensure_ascii=False)); return 2 if payload["blocking"] else 0


def probe_audio(path: Path) -> tuple[str,dict]:
    command=["ffprobe","-v","error","-select_streams","a:0","-show_entries","stream=codec_name,sample_rate,channels,duration","-of","json",str(path)]
    try: completed=subprocess.run(command,capture_output=True,text=True,timeout=30,check=False)
    except FileNotFoundError: return "FFPROBE_UNAVAILABLE",{}
    if completed.returncode: return "ERROR",{"stderr":completed.stderr.strip()}
    payload=json.loads(completed.stdout or "{}"); return ("AVAILABLE" if payload.get("streams") else "NO_AUDIO"),payload


if __name__ == "__main__": raise SystemExit(main())
