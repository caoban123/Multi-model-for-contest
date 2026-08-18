from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aic_retrieval.phase5_schema import AsrSegment, AsrTranscript, OcrDetection, OcrFrame


def parse_map_keyframes(csv_path: Path) -> dict[int, dict[str, float | int]]:
    """Map keyframe_id (n) -> {pts_time, fps, frame_idx}."""
    mapping: dict[int, dict[str, float | int]] = {}
    if not csv_path.is_file():
        return mapping
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                keyframe_id = int(row["n"])
                mapping[keyframe_id] = {
                    "pts_time": float(row["pts_time"]),
                    "fps": float(row["fps"]),
                    "frame_idx": int(row["frame_idx"]),
                }
            except (KeyError, ValueError):
                continue
    return mapping


def get_video_files(videos_dir: Path, groups: set[str]) -> dict[str, Path]:
    video_map: dict[str, Path] = {}
    if not videos_dir.exists():
        return video_map
    exts = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
    for path in videos_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts:
            video_id = path.stem
            group = video_id.split("_", 1)[0]
            if group in groups:
                video_map[video_id] = path
    return video_map


def get_keyframe_dirs(keyframes_dir: Path, groups: set[str]) -> dict[str, Path]:
    kf_map: dict[str, Path] = {}
    if not keyframes_dir.exists():
        return kf_map
    for item in keyframes_dir.iterdir():
        if item.is_dir():
            group = item.name.split("_", 1)[0]
            if group in groups:
                kf_map[item.name] = item
    return kf_map


def normalize_bbox(points: Any, width: int, height: int) -> tuple[tuple[float, float], ...]:
    if width <= 0 or height <= 0 or not isinstance(points, (list, tuple)) or len(points) != 4:
        raise ValueError("OCR bbox must contain four image-space points")
    normalized = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("OCR bbox point must contain x and y")
        x = max(0.0, min(1.0, float(point[0]) / width))
        y = max(0.0, min(1.0, float(point[1]) / height))
        normalized.append((x, y))
    return tuple(normalized)


def run_asr_extraction(
    video_map: dict[str, Path],
    whisper_model_dir: Path,
    model_name: str,
    output_path: Path,
    max_videos: int | None = None,
    run_id: str = "v1",
) -> int:
    print(f"[ASR] Initializing Whisper model '{model_name}' from '{whisper_model_dir}'...", flush=True)
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("[ASR] faster_whisper not installed in current environment.", flush=True)
        return 0

    model_path = whisper_model_dir / model_name
    if not model_path.exists():
        candidates = list(whisper_model_dir.glob(f"*{model_name}*"))
        if candidates:
            model_path = candidates[0]
        else:
            model_path = Path(model_name)

    if (model_path / "snapshots").is_dir():
        snapshots = [s for s in (model_path / "snapshots").iterdir() if s.is_dir()]
        if snapshots:
            model_path = snapshots[0]

    print(f"[ASR] Loading Whisper weights from: {model_path}", flush=True)
    model = WhisperModel(str(model_path), device="cpu", compute_type="int8", cpu_threads=6)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    video_keys = sorted(video_map.keys())
    if max_videos:
        video_keys = video_keys[:max_videos]

    with output_path.open("w", encoding="utf-8") as handle:
        for video_id in video_keys:
            video_file = video_map[video_id]
            print(f"[ASR] Transcribing {video_id} ({video_file.name})...", flush=True)
            try:
                segments_generator, info = model.transcribe(
                    str(video_file),
                    language="vi",
                    beam_size=1,
                    vad_filter=True,
                    word_timestamps=False,
                )
                segments = []
                for idx, seg in enumerate(segments_generator):
                    logprob = float(getattr(seg, "avg_logprob", 0.0))
                    conf = round(max(0.0, min(1.0, math.exp(logprob))), 3)
                    segments.append(
                        AsrSegment(
                            segment_id=idx,
                            start_time=round(float(seg.start), 3),
                            end_time=round(float(seg.end), 3),
                            text=seg.text.strip(),
                            confidence=conf,
                        )
                    )
                    if (idx + 1) % 20 == 0:
                        print(f"  [{video_id}] Transcribed {idx + 1} segments...", flush=True)
                max_end = max((seg.end_time for seg in segments), default=0.0)
                dur = max(round(float(info.duration or 0.0), 3), round(max_end, 3))
                transcript = AsrTranscript(
                    video_id=video_id,
                    status="AVAILABLE" if segments else "NO_AUDIO",
                    segments=tuple(segments),
                    language=info.language or "vi",
                    duration=dur,
                    run_id=run_id,
                )
            except Exception as exc:
                print(f"[ASR] Warning: Transcription error for {video_id}: {exc}", flush=True)
                transcript = AsrTranscript(
                    video_id=video_id,
                    status="ERROR",
                    segments=(),
                    language="vi",
                    duration=0.0,
                    run_id=run_id,
                )

            line = json.dumps(
                {
                    "video_id": transcript.video_id,
                    "status": transcript.status,
                    "language": transcript.language,
                    "duration": transcript.duration,
                    "run_id": transcript.run_id,
                    "segments": [
                        {
                            "segment_id": seg.segment_id,
                            "start_time": seg.start_time,
                            "end_time": seg.end_time,
                            "text": seg.text,
                            "confidence": seg.confidence,
                        }
                        for seg in transcript.segments
                    ],
                },
                ensure_ascii=False,
            )
            handle.write(line + "\n")
            handle.flush()
            count += 1

    print(f"[ASR] Finished! Wrote {count} transcripts to {output_path}", flush=True)
    return count


def run_ocr_extraction(
    kf_map: dict[str, Path],
    map_kf_dir: Path,
    output_path: Path,
    engine_name: str = "auto",
    max_videos: int | None = None,
    run_id: str = "v1",
) -> int:
    print(f"[OCR] Initializing OCR engine ({engine_name})...", flush=True)
    ocr_predictor = None
    if engine_name in {"auto", "paddleocr"}:
        try:
            from paddleocr import PaddleOCR
            # `show_log` was removed in PaddleOCR 3.x; keep to arguments
            # supported by both the installed runtime and older releases.
            ocr_predictor = PaddleOCR(use_angle_cls=False, lang="vi")
            print("[OCR] Using PaddleOCR (lang='vi')", flush=True)
        except ImportError:
            print("[OCR] paddleocr not installed.", flush=True)

    if ocr_predictor is None and engine_name in {"auto", "easyocr"}:
        try:
            import easyocr
            ocr_predictor = easyocr.Reader(["vi", "en"], gpu=False)
            print("[OCR] Using EasyOCR (['vi', 'en'])", flush=True)
        except ImportError:
            print("[OCR] easyocr not installed.", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    kf_keys = sorted(kf_map.keys())
    if max_videos:
        kf_keys = kf_keys[:max_videos]

    with output_path.open("w", encoding="utf-8") as handle:
        for video_id in kf_keys:
            kf_dir = kf_map[video_id]
            mapping = parse_map_keyframes(map_kf_dir / f"{video_id}.csv")
            image_files = sorted(kf_dir.glob("*.jpg")) + sorted(kf_dir.glob("*.png"))

            for img_path in image_files:
                try:
                    keyframe_id = int(img_path.stem)
                except ValueError:
                    keyframe_id = 0

                meta = mapping.get(keyframe_id, {"pts_time": 0.0, "fps": 25.0, "frame_idx": 0})
                detections = []

                if ocr_predictor is not None:
                    try:
                        with Image.open(img_path) as image:
                            image_width, image_height = image.size
                        if hasattr(ocr_predictor, "ocr"):
                            result = ocr_predictor.ocr(str(img_path), cls=False)
                            if result and result[0]:
                                for line in result[0]:
                                    bbox_raw, (text, conf) = line
                                    detections.append(
                                        OcrDetection(
                                            text=text.strip(),
                                            confidence=round(float(conf), 3),
                                            bbox=normalize_bbox(bbox_raw, image_width, image_height),
                                        )
                                    )
                        elif hasattr(ocr_predictor, "readtext"):
                            result = ocr_predictor.readtext(str(img_path))
                            for bbox_raw, text, conf in result:
                                text = text.strip()
                                if not text:
                                    continue
                                detections.append(
                                    OcrDetection(
                                        text=text,
                                        confidence=round(float(conf), 3),
                                        bbox=normalize_bbox(bbox_raw, image_width, image_height),
                                    )
                                )
                    except Exception as exc:
                        print(f"[OCR] Warning: OCR error on {img_path}: {exc}", flush=True)

                ocr_frame = OcrFrame(
                    video_id=video_id,
                    keyframe_id=keyframe_id,
                    frame_idx=int(meta["frame_idx"]),
                    pts_time=float(meta["pts_time"]),
                    fps=float(meta["fps"]),
                    detections=tuple(detections),
                    run_id=run_id,
                )

                line = json.dumps(
                    {
                        "video_id": ocr_frame.video_id,
                        "keyframe_id": ocr_frame.keyframe_id,
                        "frame_idx": ocr_frame.frame_idx,
                        "pts_time": ocr_frame.pts_time,
                        "fps": ocr_frame.fps,
                        "run_id": ocr_frame.run_id,
                        "detections": [
                            {
                                "text": det.text,
                                "confidence": det.confidence,
                                "bbox": det.bbox,
                            }
                            for det in ocr_frame.detections
                        ],
                    },
                    ensure_ascii=False,
                )
                handle.write(line + "\n")
                handle.flush()
                count += 1

    print(f"[OCR] Finished! Wrote {count} OCR frames to {output_path}", flush=True)
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Full corpus Phase 5 OCR and ASR evidence extractor.")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--groups", default="L21")
    parser.add_argument("--ocr-output", type=Path, default=ROOT / "artifacts" / "phase5" / "ocr_l21.jsonl")
    parser.add_argument("--asr-output", type=Path, default=ROOT / "artifacts" / "phase5" / "asr_l21.jsonl")
    parser.add_argument("--whisper-model-dir", type=Path, default=Path(r"D:\tool-video\models"))
    parser.add_argument("--whisper-model-name", default="models--Systran--faster-whisper-large-v3")
    parser.add_argument("--ocr-engine", choices=["auto", "paddleocr", "easyocr", "mock"], default="auto")
    parser.add_argument("--max-videos", type=int, default=None, help="Limit number of videos processed for quick testing")
    parser.add_argument("--skip-asr", action="store_true")
    parser.add_argument("--skip-ocr", action="store_true")

    args = parser.parse_args()
    groups = {g.strip() for g in args.groups.split(",") if g.strip()}

    print(f"=== Starting Phase 5 Evidence Extraction for groups: {sorted(groups)} ===", flush=True)
    # Datasets in this repository use `video/`; retain `videos/` support for
    # exports that follow the older plural directory convention.
    videos_dir = args.data_root / "videos"
    if not videos_dir.is_dir():
        videos_dir = args.data_root / "video"
    video_map = get_video_files(videos_dir, groups)
    kf_map = get_keyframe_dirs(args.data_root / "keyframes", groups)

    print(f"Found {len(video_map)} video files and {len(kf_map)} keyframe directories.", flush=True)

    if not args.skip_asr:
        run_asr_extraction(
            video_map=video_map,
            whisper_model_dir=args.whisper_model_dir,
            model_name=args.whisper_model_name,
            output_path=args.asr_output,
            max_videos=args.max_videos,
        )

    if not args.skip_ocr:
        run_ocr_extraction(
            kf_map=kf_map,
            map_kf_dir=args.data_root / "map-keyframes",
            output_path=args.ocr_output,
            engine_name=args.ocr_engine,
            max_videos=args.max_videos,
        )

    print("=== Extraction completed successfully ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
