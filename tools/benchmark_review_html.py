from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a static HTML reviewer for text-query benchmark results.")
    parser.add_argument("--input", default="artifacts/benchmarks/l21_text_benchmark.json")
    parser.add_argument("--output", default="artifacts/benchmarks/l21_text_review.html")
    parser.add_argument("--result-set", choices=["results", "raw_results"], default="results")
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    payload["_review_result_set"] = args.result_set
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_review_html(payload, output_path), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "query_count": len(payload.get("queries", []))}, indent=2))
    return 0


def render_review_html(payload: dict[str, Any], output_path: Path) -> str:
    result_set = str(payload.get("_review_result_set", "results"))
    query_sections = "\n".join(render_query(query, output_path, result_set) for query in payload.get("queries", []))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>L21 Text Benchmark Review</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f7f4;
      --ink: #1f2428;
      --muted: #697077;
      --line: #d9ded8;
      --panel: #ffffff;
      --accent: #25636f;
      --accent-soft: #dcebef;
      --warn: #8a5a00;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: var(--bg);
      color: var(--ink);
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 2;
      background: rgba(247, 247, 244, 0.96);
      border-bottom: 1px solid var(--line);
      padding: 16px 24px;
      display: flex;
      gap: 16px;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
    }}
    h1 {{
      font-size: 20px;
      line-height: 1.2;
      margin: 0;
    }}
    .meta {{
      color: var(--muted);
      font-size: 13px;
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
    }}
    button {{
      border: 1px solid var(--accent);
      background: var(--accent);
      color: white;
      border-radius: 6px;
      padding: 9px 12px;
      font-weight: 700;
      cursor: pointer;
    }}
    main {{
      padding: 18px 24px 36px;
      max-width: 1440px;
      margin: 0 auto;
    }}
    section {{
      border-top: 1px solid var(--line);
      padding: 18px 0 24px;
    }}
    .query-head {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: baseline;
      margin-bottom: 12px;
    }}
    .query-title {{
      font-size: 18px;
      margin: 0;
    }}
    .query-notes {{
      color: var(--muted);
      font-size: 13px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
    }}
    article {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      min-width: 0;
    }}
    img {{
      width: 100%;
      aspect-ratio: 16 / 9;
      object-fit: contain;
      background: #111;
      display: block;
    }}
    .result-body {{
      padding: 10px;
      display: grid;
      gap: 8px;
    }}
    .result-title {{
      display: flex;
      justify-content: space-between;
      gap: 8px;
      font-size: 14px;
      font-weight: 700;
    }}
    .path {{
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }}
    .choices {{
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
    }}
    label {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 5px 8px;
      font-size: 12px;
      cursor: pointer;
    }}
    input[type="radio"] {{
      margin-right: 4px;
    }}
    textarea {{
      width: 100%;
      min-height: 54px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 7px;
      font: inherit;
      font-size: 13px;
    }}
    .missing {{
      min-height: 124px;
      display: grid;
      place-items: center;
      color: var(--warn);
      background: #fff7dd;
      font-size: 13px;
      padding: 16px;
      text-align: center;
    }}
    @media (max-width: 640px) {{
      header, main {{ padding-left: 14px; padding-right: 14px; }}
      .query-head {{ display: block; }}
    }}
  </style>
</head>
<body>
  <header>
    <div>
      <h1>L21 Text Benchmark Review</h1>
      <div class="meta">
        <span>Queries: {html.escape(str(payload.get("query_count", len(payload.get("queries", [])))))}</span>
        <span>Top-K: {html.escape(str(payload.get("top_k", "")))}</span>
        <span>Review set: {html.escape(result_set)}</span>
        <span>Index vectors: {html.escape(str(payload.get("index_vectors", "")))}</span>
      </div>
    </div>
    <button type="button" onclick="exportCsv()">Export CSV</button>
  </header>
  <main>
    {query_sections}
  </main>
  <script>
    function csvEscape(value) {{
      const text = String(value ?? "");
      return '"' + text.replaceAll('"', '""') + '"';
    }}
    function exportCsv() {{
      const rows = [[
        "query_id", "query_text", "rank", "score", "video_id", "keyframe_id",
        "frame_idx", "pts_time", "keyframe_path", "manual_judgement", "manual_notes"
      ]];
      document.querySelectorAll("[data-result]").forEach((node) => {{
        const selected = node.querySelector("input[type=radio]:checked");
        const notes = node.querySelector("textarea");
        rows.push([
          node.dataset.queryId,
          node.dataset.queryText,
          node.dataset.rank,
          node.dataset.score,
          node.dataset.videoId,
          node.dataset.keyframeId,
          node.dataset.frameIdx,
          node.dataset.ptsTime,
          node.dataset.keyframePath,
          selected ? selected.value : "",
          notes ? notes.value : ""
        ]);
      }});
      const csv = rows.map((row) => row.map(csvEscape).join(",")).join("\\n");
      const blob = new Blob([csv], {{ type: "text/csv;charset=utf-8" }});
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "l21_text_benchmark_judged.csv";
      link.click();
      URL.revokeObjectURL(url);
    }}
  </script>
</body>
</html>
"""


def render_query(query: dict[str, Any], output_path: Path, result_set: str = "results") -> str:
    query_id = str(query.get("id", ""))
    query_text = str(query.get("text", ""))
    notes = str(query.get("notes", ""))
    results = query.get(result_set, query.get("results", []))
    cards = "\n".join(render_result(query_id, query_text, result, output_path) for result in results)
    return f"""<section>
  <div class="query-head">
    <h2 class="query-title">{html.escape(query_id)} - {html.escape(query_text)}</h2>
    <div class="query-notes">{html.escape(notes)}</div>
  </div>
  <div class="grid">
    {cards}
  </div>
</section>"""


def render_result(query_id: str, query_text: str, result: dict[str, Any], output_path: Path) -> str:
    keyframe_path = str(result.get("keyframe_path") or "")
    image_src = image_src_for(keyframe_path, output_path)
    rank = str(result.get("rank", ""))
    video_id = str(result.get("video_id", ""))
    keyframe_id = str(result.get("keyframe_id", ""))
    score = f"{float(result.get('score', 0.0)):.6f}"
    image_html = (
        f'<img src="{html.escape(image_src)}" alt="{html.escape(video_id)} keyframe {html.escape(keyframe_id)}">'
        if image_src
        else '<div class="missing">No keyframe image path</div>'
    )

    return f"""<article
  data-result
  data-query-id="{html.escape(query_id)}"
  data-query-text="{html.escape(query_text)}"
  data-rank="{html.escape(rank)}"
  data-score="{html.escape(score)}"
  data-video-id="{html.escape(video_id)}"
  data-keyframe-id="{html.escape(keyframe_id)}"
  data-frame-idx="{html.escape(str(result.get("frame_idx", "")))}"
  data-pts-time="{html.escape(str(result.get("pts_time", "")))}"
  data-keyframe-path="{html.escape(keyframe_path)}">
  {image_html}
  <div class="result-body">
    <div class="result-title">
      <span>#{html.escape(rank)} {html.escape(video_id)} / {html.escape(keyframe_id)}</span>
      <span>{html.escape(score)}</span>
    </div>
    <div class="path">{html.escape(keyframe_path)}</div>
    <div class="choices" aria-label="Manual judgement">
      <label><input type="radio" name="{html.escape(query_id)}-{html.escape(rank)}" value="good">good</label>
      <label><input type="radio" name="{html.escape(query_id)}-{html.escape(rank)}" value="partial">partial</label>
      <label><input type="radio" name="{html.escape(query_id)}-{html.escape(rank)}" value="bad">bad</label>
    </div>
    <textarea placeholder="manual notes"></textarea>
  </div>
</article>"""


def image_src_for(keyframe_path: str, output_path: Path) -> str:
    if not keyframe_path:
        return ""
    path = Path(keyframe_path)
    if not path.is_absolute():
        path = ROOT / path
    try:
        return Path(os.path.relpath(path.resolve(), output_path.parent.resolve())).as_posix()
    except ValueError:
        return path.resolve().as_uri()


if __name__ == "__main__":
    raise SystemExit(main())
