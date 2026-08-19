# Project Status - 19/08/2026

## 1. Tom tat dieu hanh

Du an hien da vuot qua baseline ban dau va dang co mot he thong L21-focused kha day du cho cac luong:

- Textual KIS / visual retrieval.
- Metadata search.
- Structured retrieval opt-in.
- Phase 5 ASR/OCR store.
- Phase 6 query planning/reranking.
- Phase 7 Q&A co evidence, review va optional Gemini VLM.
- Phase 9 internal submission hardening.

Phase 8 TRAKE chua duoc trien khai. FAISS/vector database/SQLite retrieval index tong quat chua thay the NumPy index; hien tai van dung NumPy CLIP index cho visual retrieval.

Ket qua test gan nhat:

```text
python -m pytest -q
188 passed
```

## 2. Pham vi du lieu hien tai

### 2.1. L21 la scope chinh

He thong hien dang duoc chay va kiem tra chu yeu tren L21. Cac group L22-L30 co the co feature/mapping/metadata trong data tong, nhung khong duoc xem la da san sang cho visual UI neu may local khong co keyframe/video tuong ung.

### 2.2. Cac artifact quan trong

```text
artifacts/registry/data_registry.json
artifacts/indexes/l21_numpy/
artifacts/structured/l21_objects.sqlite
artifacts/structured/l21_objects_manifest.json
artifacts/phase5/ocr_l21.jsonl
artifacts/phase5/asr_l21.jsonl
artifacts/phase5/phase5_store.sqlite
artifacts/phase5/stores/phase5_l21.sqlite3
artifacts/qa/phase7_qa.sqlite3
artifacts/submissions/
```

Luu y: `data/` va `artifacts/` la du lieu local, khong nen commit len GitHub neu khong co yeu cau ro.

## 3. Registry, index va retrieval nen

### 3.1. Registry

Registry du lieu duoc tao bang:

```powershell
python tools\data_registry.py --data-root data --output artifacts\registry\data_registry.json --validation-output artifacts\registry\validation_report.json
```

Registry dung de map video, CLIP feature, keyframe, metadata, object, video raw va cac duong dan local.

### 3.2. CLIP NumPy index

Index L21 duoc dung boi UI va search tools:

```text
artifacts/indexes/l21_numpy/
```

Day van la NumPy cosine search. FAISS chua duoc dua vao pipeline chinh.

### 3.3. UI search chinh

UI co:

- Visual search.
- Metadata search.
- Video Ranking mac dinh.
- Frame Ranking debug.
- Pin/history.
- CSV export.
- Raw video preview.
- Keyframe neighborhood viewer.
- Matched-frame exploration.

Visual query hien tu dong goi Gemini de rewrite thanh English CLIP query, sau do moi search bang CLIP.

## 4. Structured retrieval

Structured retrieval da co san, nhung la opt-in/experimental. No nam o:

```text
src/aic_retrieval/structured_query.py
src/aic_retrieval/hybrid_candidates.py
src/aic_retrieval/hybrid_ranking.py
src/aic_retrieval/retrieval_ui.py
web/retrieval_ui/
```

Endpoint:

```text
/api/structured-search
```

Nguon evidence co the dung:

| Kenh | Nguon du lieu | Trang thai |
|---|---|---|
| CLIP | `artifacts/indexes/l21_numpy/` | Co |
| Object | `artifacts/structured/l21_objects.sqlite` | Co, L21 |
| Attribute/color | service color/attribute noi bo | Co |
| Metadata | `data/media-info/`, registry | Co |
| OCR | `artifacts/phase5/phase5_store.sqlite` | Co store, nhung detections rong |
| ASR | `artifacts/phase5/phase5_store.sqlite` | Co, 563 segments |

Q&A hien chua dung structured retrieval mac dinh. Q&A candidate retrieval hien dung Gemini-optimized CLIP event query.

## 5. Object data va object store

Raw object JSON da ton tai trong:

```text
data/objects/
```

Kiem tra gan nhat:

```text
177,321 file .json
```

Object store da build:

```text
artifacts/structured/l21_objects.sqlite
artifacts/structured/l21_objects_manifest.json
```

Manifest hien tai:

```text
schema_version: phase4-object-store-1.0
video_count: 29
frame_count: 7800
detection_count: 780000
coverage.ratio: 1.0
sqlite_integrity_check: ok
```

Trong raw JSON, label khong nam o key `label`. Cac key quan trong la:

```text
detection_class_names
detection_class_entities
detection_class_labels
detection_scores
detection_boxes
```

Ten object doc duoc nam trong:

```text
detection_class_entities
```

Vi du tu L21_V001:

```text
Lantern, Skyscraper, Poster, Tower, Building, Vehicle, Car, Boat, Table, Tent
Boat, Clothing, Person, Woman, Human face, Man, Girl, Television, Jacket, Furniture
```

Object detection giup nhan biet object label + bbox + confidence. No khong thay the scene understanding/VLM.

## 6. Phase 5 OCR/ASR

Artifact hien co:

```text
artifacts/phase5/ocr_l21.jsonl
artifacts/phase5/asr_l21.jsonl
artifacts/phase5/phase5_store.sqlite
artifacts/phase5/stores/phase5_l21.sqlite3
```

Trang thai kiem tra gan nhat:

```text
OCR records: 569
OCR detections: 0
ASR records: 2
ASR segments: 563
```

Ket luan:

- ASR da dung duoc cho search noi dung/lai tin tuc.
- OCR artifact co ton tai nhung chua huu dung vi detections rong.
- Cac query ve "chu tren man hinh", "bien hieu ghi gi" van yeu neu khong dung Gemini VLM nhin anh.

## 7. Phase 6 query planner va reranker

Da co:

```text
configs/phase6_reranker_v1.json
src/aic_retrieval/query_planner.py
src/aic_retrieval/reranking.py
tools/phase6_tune.py
```

UI co option hien query plan va rerank Top-N trong structured filters.

## 8. Phase 7 Q&A

Q&A hien co workspace rieng trong UI:

```text
Event query
Auto CLIP event query
Question
Candidate videos
Evidence selection
Answer review
Session history
```

### 8.1. Candidate videos

`Event query` nguoi dung nhap duoc rewrite tu dong bang Gemini thanh English CLIP-ready query. Query da rewrite duoc hien thi trong:

```text
Auto CLIP event query
```

Backend van luu event query goc de audit, nhung dung retrieval query tieng Anh de lay candidate.

### 8.2. Evidence selection

Evidence co the gom:

- keyframe image.
- CLIP rank/score.
- ASR transcript.
- OCR text neu co.
- metadata.
- object.
- attribute/color.

Evidence la nguon ma answer draft duoc phep dua vao.

### 8.3. Evidence-first answer

Mac dinh Q&A dung answerer deterministic:

```text
src/aic_retrieval/qa_answering.py
```

No tot cho:

- OCR/ASR text.
- metadata.
- object label.
- attribute/color.

No khong tu suy luan hinh anh phuc tap neu chi co keyframe.

### 8.4. Gemini VLM answer option

Checkbox UI:

```text
Draft with Gemini
```

Khi bat checkbox nay:

- backend gom selected evidence;
- tu tim keyframe image lien quan theo `video_id + keyframe_id`;
- gui selected evidence + toi da 4 anh keyframe cho Gemini;
- Gemini tra draft answer JSON;
- nguoi dung van phai Confirm/Edit/Reject.

Generation method:

```text
gemini_vlm_evidence
```

Nguyen tac an toan:

- Gemini chi la draft.
- Khong tu confirm.
- Khong tu export.
- Answer phai qua human review.

## 9. Phase 8 TRAKE

Chua trien khai.

Plan nam o:

```text
plan/09_PHASE_8_TRAKE.md
```

Can lam:

- tach query thanh event con;
- retrieve tung event;
- gom candidate theo video;
- tim chain keyframe tang dan theo thoi gian;
- tinh sequence score;
- UI timeline/manual correction.

## 10. Phase 9 submission hardening

Da bat dau.

Da co:

```text
src/aic_retrieval/submission.py
tools/export_qa_submissions.py
docs/PHASE9_SUBMISSION_HARDENING.md
tests/test_submission.py
```

Schema noi bo:

```text
aic-internal-submission-v1
```

Tool export:

```powershell
python tools\export_qa_submissions.py --qa-store artifacts\qa\phase7_qa.sqlite3
```

Output mac dinh:

```text
artifacts/submissions/qa_submission_candidates.jsonl
artifacts/submissions/qa_submission_candidates.csv
artifacts/submissions/qa_submission_validation.json
```

Trang thai hien tai:

- Neu chua co Q&A review nao duoc confirm/export trong UI, tool se bao `EMPTY_SUBMISSION`.
- Day la dung, khong phai loi pipeline.
- Adapter format BTC chinh thuc chua lam vi BTC chua chot schema cuoi.

## 11. Cach chay UI hien tai

PowerShell tai `D:\AIC1`:

```powershell
$env:AIC_CLIP_MODEL_ID="D:\AIC\.cache\huggingface\hub\models--openai--clip-vit-base-patch32\snapshots\3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
$env:AIC_TRANSLATION_API_KEY="<your-gemini-api-key>"
$env:AIC_TRANSLATION_PROVIDER="gemini"
$env:AIC_TRANSLATION_MODEL="gemini-3.5-flash"

python tools\retrieval_ui.py `
  --registry artifacts\registry\data_registry.json `
  --index-dir artifacts\indexes\l21_numpy `
  --groups L21 `
  --clip-local-files-only `
  --phase5-store artifacts\phase5\phase5_store.sqlite `
  --qa-store artifacts\qa\phase7_qa.sqlite3
```

Mo:

```text
http://127.0.0.1:8765
```

Neu port bi chiem:

```powershell
python tools\retrieval_ui.py `
  --registry artifacts\registry\data_registry.json `
  --index-dir artifacts\indexes\l21_numpy `
  --groups L21 `
  --clip-local-files-only `
  --phase5-store artifacts\phase5\phase5_store.sqlite `
  --qa-store artifacts\qa\phase7_qa.sqlite3 `
  --port 8766
```

## 12. Cach dung Q&A khuyen nghi

1. Nhap `Event query` bang tieng Viet hoac Anh.
2. Nhap `Question`.
3. Bam `Prepare evidence`.
4. Kiem tra `Auto CLIP event query`.
5. Xem `Candidate videos`.
6. Tick evidence tot trong `Evidence selection`.
7. Neu cau hoi can nhin anh/suy luan hinh anh, bat `Draft with Gemini`.
8. Bam `Draft answer from selected evidence`.
9. Sua answer neu can.
10. Confirm/Edit/Reject.
11. Bam `Export confirmed record` neu can tao record noi bo.
12. Chay `tools/export_qa_submissions.py` de tao JSONL/CSV noi bo.

## 13. Han che hien tai

- OCR hien rong detections, nen chua dung duoc cho text tren man hinh.
- Q&A candidate retrieval chua dung structured retrieval mac dinh.
- Gemini VLM can Internet/API key.
- Phase 8 TRAKE chua trien khai.
- Chua co FAISS/vector DB.
- Chua co official submission adapter.
- Data va artifacts la local, nguoi khac pull repo ve khong co san du lieu lon.

## 14. Viec nen lam tiep

Thu tu de xuat:

1. Push/code review cac thay doi hien tai neu can dong bo branch.
2. Cho Q&A candidate retrieval co option structured-lite: CLIP + object + attribute + ASR/metadata khi phu hop.
3. Cai thien OCR extraction neu cau hoi ve chu tren man hinh quan trong.
4. Bat dau Phase 8 TRAKE backend prototype.
5. Khi BTC chot format, viet adapter official submission tu `SubmissionCandidate`.

## 15. File/chuc nang moi gan day

Gan day da them/sua:

- Auto Gemini rewrite cho visual search.
- Auto Gemini rewrite cho Q&A event query.
- Q&A thumbnail display cho candidate/evidence.
- Gemini evidence answer option.
- Gemini VLM image input cho Q&A.
- Internal submission schema/export/validator.
- Phase 9 runbook.

Test suite hien tai:

```text
188 passed
```
