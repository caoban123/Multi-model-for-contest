# Phase 9 Agent Workspace UI Redesign Plan

## 1. Mục tiêu

Thiết kế lại UI Agent thành một workspace dùng được trong ngày thi, trong đó:

- Kết quả Agent phải có chất lượng hiển thị tương đương hoặc tốt hơn trang Retrieval `/` hiện tại.
- Người dùng luôn nhìn thấy query gốc, query Gemini tạo ra, route CLIP/BGE/BM25 và trạng thái fallback.
- Candidate có ảnh lớn, rank, video/frame/timestamp, provenance, evidence, video preview và frame lân cận.
- Candidate được chọn có thể đi tiếp vào review và submission mà không phải nhập lại thông tin.
- KIS, Q&A và TRAKE dùng chung ngôn ngữ thiết kế nhưng giữ workflow riêng theo luật từng task.
- UI ưu tiên tốc độ thao tác, khả năng kiểm chứng và tránh nộp nhầm hơn là trang trí.

Kế hoạch này mở rộng `plan/11_PHASE_9_SUBMISSION_AGENT_UI_REPLAN.md`. Nó không thay đổi quyết định hiện tại rằng CLIP là baseline mặc định và Hybrid Retrieval vẫn opt-in cho đến khi benchmark chứng minh tốt hơn.

## 2. Phạm vi của kế hoạch

### Trong phạm vi

- Redesign Agent mode trên `/`.
- Nâng candidate rendering của `/submission` lên cùng chuẩn với `/`.
- Hiển thị Gemini response theo cách có thể audit.
- Hợp nhất query analysis, retrieval trace, candidate review và handoff sang submission.
- Đồng bộ trải nghiệm cho KIS, Q&A và TRAKE.
- Responsive desktop/tablet/mobile.
- Accessibility, loading/error/empty/fallback states.
- API contract bổ sung cho planner trace, không phá response v1.
- Test frontend tĩnh, API integration, browser acceptance và performance budget.

### Ngoài phạm vi

- Không đổi CLIP/BGE/BM25 model.
- Không promote Hybrid thành mặc định.
- Không tự suy ra official frame ID từ keyframe nội bộ.
- Không thay đổi format CSV/ZIP chính thức.
- Không thêm frontend framework trong milestone này.
- Không phụ thuộc CDN hoặc Internet để render UI trong lúc thi.
- Không dùng Gemini để tự confirm hoặc tự submit.

## 3. Audit UI hiện tại

### 3.1 Điểm đang hoạt động tốt ở `/`

- Result card có thumbnail, rank, video ID, score và metadata.
- Có provenance/evidence chips.
- Có pin, manual judgement và note.
- Có `Open video`, `Explore video` và `Neighbors`.
- Có video modal và timeline neighborhood.
- Có frame ranking và video ranking.
- Có search history và export phục vụ review.

Các capability trên phải được giữ lại khi chạy Agent. Không tạo một candidate list giản lược riêng cho Agent.

### 3.2 Điểm còn yếu của Agent mode

- Agent analysis đang là một panel nhỏ nằm trong search form.
- UI chỉ cho thấy plan đã parse, chưa phân biệt rõ Gemini trả về gì và hệ thống dùng plan nào.
- Không có stage progress rõ ràng: planning, CLIP, BGE, BM25, fusion, render.
- Health, latency và failure mới hiển thị rời rạc hoặc quá kỹ thuật.
- Contribution của từng retriever chưa đủ trực quan để user hiểu vì sao candidate đứng ở rank đó.
- Agent mode và Submission Agent chưa có chung candidate component/interaction model.

### 3.3 Điểm còn yếu của `/submission`

- Candidate là row nhỏ, ảnh và evidence ít hơn trang `/`.
- Chưa có đầy đủ title, author, date, matched text và channel contribution.
- Agent trace chỉ là các pill ngắn.
- Chưa có candidate comparison và review rationale rõ ràng.
- Q&A/TRAKE cần nhập review session ID thủ công.

## 4. Nguyên tắc UX

1. **Query là trung tâm:** query gốc luôn còn trên màn hình khi xem kết quả.
2. **AI phải minh bạch:** Gemini request state, model, raw text, parsed JSON, validated plan và fallback phải phân biệt được.
3. **Candidate là bằng chứng:** ảnh, frame, matched text và provenance phải được xem ngay tại rank hiện tại.
4. **Human confirms:** AI chỉ đề xuất; user chọn candidate, official frame và answer.
5. **Một hành động chính mỗi vùng:** Search, Select, Confirm, Validate và Download không cạnh tranh thị giác.
6. **Không nested cards:** page dùng các band/panel phẳng; card chỉ dành cho từng candidate hoặc modal/tool thật sự.
7. **Không giấu failure:** channel unavailable hoặc Gemini fallback phải hiện cạnh run hiện tại.
8. **Không làm layout nhảy:** thumbnail, toolbar, badges và action row có kích thước ổn định.
9. **Không phụ thuộc mạng cho giao diện:** icon/font cần local hoặc system fallback.
10. **Tốc độ thi đấu:** thao tác quan trọng phải dùng được bằng bàn phím và không cần cuộn ngược lên đầu trang.

## 5. Kiến trúc thông tin mới

```text
Global command bar
  - Brand / current task
  - Index + retriever + Gemini health
  - Navigation: Retrieval | Q&A | TRAKE | Submission

Query workspace
  - Query ID
  - Task selector
  - Query / Q&A split / TRAKE events
  - Agent options
  - Run button

Execution strip
  - Analyze -> Gemini -> Retrieve -> Fuse -> Ready
  - status, latency, fallback

Agent analysis
  - Final validated plan
  - Gemini inspector
  - Retriever routes and source filters

Results workspace
  - Toolbar: result mode, sort/view, count, selected count
  - Candidate grid/list
  - Candidate detail drawer
  - Pin/review/selection controls

Action dock
  - selected candidate summary
  - review choice
  - send to Q&A / TRAKE / Submission
```

## 6. Desktop layout

Mục tiêu chính là viewport từ 1366 x 768 trở lên.

### 6.1 Command bar

- Cao cố định khoảng 56-64 px.
- Bên trái: `AIC Retrieval Agent`, task hiện tại và session ID nếu có.
- Giữa: navigation dạng tabs.
- Bên phải: compact health indicators cho CLIP, BGE, BM25 và Gemini.
- Health dùng icon + nhãn ngắn, không chỉ dùng màu.

### 6.2 Query composer

- Query ID là input ngắn.
- Task là segmented control `KIS | Q&A | TRAKE`.
- Query dùng textarea 2-4 dòng để chứa đề dài.
- Q&A sau khi Analyze hiển thị hai field có thể sửa: `Event query` và `Question`.
- TRAKE sau khi Analyze hiển thị event list có thứ tự, mỗi event có visual query và Vietnamese trace query.
- `Run Agent` là action chính, giữ chiều rộng ổn định khi loading.
- Tùy chọn Gemini/Hybrid đặt trong compact settings menu; trạng thái thực tế vẫn hiển thị ở execution strip.

### 6.3 Main workspace

- Mặc định dùng hai vùng:
  - Trái 68-74%: result grid/list.
  - Phải 26-32%: Agent inspector hoặc selected candidate detail.
- Inspector có thể collapse để tăng diện tích xem ảnh.
- Không đặt result grid bên trong một card trang trí lớn.
- Sticky result toolbar và sticky action dock giúp thao tác khi danh sách dài.

## 7. Mobile và tablet

### Tablet

- Query composer chuyển thành hai hàng.
- Inspector thành drawer bên phải hoặc bottom sheet.
- Result grid 2 cột nếu đủ rộng, nếu không chuyển list.

### Mobile

- Command bar chỉ giữ task, health tổng hợp và menu.
- Result dùng một cột.
- Candidate action chuyển thành icon toolbar có tooltip/aria-label.
- Neighbor viewer dùng horizontal strip, không làm tràn viewport.
- Action dock nằm cuối màn hình nhưng không che candidate content.
- Gemini inspector mở bằng bottom sheet toàn chiều rộng.

## 8. Gemini Transparency Inspector

Đây là yêu cầu bắt buộc của redesign.

### 8.1 Các lớp thông tin phải hiển thị

1. **Original input**: nội dung người dùng nhập.
2. **Gemini status**: called, skipped, unavailable, failed, fallback hoặc succeeded.
3. **Provider/model**: ví dụ `Gemini / gemini-3.5-flash`.
4. **Raw model text**: phần text Gemini trả về, đã giới hạn kích thước và loại bỏ dữ liệu nhạy cảm.
5. **Parsed output**: JSON sau khi parse.
6. **Validated plan**: plan cuối cùng thực sự được gửi vào retrievers.
7. **Normalization notes**: field bị sửa, route bị loại, default được thêm hoặc local rule bổ sung.
8. **Warnings/fallback**: lỗi HTTP, JSON invalid, timeout hoặc local planner được sử dụng.
9. **Latency**: thời gian Gemini và tổng planning.

### 8.2 Cách trình bày

- Tab `Summary`: bản plan dễ đọc cho người thi.
- Tab `Gemini output`: raw text và parsed JSON.
- Tab `System trace`: validated plan, normalization, warnings và latency.
- Raw JSON dùng monospace, có nút copy và wrap text.
- Panel mặc định mở ở Summary; raw output là opt-in để giao diện không quá nặng.
- Không hiển thị API key, authorization header hoặc URL có secret.

### 8.3 Trạng thái fallback

Nếu Gemini lỗi:

```text
Gemini failed: HTTP 400
Fallback: local deterministic planner
Search continued with: CLIP + BM25
```

Candidate vẫn được render nếu retriever còn hoạt động. Fallback không được biến thành một lỗi chung khiến user tưởng search thất bại hoàn toàn.

## 9. API contract cho Agent trace

Response hiện tại giữ nguyên field v1. Thêm field mới theo kiểu additive:

```json
{
  "schema_version": "hybrid-retrieval-response-v2",
  "query_id": "query-1-kis",
  "query": "người dẫn chương trình mặc áo đỏ",
  "agent_trace": {
    "run_id": "agent-run-...",
    "provider": "gemini",
    "model": "gemini-3.5-flash",
    "requested": true,
    "status": "SUCCEEDED",
    "raw_text": "{...}",
    "parsed_output": {
      "clip_query": "a news presenter wearing a red shirt",
      "semantic_query": "người dẫn chương trình mặc áo đỏ",
      "lexical_query": "người dẫn chương trình áo đỏ",
      "routes": ["clip", "bge", "bm25"]
    },
    "validated_plan": {
      "visual_clip_query_en": "a news presenter wearing a red shirt",
      "semantic_text_query": "người dẫn chương trình mặc áo đỏ",
      "lexical_text_query": "người dẫn chương trình áo đỏ",
      "enabled_retrievers": ["clip", "bge", "bm25"],
      "fusion_method": "rrf"
    },
    "normalization_notes": [],
    "warnings": [],
    "fallback": null,
    "latency_ms": 842.1
  },
  "channel_hit_counts": {
    "clip": 100,
    "bge": 60,
    "bm25": 35
  },
  "latency_ms": {
    "planning": 842.1,
    "clip": 91.4,
    "bge": 37.2,
    "bm25": 8.3,
    "fusion": 1.7,
    "total": 984.9
  },
  "results": []
}
```

### Quy tắc an toàn cho trace

- `raw_text` chỉ chứa model text, không chứa HTTP headers.
- Giới hạn raw text, đề xuất tối đa 16 KB.
- Nếu vượt giới hạn, set `raw_text_truncated=true`.
- Không ghi API key vào log, SQLite, DOM, URL hoặc exported JSON.
- Lỗi HTTP chỉ giữ status code và message đã sanitize.
- Trace là debug/audit data, không được đưa vào official CSV.

## 10. Result parity với trang `/`

Agent result phải dùng chung normalization và component rendering với Visual result.

### 10.1 Candidate card bắt buộc

- Thumbnail với tỷ lệ ổn định.
- Rank lớn, dễ quét.
- Video ID, keyframe ID, frame index và timestamp.
- Title, author/channel và publish date nếu có.
- Final RRF rank/score.
- Retriever badges: CLIP, BGE, BM25.
- Rank từng channel, ví dụ `CLIP #2`, `BGE #9`, `BM25 -`.
- Matched source và matched text preview.
- Mapping warning khi chuẩn bị submission.
- Buttons: Pin, Select, Open video, Explore, Neighbors.
- Manual judgement: Good, Partial, Bad.
- Note ngắn.

### 10.2 Visual provenance

Mỗi retriever có màu nhấn phụ nhưng không biến UI thành một palette một màu:

- CLIP: blue.
- BGE: teal/green.
- BM25: amber.
- Fusion/final: neutral dark.
- Warning: amber/red theo severity.

Luôn kèm chữ/icon; không dựa duy nhất vào màu.

### 10.3 Candidate detail drawer

- Ảnh lớn hơn.
- Toàn bộ metadata.
- Matched ASR/OCR/object/metadata evidence.
- Channel contribution và rank trace.
- Neighborhood timeline.
- Raw result JSON trong tab debug, collapsed mặc định.
- Official frame ID là field riêng, không đồng nhất với keyframe ID hoặc frame index.

## 11. KIS workflow

```text
Input query
-> Gemini/local analysis
-> Hybrid retrieval
-> inspect result cards
-> open video/neighbors
-> select one or more candidates
-> enter verified official frame ID
-> optional candidate analysis
-> confirm to submission queue
```

UI phải giữ candidate đang chọn khi mở neighbor hoặc inspector. Khi chọn candidate khác, official frame field không được tự copy sang candidate mới.

## 12. Q&A workflow

```text
Input full prompt
-> Agent splits event query + question
-> user reviews/edits split
-> retrieve candidate videos
-> choose video/keyframes/evidence
-> optional Gemini answer with selected images + text evidence
-> show Gemini answer trace
-> user edits final answer
-> enter verified official frame ID
-> confirm reviewed record
```

Q&A cần hai Gemini inspector riêng:

- `Retrieval planner`: Gemini đã phân tích event query ra sao.
- `Answer generator`: question, evidence summary, số ảnh gửi, raw answer và warning.

Không cần hiển thị base64 ảnh trong trace. Chỉ hiển thị danh sách evidence ID/image thumbnail đã gửi.

## 13. TRAKE workflow

```text
Input prompt
-> Agent splits ordered events
-> show per-event visual/text queries
-> retrieve candidates per event
-> temporal alignment
-> show same-video sequence candidates
-> inspect neighbors for each event
-> enter N verified official frame IDs
-> validate order and same-video rule
-> confirm
```

Result nên dùng timeline row cho mỗi sequence. Mỗi event có thumbnail, timestamp, retriever provenance và neighbor action.

## 14. Submission integration

- Thêm action `Add to submission` ngay trong Agent result card/detail drawer.
- Nếu chưa có active session, UI yêu cầu start session trong cùng context.
- Candidate payload được truyền nội bộ; user không nhập lại video ID.
- Official frame ID vẫn bắt buộc manual hoặc từ BTC-certified mapping.
- Queue hiển thị Query ID, task, prediction count, mapping source và validation state.
- Có deep link từ queue quay lại Agent run/candidate đã chọn.
- `Done` vẫn freeze session, validate toàn bộ và tạo ZIP như implementation hiện tại.

## 15. Component inventory

Không thêm framework. Tách JavaScript/CSS theo trách nhiệm trong giới hạn frontend hiện tại:

```text
web/retrieval_ui/
  index.html
  styles.css
  app.js
  agent-workspace.js
  result-renderer.js
  gemini-inspector.js
  neighborhood-viewer.js

web/submission_ui/
  index.html
  styles.css
  app.js
```

Có thể dùng ES modules nếu server static handler hỗ trợ đúng MIME. Nếu thay đổi này tạo rủi ro, giữ script thường nhưng tách pure functions vào file riêng và kiểm tra load order.

Component logic chính:

- `AgentQueryComposer`
- `ExecutionProgress`
- `GeminiInspector`
- `RetrieverHealth`
- `ResultToolbar`
- `ResultCard`
- `CandidateDetailDrawer`
- `NeighborhoodStrip`
- `SelectionDock`
- `SubmissionQueuePreview`
- `Toast/InlineStatus`

Icon nên dùng một tập Lucide nhỏ được vendored local sau khi duyệt dependency; không dùng CDN trong lúc thi. Nếu chưa vendor icon, giữ text button hiện tại cho đến milestone polish cuối.

## 16. State model phía frontend

```text
IDLE
ANALYZING
PLANNED
RETRIEVING
FUSING
READY
PARTIAL_READY
FAILED
```

State chính:

```json
{
  "activeTask": "KIS",
  "queryId": "query-1-kis",
  "query": "...",
  "runId": null,
  "runState": "IDLE",
  "agentTrace": null,
  "results": [],
  "selectedCandidateIds": [],
  "pinnedCandidateIds": [],
  "activeCandidateId": null,
  "submissionSessionId": null,
  "errors": [],
  "warnings": []
}
```

Mỗi lần chạy query mới phải abort hoặc bỏ qua response cũ bằng `runId`/request token để response chậm không ghi đè kết quả mới.

## 17. Error, empty và loading states

### Loading

- Skeleton giữ đúng kích thước card.
- Execution strip cho biết stage đang chạy.
- Không đổi text button làm thay đổi chiều rộng.

### Partial success

Ví dụ BGE lỗi nhưng CLIP/BM25 vẫn chạy:

```text
Results ready from CLIP + BM25
BGE unavailable: index/model could not be loaded
```

### Empty

- Phân biệt `no result` với `all retrievers failed`.
- Gợi ý cụ thể: giảm lexical constraint, dùng visual query khác hoặc kiểm tra coverage ASR/OCR.

### Gemini error

- Hiện provider/model/status.
- Hiện fallback planner đã dùng.
- Có nút retry planning, không bắt buộc chạy lại toàn bộ nếu API contract hỗ trợ.

## 18. Accessibility

- Tất cả input có label thật.
- Tab/segmented control dùng đúng `role`, `aria-selected` và keyboard arrows.
- Drawer/modal giữ focus trap và trả focus về trigger khi đóng.
- Status dùng `aria-live` hợp lý, không đọc lại toàn bộ result list.
- Button icon có tooltip và `aria-label`.
- Contrast đạt WCAG AA cho text/control chính.
- Không dùng màu là dấu hiệu duy nhất.
- Focus outline luôn nhìn thấy.
- Candidate selection dùng checkbox/radio đúng semantics theo task.

## 19. Performance budget

- First UI render từ local server: mục tiêu dưới 500 ms, không tính model startup.
- Sau khi API trả response, render 30 candidates: mục tiêu dưới 150 ms trên máy thi.
- Không render toàn bộ raw JSON cho mọi card.
- Thumbnail dùng lazy loading ngoài viewport.
- Neighborhood chỉ fetch khi user mở.
- Debounce note/local persistence.
- Raw Gemini output và result debug JSON chỉ tạo DOM khi inspector được mở.
- Không duplicate ảnh base64 trong Agent response; dùng URL local hiện tại.

## 20. Milestone triển khai

### UI0 - Baseline audit và visual contract

- Chụp desktop/mobile hiện trạng `/`, `/submission`, `/trake`.
- Ghi inventory DOM/API/state hiện tại.
- Chốt result fields dùng chung.
- Chốt design tokens và interaction map.

Gate:

- Không đổi code behavior.
- Có screenshot baseline và checklist parity.

### UI1 - Agent trace backend contract

- Thêm typed trace object cho Gemini planner.
- Lưu raw model text đã sanitize.
- Phân biệt raw, parsed và validated plan.
- Giữ v1 fields để không phá UI/test cũ.
- Thêm timeout/fallback/normalization diagnostics.

Gate:

- Unit test success, invalid JSON, HTTP error, timeout và local fallback.
- Test xác nhận API key/header không xuất hiện trong response/log.

### UI2 - Shared result normalization

- Tách normalization giữa visual và agent result.
- Một result shape chung cho card renderer.
- Chuẩn hóa image URL, frame fields, metadata, provenance và matched text.

Gate:

- Visual baseline output không đổi.
- Agent results render cùng dữ liệu cốt lõi như Visual results.

### UI3 - Workspace shell và responsive layout

- Tạo command bar, query composer, execution strip và main workspace.
- Thiết lập tokens màu, spacing, typography và stable dimensions.
- Implement desktop/tablet/mobile layout.

Gate:

- Không overlap ở 360 x 800, 768 x 1024, 1366 x 768 và 1920 x 1080.
- Search controls dùng được hoàn toàn bằng bàn phím.

### UI4 - Gemini inspector

- Summary, Gemini output và System trace tabs.
- Copy JSON, raw text wrap, fallback warning.
- Provider/model/latency/status.

Gate:

- User nhìn được chính xác model text và plan cuối.
- Raw output không chứa secret.
- Empty/error/fallback states có test.

### UI5 - Result parity

- Agent dùng full result cards như `/`.
- Provenance badges, matched evidence, pin/review/note.
- Open video, Explore và Neighbors.
- Grid/list toggle nếu cần sau usability test.

Gate:

- Mọi capability chính của result card `/` có trong Agent mode.
- Không duplicate event listeners hoặc renderer logic.

### UI6 - Candidate detail và selection dock

- Detail drawer với ảnh/evidence/provenance/debug.
- Sticky selected summary.
- Official frame input tách riêng.
- Candidate comparison tối đa 3 candidate là P1 nếu không ảnh hưởng tốc độ.

Gate:

- Selection không mất khi xem video/neighbors.
- Không thể confirm khi thiếu official mapping.

### UI7 - Q&A và TRAKE parity

- Q&A dùng candidate cards và dual Gemini inspector.
- TRAKE dùng sequence timeline và per-event evidence.
- Thêm reviewed-session picker thay cho nhập ID tay.

Gate:

- Q&A/TRAKE import không yêu cầu copy session ID thủ công.
- Existing workflow tests vẫn pass.

### UI8 - Submission handoff

- Add to submission từ candidate context.
- Queue preview và deep link về Agent run.
- Validate/Done/Download rõ ràng.

Gate:

- KIS, Q&A, TRAKE tạo ZIP đúng format hiện tại.
- Không có debug field lọt vào CSV.

### UI9 - Accessibility, performance và visual polish

- Keyboard/focus/aria audit.
- Loading skeleton, toast/inline status consistency.
- Local icon bundle nếu được duyệt.
- Performance profiling 30/50 candidates.

Gate:

- Accessibility checklist pass.
- Performance budget đạt.
- Không cần Internet để render UI.

### UI10 - Acceptance và rollout

- Full test suite.
- Browser smoke trên server thật.
- Screenshot desktop/mobile cho cả success, fallback, empty và error.
- User acceptance với ít nhất 10 query KIS, 5 Q&A, 3 TRAKE.
- Rollback flag để quay về UI cũ trong thời gian thử nghiệm.

Gate:

- Không regression baseline search.
- Submission ZIP kiểm tra độc lập pass.
- Report mới ghi đầy đủ benchmark, issue và quyết định.

## 21. Test matrix

### Static/frontend tests

- Required DOM IDs tồn tại.
- Mỗi ID chỉ xuất hiện một lần.
- Event listeners không đăng ký trùng.
- Escape untrusted Gemini/query/evidence text.
- Tab/drawer/dialog semantics.
- JavaScript syntax check.

### Backend/API tests

- Gemini success/invalid JSON/empty response/HTTP 400/timeout.
- Planner fallback giữ search hoạt động.
- Partial retriever failure.
- v1 compatibility.
- Agent trace redaction và truncation.
- Result normalization cho frame-level và video-level hit.

### Browser acceptance

- Search Vietnamese query với Gemini success.
- Search khi Gemini không có key.
- Search khi Gemini trả JSON sai.
- Search khi một retriever unavailable.
- Open video và seek đúng timestamp.
- Open neighborhood và giữ selection.
- Pin/judge/note/export.
- Add KIS candidate vào submission.
- Q&A selected evidence -> Gemini answer -> confirm.
- TRAKE sequence -> official frames -> confirm.
- Validate và download ZIP.

### Viewports

- 360 x 800.
- 390 x 844.
- 768 x 1024.
- 1366 x 768.
- 1536 x 864.
- 1920 x 1080.

## 22. Benchmark và regression gates

UI redesign không được làm thay đổi ranking. Đối với cùng request/API settings:

- Query plan phải giống trước redesign, trừ field trace mới.
- Candidate IDs và thứ tự phải giống.
- RRF score/provenance phải giống.
- CLIP baseline khi Agent/Hybrid tắt phải giống.
- Official CSV/ZIP phải byte-equivalent với cùng confirmed predictions, ngoại trừ timestamp/path không nằm trong official file.

Theo dõi thêm:

- Planner latency.
- Per-channel latency.
- API payload size.
- Client render time.
- Neighborhood load time.
- Số click từ query đến confirmed prediction.

## 23. Rollout và feature flags

Đề xuất:

```text
--enable-agent-workspace-v2
```

- V1 tiếp tục tồn tại trong giai đoạn UI1-UI9.
- V2 dùng cùng backend retrieval và official submission builder.
- Chỉ đổi default sau browser acceptance.
- Nếu có lỗi trong ngày thi, tắt V2 và dùng UI cũ mà không rebuild index.

## 24. File/module dự kiến tác động

```text
src/aic_retrieval/hybrid_query_planner.py
src/aic_retrieval/hybrid_engine.py
src/aic_retrieval/retrieval_ui.py
src/aic_retrieval/qa_gemini_answering.py
src/aic_retrieval/submission_session.py
tools/retrieval_ui.py

web/retrieval_ui/index.html
web/retrieval_ui/styles.css
web/retrieval_ui/app.js
web/retrieval_ui/agent-workspace.js          # nếu tách module
web/retrieval_ui/result-renderer.js          # nếu tách module
web/retrieval_ui/gemini-inspector.js         # nếu tách module

web/submission_ui/index.html
web/submission_ui/styles.css
web/submission_ui/app.js

tests/test_hybrid_query_planner.py
tests/test_hybrid_engine.py
tests/test_retrieval_ui.py
tests/test_retrieval_ui_static.py
tests/test_submission_api.py
tests/test_submission_ui_static.py
```

Không quyết định trước rằng phải tạo tất cả file mới; việc tách file chỉ được thực hiện khi giảm complexity thật sự và không phá static serving hiện tại.

## 25. Trạng thái triển khai ngày 20-08-2026

| Milestone | Trạng thái | Bằng chứng / ghi chú |
|---|---|---|
| UI0 | `PARTIAL` | Đã audit DOM/API/state; server endpoint smoke pass, chưa có browser screenshot automation |
| UI1 | `VERIFIED` | Agent trace raw/parsed/validated, sanitize/size limit, success/fallback tests; sửa lỗi live Gemini plan bị fallback |
| UI2 | `VERIFIED_BEHAVIOR` | Agent `/` dùng result renderer hiện tại; Submission API nhận metadata/provenance/trace, chưa tách shared JS module |
| UI3 | `IMPLEMENTED_STATIC_VERIFIED` | Workspace shell, execution strip, responsive CSS; static/syntax test pass, chưa visual screenshot |
| UI4 | `VERIFIED` | Gemini Inspector trên `/` và `/submission`, raw JSON + parsed JSON + validated plan + fallback |
| UI5 | `VERIFIED_BEHAVIOR` | Agent result cards trên `/`; `/submission` có ảnh, RRF, provenance, metadata, video và neighbors |
| UI6 | `PARTIAL` | Submission selection dock và official-frame guard hoạt động; candidate comparison/detail drawer chưa làm |
| UI7 | `PARTIAL_INHERITED` | Q&A/TRAKE workflow cũ vẫn hoạt động; reviewed-session picker và parity redesign chưa làm |
| UI8 | `VERIFIED_GUARDED_INHERITED` | Submission handoff/CSV/ZIP giữ nguyên và full tests pass |
| UI9 | `PARTIAL` | Responsive/accessibility states đã bổ sung; chưa có browser visual/performance profiling |
| UI10 | `NOT_STARTED` | Chưa user acceptance 10 KIS / 5 Q&A / 3 TRAKE |

Quyết định triển khai:

- Gemini structured filter JSON là suggestion đã validate.
- Suggestion không tự thay đổi Agent ranking hiện tại.
- User phải bấm `Use in Structured Search`; UI chuyển sang Visual Structured Search, điền các field khả dụng và yêu cầu user review rồi chạy lại.
- Kênh object/attribute/OCR/ASR không khả dụng sẽ bị bỏ qua và UI cảnh báo.
- `agent_trace` không được đi vào official submission CSV/ZIP.

## 26. Issue và rủi ro

| ID | Vấn đề | Mức độ | Trạng thái | Hướng xử lý |
|---|---|---:|---|---|
| P9-UI-001 | Agent và Submission đang dùng hai kiểu candidate khác nhau | High | OPEN | Shared result contract/renderer |
| P9-UI-002 | Chưa thấy raw Gemini output và normalization | High | OPEN | Agent trace v2 + inspector |
| P9-UI-003 | Raw Gemini output có thể làm lộ dữ liệu nhạy cảm nếu log sai | Critical | OPEN | Sanitize, size limit, secret tests |
| P9-UI-004 | Response cũ có thể ghi đè query mới khi chạy nhanh liên tục | Medium | OPEN | run token/AbortController |
| P9-UI-005 | UI lớn có nguy cơ chậm khi render 30-50 candidate | Medium | OPEN | lazy render/image, performance budget |
| P9-UI-006 | Browser automation hiện chưa có bằng chứng chạy trong môi trường này | Medium | OPEN | Manual browser acceptance + Playwright khi khả dụng |
| P9-UI-007 | Official frame mapping vẫn chưa có nguồn BTC-certified | Critical | GUARDED | Bắt buộc manual/BTC mapping |
| P9-UI-008 | Q&A/TRAKE session picker chưa có | Medium | OPEN | UI7 |
| P9-UI-009 | Thêm icon library có thể tạo dependency Internet | Low | NEEDS_DECISION | Vendor local subset hoặc giữ text controls |
| P9-UI-010 | V2 có thể gây regression Visual baseline | High | OPEN | Feature flag + parity tests |

## 27. Definition of Done

Redesign chỉ được coi là hoàn thành khi:

1. Agent result hiển thị đầy đủ như Visual result hiện tại.
2. User nhìn thấy raw Gemini text, parsed output và validated final plan.
3. Gemini fallback được giải thích rõ và search vẫn hoạt động.
4. CLIP/BGE/BM25 health, latency và contribution được hiển thị.
5. Video preview, Explore và Neighbors hoạt động trong Agent mode.
6. Candidate selection đi được vào official submission workflow.
7. KIS, Q&A và TRAKE có flow nhất quán nhưng đúng luật riêng.
8. Official frame guard không bị nới lỏng.
9. Không secret nào xuất hiện trong DOM/API trace/log/export.
10. Full repository tests, browser acceptance và responsive screenshots pass.
11. Ranking baseline không thay đổi ngoài những thay đổi đã được benchmark/duyệt.
12. Có feature flag/rollback path.
13. README, PLAN và report protocol được cập nhật.

## 28. Thứ tự triển khai được khuyến nghị

```text
UI0 audit
-> UI1 Agent trace contract
-> UI2 shared result normalization
-> UI3 workspace shell
-> UI4 Gemini inspector
-> UI5 result parity
-> UI6 selection/detail
-> UI7 Q&A/TRAKE
-> UI8 submission handoff
-> UI9 polish/performance/accessibility
-> UI10 acceptance/rollout
```

Không bắt đầu bằng việc đổi màu/CSS toàn bộ. Hai dependency quan trọng nhất là Agent trace contract và shared result normalization; hoàn thành đúng hai phần này trước sẽ giúp phần UI đẹp hơn nhưng vẫn phản ánh chính xác hệ thống đang làm gì.
