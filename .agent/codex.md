## CHỈ DẪN BẮT BUỘC DÀNH CHO CODEX AGENT

Tài liệu này là nguồn ngữ cảnh trung tâm của dự án. Codex phải đọc toàn bộ tài liệu trước khi phân tích repository, đề xuất kiến trúc chi tiết hoặc thực hiện bất kỳ thay đổi nào.

Ở thời điểm hiện tại, Codex **không được viết mã nguồn**, không tạo project skeleton, không cài thư viện, không tải model, không tải dữ liệu và không thay đổi cấu trúc repository. Nhiệm vụ trước mắt chỉ gồm: hiểu bài toán, kiểm tra các giả định, phân rã hệ thống, lập kế hoạch và báo cáo những điểm còn thiếu.

Mọi thông tin trong tài liệu được chia thành ba mức độ:

1. **Thông tin chính thức từ BTC**: được lấy từ tài liệu “Thông tin vòng Sơ tuyển AIC2026”, ba buổi tập huấn và bảng “[AIC 2026] TỔNG HỢP THẮC MẮC THÍ SINH”.
2. **Suy luận kỹ thuật của nhóm**: được rút ra từ cấu trúc dữ liệu thực tế và cơ chế chấm điểm.
3. **Đề xuất kiến trúc**: được tham khảo từ tư duy chung của các hệ thống AIC/SOICT, Lifelog Search Challenge, Video Browser Showdown, VISIONE và vitrivr. Đây không phải yêu cầu bắt buộc của BTC.

Codex không được biến suy luận hoặc đề xuất thành “quy định của BTC”. Khi báo cáo, phải ghi rõ đâu là dữ kiện chính thức, đâu là giả định và đâu là đề xuất.

Trước khi được cho phép triển khai, Codex phải hoàn thành các đầu ra sau:

- Bản kiểm kê repository hiện tại.
- Bản kiểm kê dữ liệu hiện có trên máy.
- Sơ đồ dependency giữa các module dự kiến.
- Danh sách quyết định kỹ thuật cần người dùng phê duyệt.
- Danh sách rủi ro và điểm chưa được BTC trả lời.
- Lộ trình triển khai theo từng milestone.
- Tiêu chí nghiệm thu cho từng milestone.
- Đề xuất cấu hình tối thiểu và cấu hình mở rộng.
- Ước lượng tài nguyên lưu trữ, RAM, CPU, GPU và thời gian tiền xử lý ở mức tương đối.
- Kế hoạch kiểm thử không phụ thuộc vào ground truth chính thức.

Codex phải hỏi lại người dùng trước những hành động sau:

- Tải model hoặc dữ liệu có dung lượng lớn.
- Thay đổi framework backend hoặc frontend.
- Chọn vector database ngoài FAISS.
- Chọn dịch vụ cloud hoặc API trả phí.
- Re-encode video hoặc thay đổi frame rate.
- Xóa, di chuyển hoặc giải nén hàng loạt dữ liệu.
- Thêm mô hình nặng vào đường chạy online.
- Thiết kế hệ thống phụ thuộc Internet trong lúc thi.

---

## 0.1. PROTOCOL ĐỌC VÀ GHI REPORT TIẾN ĐỘ — BẮT BUỘC CHO MỌI AGENT

Repository này có nhiều người và nhiều AI Agent cùng cộng tác. Vì vậy, **report tiến độ là nguồn trạng thái vận hành chính của dự án**. Không Agent nào được bắt đầu triển khai chỉ dựa trên PLAN, README, lịch sử chat hoặc suy đoán về trạng thái repository.

### A. Quy tắc trước khi triển khai

Trước **mỗi lần** bắt đầu một task có khả năng thay đổi source code, dữ liệu, cấu hình, index, kiến trúc hoặc hành vi hệ thống, Agent bắt buộc phải thực hiện theo thứ tự:

1. Đọc file PLAN này để hiểu mục tiêu tổng thể, giới hạn và nguyên tắc kiến trúc.
2. Tìm **report có tiến độ mới nhất** của Phase hiện tại.
3. Đọc **toàn bộ report mới nhất**, không chỉ phần summary.
4. Nếu report mới nhất có dẫn chiếu đến report trước, decision cũ, issue cũ hoặc milestone cũ, Agent phải quay lại đọc đúng tài liệu được dẫn chiếu trước khi quyết định.
5. Kiểm tra repository hiện tại để xác minh report vẫn phản ánh đúng trạng thái thực tế.
6. Chỉ sau các bước trên mới được lập kế hoạch thay đổi.
7. Nếu trạng thái repository và report mâu thuẫn, **không được tự chọn một bên**. Agent phải ghi nhận mismatch, xác minh nguyên nhân và cập nhật report hoặc hỏi người dùng trước khi triển khai thay đổi có rủi ro.

### B. Thứ tự ưu tiên nguồn thông tin

Khi nhiều tài liệu nói khác nhau, dùng thứ tự ưu tiên sau:

1. **Quy định mới nhất từ BTC**, nếu liên quan luật cuộc thi hoặc dữ liệu chính thức.
2. **Report tiến độ mới nhất của Phase hiện tại**.
3. **Decision Log mới nhất đã được phê duyệt**.
4. Report cũ được report mới nhất dẫn chiếu.
5. PLAN tổng thể này.
6. README.
7. Comment trong code hoặc tài liệu cũ.
8. Suy luận của Agent.

PLAN mô tả hướng đi tổng thể. Report mới nhất mô tả **project đang thực sự ở đâu**.

Nếu PLAN và report mới nhất khác nhau vì dự án đã thay đổi có chủ đích, Agent phải ưu tiên report mới nhất và ghi nhận rằng PLAN cần được đồng bộ nếu thay đổi đó mang tính kiến trúc lâu dài.

### C. Cách xác định report mới nhất

Report phải được lưu trong thư mục thống nhất, khuyến nghị:

`reports/`

Tên report bắt buộc theo mẫu:

`Phase_[số]_[ngày-tháng-năm]_[lần].md`

Ví dụ:

- `Phase_0_11-08-2026_1.md`
- `Phase_0_11-08-2026_2.md`
- `Phase_1_14-08-2026_1.md`
- `Phase_3_21-08-2026_2.md`

Ý nghĩa:

- `Phase_[số]`: Phase đang thực hiện theo PLAN.
- `[ngày-tháng-năm]`: ngày report được tạo hoặc cập nhật thành một snapshot tiến độ mới.
- `[lần]`: số thứ tự report trong cùng ngày và cùng Phase, bắt đầu từ `1`.

Không ghi đè report cũ để thay đổi lịch sử. Khi cần một snapshot tiến độ mới, tạo report mới với số lần tăng lên.

Report mới nhất là report có:
1. Phase đang được thực hiện.
2. Ngày mới nhất.
3. Nếu cùng ngày, số `[lần]` lớn nhất.

Nếu một task liên quan nhiều Phase, Agent phải đọc report mới nhất của tất cả Phase có liên quan.

### D. Report mới nhất là checkpoint bắt buộc

Report mới nhất phải cho Agent khác trả lời được tối thiểu các câu hỏi:

- Hiện dự án đang ở Phase nào?
- Mục tiêu Phase hiện tại là gì?
- Những gì đã hoàn thành?
- Những gì mới chỉ làm một phần?
- Những gì chưa bắt đầu?
- Module/file nào vừa được thay đổi?
- Dữ liệu nào hiện có trên máy?
- Dữ liệu nào còn thiếu?
- Mapping/index nào đã được validate?
- Benchmark gần nhất là gì?
- Metric gần nhất là gì?
- Có regression nào không?
- Có bug nào đang mở?
- Có giả định nào chưa được BTC xác nhận?
- Có quyết định nào đang chờ người dùng duyệt?
- Bước tiếp theo chính xác là gì?
- Agent tiếp theo **không được làm gì**?

### E. Mọi khúc mắc và nhu cầu nâng cấp phải đi vào report mới nhất

Nếu trong quá trình đọc code, chạy test, kiểm tra dữ liệu hoặc triển khai, Agent phát hiện:

- Bug.
- Mapping không nhất quán.
- Thiếu dữ liệu.
- Technical debt.
- Module hoạt động chưa đúng luật BTC.
- Performance bottleneck.
- Thiếu test.
- Benchmark không đáng tin cậy.
- Hard-code.
- Dependency không cần thiết.
- Rủi ro về storage.
- Rủi ro Internet/API.
- Kiến trúc cần refactor.
- Một giải pháp tốt hơn nhưng chưa nên triển khai ngay.
- Một câu hỏi chưa có đủ dữ kiện để quyết định.

Agent **không được chỉ ghi trong chat hoặc giữ trong suy luận nội bộ**.

Phải ghi vào report mới nhất dưới một trong các trạng thái:

- `OPEN`
- `IN_PROGRESS`
- `BLOCKED`
- `NEEDS_DECISION`
- `DEFERRED`
- `RESOLVED`

Mỗi issue nên có:

- ID.
- Mô tả.
- Mức độ ảnh hưởng.
- Evidence.
- Module/file liên quan.
- Trạng thái.
- Giải pháp đề xuất.
- Điều kiện để đóng issue.

### F. Report mới nhất phải phân biệt DONE và “đã có code”

Một module **không được đánh dấu DONE chỉ vì file hoặc function đã tồn tại**.

Chỉ đánh dấu DONE khi:

1. Requirement đã rõ.
2. Implementation phù hợp requirement.
3. Có test hoặc bằng chứng xác minh.
4. Không còn known blocker critical.
5. Nếu có benchmark liên quan, metric đã được ghi.
6. Documentation/report đã cập nhật.

Dùng các mức:

- `NOT_STARTED`
- `PROTOTYPE`
- `PARTIAL`
- `IMPLEMENTED_UNVERIFIED`
- `VERIFIED`
- `DONE`
- `DEPRECATED`

### G. Quy tắc khi report mới nhắc lại tiến độ cũ

Report mới không cần copy toàn bộ lịch sử.

Nếu một quyết định hoặc issue cũ vẫn quan trọng, report mới được phép ghi:

`Inherited from: Phase_1_10-08-2026_2.md / ISSUE-P1-004`

Khi thấy kiểu dẫn chiếu này, Agent **bắt buộc đọc report được dẫn chiếu** trước khi sửa module liên quan.

Nếu report cũ đã bị superseded hoàn toàn, report mới phải ghi rõ:

`Supersedes: Phase_1_10-08-2026_2.md`

### H. Quy tắc cho repository nhiều người cộng tác

Mọi Agent, bất kể thuộc thành viên nào, đều phải tuân theo cùng protocol.

Không được giả định:

- Agent trước đã đọc PLAN.
- Agent trước đã chạy test.
- README là mới nhất.
- Branch hiện tại giống branch của người khác.
- Code tồn tại nghĩa là code đúng.
- Report của thành viên khác không liên quan.
- Local data của mọi người giống nhau.

Trước khi làm task, Agent phải ghi nhận tối thiểu:

- Branch hiện tại.
- Commit hoặc HEAD hiện tại.
- Report mới nhất đã đọc.
- Phase đang làm.
- Scope task.
- File/module dự kiến tác động.

### I. Quy tắc cập nhật report sau mỗi task

Sau một task có thay đổi đáng kể, Agent phải tạo report snapshot mới trước khi kết thúc.

Report mới phải có tối thiểu các mục:

1. `Current Status`
2. `What Changed`
3. `Files / Modules Affected`
4. `Data Status`
5. `Validation / Tests`
6. `Benchmark / Metrics`
7. `Open Issues`
8. `Upgrade Opportunities`
9. `Decisions Needed`
10. `Risks`
11. `References to Previous Reports`
12. `Next Recommended Action`
13. `Do Not Do Yet`
14. `Git Context`

### J. Template tối thiểu của một report

```markdown
# Phase X Progress Report

## Metadata
- Phase:
- Date:
- Revision:
- Branch:
- Commit:
- Agent/Contributor:
- Previous report:

## Current Status
...

## What Changed
...

## Files / Modules Affected
...

## Data Status
...

## Validation / Tests
...

## Benchmark / Metrics
...

## Open Issues
| ID | Issue | Severity | Status | Evidence | Next action |
|---|---|---|---|---|---|

## Upgrade Opportunities
...

## Decisions Needed
...

## Risks
...

## References to Previous Reports
...

## Next Recommended Action
...

## Do Not Do Yet
...

## Git Context
...
```

### K. Quy tắc kết thúc task

Trước khi kết thúc task, Agent phải tự kiểm tra:

- Đã đọc report mới nhất chưa?
- Có report cũ nào được dẫn chiếu mà chưa đọc không?
- Task có làm thay đổi trạng thái Phase không?
- Có issue mới nào chưa ghi vào report không?
- Có upgrade opportunity nào mới phát hiện không?
- Có benchmark/test nào cần ghi lại không?
- Có quyết định nào cần người dùng phê duyệt không?
- Report mới có giúp Agent tiếp theo tiếp tục mà không phải đoán không?

Nếu một câu trả lời là “chưa”, task chưa được coi là hoàn tất về mặt cộng tác.
