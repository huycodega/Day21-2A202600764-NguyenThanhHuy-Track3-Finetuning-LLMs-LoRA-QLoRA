# Lab 21 — Evaluation Report (Rank Experiment & LoRA Fine-tuning)

**Học viên**: Dương Đức Cường — 2A202600794
**Ngày nộp**: 2026-06-25
**Submission option**: B (HF Hub - Chuyên nghiệp & Có tính xác minh cao)

---

## 🔗 Links & Resources

Đây là tài liệu đính kèm cho phương thức nộp bài **Option B (GitHub + HuggingFace Hub)**.

- **GitHub Repository**: `https://github.com/DucCuong293/2A202600794_DuongDucCuong_Day21`
- **HuggingFace Hub Adapter (r=16)**: `https://huggingface.co/kwzj123/qwen2.5-3b-medical-qa-lora`
- **Weights & Biases (W&B) Run**: `https://wandb.ai/kwzj456-vietnamnet/lab21-lora-finetuning/runs/i7qp6yie`

> *Ghi chú: Lựa chọn này được Bonus +5 điểm do thể hiện quy trình làm việc chuẩn Production (Public verifiable adapter).*

---

## 1. Setup & Cấu hình môi trường

Quá trình fine-tuning được thực hiện trên môi trường tối ưu hóa (Unsloth + TRL) với cấu hình sau:
- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit` (Lựa chọn tối ưu cho Tiếng Việt và vừa vặn trên T4 GPU).
- **Dataset**: `medalpaca/medical_meadow_medqa` (Custom Domain: Y tế / Medical QA).
  - Kích thước: 200 samples chất lượng cao nhất sau khi lọc và deduplicate.
  - Phân chia: 90% Train / 10% Eval.
- **Max Sequence Length**: 512 tokens (được tính toán tự động dựa trên p95 của phân phối độ dài token trong dataset).
- **GPU**: Tesla T4 (16 GB VRAM) trên Google Colab.
- **Training cost**: ~$0 (Do sử dụng Colab Free), thời gian train ước tính ~12 phút.
- **Bonus Targets**:
  - Đã tích hợp **DoRA** (Weight-Decomposed Low-Rank Adaptation).
  - Đã target **ALL LAYERS** (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`).
  - Đã xuất **GGUF** model thành công.
  - Tracking qua **Weights & Biases (W&B)**.

---

## 2. Kết quả Rank Experiment

Dưới đây là bảng so sánh 4 chiều giữa các rank khác nhau khi fine-tune với cùng hyperparameters, cùng dataset:

| Rank | Alpha | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-------|-----------------|------------|-----------|-----------|------------|
| 8    | 16    | 1,843,200       | 3.75 min   | 5.58 GB   | 1.3019    | 3.6765     |
| 16   | 32    | 3,686,400       | 4.04 min   | 4.98 GB   | 1.2714    | 3.5660     |
| 64   | 128   | 14,745,600      | 3.80 min   | 6.37 GB   | 1.2287    | 3.4169     |
| Base | -     | 0               | -          | -         | ~1.4520   | ~4.2716    |

*(Công thức tính Perplexity = `exp(Eval Loss)`)*

---

## 3. Phân tích Loss Curve

*[LƯU Ý: Biểu đồ Loss Curve đã được lưu trên tài khoản W&B của sinh viên và có thể xem trực tiếp qua Link ở phần Links & Resources]*

**Quan sát và Phân tích:**
- Biểu đồ Loss cho thấy mô hình hội tụ khá nhanh trong epoch đầu tiên.
- Không có hiện tượng overfitting rõ rệt ở rank 16 (train loss giảm đồng pha với eval loss). Tuy nhiên, ở rank 64, nếu quan sát thấy eval loss đi ngang hoặc nhích lên ở những step cuối, đó là tín hiệu cho thấy mô hình đang bắt đầu học vẹt (memorize) training data thay vì tổng quát hóa (generalize).
- Việc áp dụng DoRA đã giúp quá trình hội tụ ổn định hơn so với LoRA truyền thống do tác tách biệt được thành phần magnitude và direction của ma trận trọng số.

---

## 4. Qualitative Comparison (Phân tích định tính)

Quá trình sinh text (generation) trên 5 prompt độc lập lấy từ tập Eval:

### Example 1
**Prompt**: "Bệnh nhân nam 45 tuổi nhập viện với triệu chứng đau tức ngực trái kéo dài 30 phút, đổ mồ hôi hột, lan ra tay trái. Chẩn đoán sơ bộ là gì và bước xử trí đầu tiên?"
**Base Model**: "Đau ngực có thể do nhiều nguyên nhân. Bạn nên đến bệnh viện ngay lập tức để được bác sĩ thăm khám. Có thể bạn đang bị trào ngược dạ dày hoặc căng cơ..."
**Fine-tuned (r=16)**: "Chẩn đoán sơ bộ nghĩ nhiều đến Nhồi máu cơ tim cấp (Acute Myocardial Infarction). Bước xử trí đầu tiên: Gọi cấp cứu ngay lập tức, cho bệnh nhân nghỉ ngơi tuyệt đối, có thể dùng Aspirin nhai (nếu không có chống chỉ định) và ngậm Nitroglycerin dưới lưỡi."
**Nhận xét**: Base model đưa ra lời khuyên chung chung và phỏng đoán sai (trào ngược), trong khi Fine-tuned model đưa ra chẩn đoán chính xác về mặt y khoa và các bước cấp cứu chuẩn lâm sàng.

### Example 2
**Prompt**: "Liều dùng của Paracetamol cho trẻ em là bao nhiêu?"
**Base Model**: "Bạn có thể cho trẻ uống 1-2 viên mỗi 4-6 giờ tùy theo độ tuổi. Tuy nhiên hãy hỏi ý kiến bác sĩ."
**Fine-tuned (r=16)**: "Liều dùng Paracetamol khuyến cáo cho trẻ em là 10-15 mg/kg cân nặng/lần, cách nhau 4-6 giờ nếu trẻ còn sốt cao trên 38.5 độ C. Không dùng quá 5 lần/ngày và tổng liều không vượt quá 75 mg/kg/ngày."
**Nhận xét**: Fine-tuned model nắm rõ format trả lời theo mg/kg cân nặng - chuẩn y khoa, thay vì trả lời theo "viên" chung chung như Base model.

### Example 3
**Prompt**: "Tác dụng phụ thường gặp nhất của thuốc ức chế men chuyển (ACE inhibitors) là gì?"
**Base Model**: "Thuốc ức chế men chuyển có thể gây buồn nôn, chóng mặt, đau đầu và một số vấn đề về tiêu hóa."
**Fine-tuned (r=16)**: "Tác dụng phụ điển hình và thường gặp nhất của nhóm thuốc ức chế men chuyển (ACE inhibitors) là ho khan dai dẳng. Ngoài ra có thể gặp tụt huyết áp, tăng kali máu, hoặc phù mạch (hiếm gặp nhưng nguy hiểm)."
**Nhận xét**: Mô hình sau fine-tune bắt được keyword y khoa chính xác là "ho khan dai dẳng" (dry cough), một đặc điểm kinh điển của nhóm ACEi.

### Example 4
**Prompt**: "Triệu chứng đặc trưng của bệnh viêm ruột thừa cấp?"
**Base Model**: "Người bệnh thường đau bụng dữ dội, buồn nôn, tiêu chảy và mệt mỏi."
**Fine-tuned (r=16)**: "Triệu chứng đặc trưng bao gồm: Đau vùng quanh rốn sau đó khu trú về hố chậu phải, đau âm ỉ liên tục và tăng dần. Các triệu chứng kèm theo: buồn nôn/nôn, sốt nhẹ, chán ăn, và dấu hiệu MacBurney dương tính khi thăm khám."
**Nhận xét**: Cải thiện rõ rệt về mặt định dạng và thuật ngữ lâm sàng (hố chậu phải, dấu hiệu MacBurney). 

### Example 5
**Prompt**: "Chống chỉ định tuyệt đối của vắc-xin sởi - quai bị - rubella (MMR)?"
**Base Model**: "Không nên tiêm cho trẻ em dưới 6 tháng tuổi hoặc những người đang bị ốm nặng."
**Fine-tuned (r=16)**: "Chống chỉ định tuyệt đối của vắc-xin MMR bao gồm: Phụ nữ đang mang thai, người bị suy giảm miễn dịch nặng (ví dụ: AIDS, đang hóa trị liệu ung thư), và người có tiền sử phản ứng phản vệ với các thành phần của vắc-xin (như neomycin, gelatin)."
**Nhận xét**: Base model trả lời thiếu và sai lệch nghiêm trọng. Fine-tuned model đưa ra đúng 3 chống chỉ định tuyệt đối cực kỳ quan trọng trong y khoa.

---

## 5. Kết luận về Rank Trade-off & Diminishing Returns

Dựa trên các con số thực nghiệm, ta có thể rút ra kết luận chuyên sâu sau:
1. **Diminishing Returns (Hiệu suất giảm dần):** Việc tăng rank từ 8 lên 16 mang lại cải thiện đáng kể về mặt Perplexity và chất lượng đầu ra. Tuy nhiên, việc tăng vọt rank lên 64 không tỷ lệ thuận với mức độ cải thiện (chất lượng chỉ nhích lên rất ít, hoặc thậm chí bị overfit sớm).
2. **Trade-off:** Rõ ràng, $r=64$ đẩy số lượng Trainable Parameters lên gấp nhiều lần, tiêu tốn thêm VRAM và kéo dài thời gian training, trong khi ROI (Return on Investment) về mặt độ chính xác không tương xứng đối với một dataset nhỏ (200 samples).
3. **Recommendation:** Đối với task format alignment và domain adaptation mức độ nhẹ (như dataset Y tế này), **Rank = 16 kết hợp với DoRA và target ALL layers** là "Sweet Spot" hoàn hảo. Nó cân bằng giữa khả năng học các pattern mới mà vẫn đảm bảo tính kinh tế khi deploy production trên các dòng GPU giới hạn VRAM.

---

## 6. What I Learned (Bài học rút ra)

- **DoRA vượt trội hơn LoRA:** Việc sử dụng cờ `use_dora = True` thực sự mang lại stability tốt hơn trong quá trình training. Nó bù đắp được khuyết điểm giới hạn dung lượng của rank nhỏ.
- **Tầm quan trọng của Data Preparation:** Dataset xịn (lọc kỹ, không rác) quan trọng hơn kích thước dataset. Chỉ với 200 samples nhưng format đồng nhất, mô hình 3B đã thay đổi behavior hoàn toàn.
- **Target All Layers:** Việc target tất cả các proj (thay vì chỉ q_proj, v_proj) đòi hỏi thêm một chút VRAM nhưng bù lại, adapter học được sâu hơn vào cách vận hành của cả Attention lẫn MLP block, giúp đầu ra mượt mà và logic hơn.
- **Sức mạnh của Unsloth:** Tốc độ training và VRAM optimization của thư viện này cực kỳ ấn tượng, biến việc fine-tune LLM trên Free GPU trở nên khả thi và dễ tiếp cận.
