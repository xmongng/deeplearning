# RealWaste Edge Quantization

Dự án thực nghiệm 3 kiến trúc `MobileNetV2`, `MobileNetV3-Large`, `ShuffleNetV2 x1.0` trên RealWaste (4.752 ảnh, 9 lớp), với 9 cấu hình FP32/PTQ INT8/QAT INT8. FP32 và QAT được train bằng CUDA trên Kaggle. PTQ, chuyển đổi W8A8 và export `.pte` dùng CPU/XNNPACK vì đây là backend triển khai Edge cần đánh giá.

## Dữ liệu và giao thức đã khóa

- Dataset: RealWaste, 9 lớp.
- Split cố định theo seed 42: 3.802 train, 476 validation, 474 test.
- Calibration PTQ: 200 ảnh chỉ lấy từ train, có đủ 9 lớp.
- Các cặp gần trùng pHash được giữ trong cùng split để tránh leakage.
- Test không được dùng để chọn checkpoint hoặc calibration.
- Best FP32 và QAT được chọn bằng Macro-F1 trên validation.
- Accuracy cuối được đo từ chính file ExecuTorch `.pte`, không đo từ model trung gian.

Audit tự động đã đọc được đủ 4.752 ảnh, không có ảnh hỏng. Có 14 cặp ứng viên gần trùng; chúng được gom nhóm bảo thủ khi chia tập. `manual_review_complete=false` trong `data/splits/summary.json`, vì vậy báo cáo cuối phải ghi rõ chưa hoàn tất rà soát nhãn bằng mắt. Có thể mở `data/review.html` để rà soát sau.

## 1. Chuẩn bị Kaggle

Tạo một Kaggle Dataset riêng chứa đúng file:

```text
realwaste.zip
```

Đặt slug/directory của Kaggle Dataset là `realwaste`, để file xuất hiện tại:

```text
/kaggle/input/realwaste/realwaste.zip
```

Nếu Kaggle tạo đường dẫn khác, sửa duy nhất trường `data.archive` trong `configs/kaggle.yaml`.

Nén toàn bộ thư mục dự án `waste-edge-quantization` thành ZIP và upload nó thành một Kaggle Dataset thứ hai, hoặc đưa mã nguồn lên GitHub. Không cần đưa thư mục `.venv`, `data/raw`, `results/smoke` vào ZIP.

Tạo Kaggle Notebook, chọn:

- Accelerator: GPU (T4 x2, P100 hoặc GPU khả dụng).
- Internet: On, vì TorchVision cần tải pretrained ImageNet lần đầu.
- Persistence: nên bật để giảm rủi ro mất checkpoint.

Kiểm tra GPU trong một cell:

```bash
!nvidia-smi
```

## 2. Đưa source vào `/kaggle/working`

Nếu upload source dưới dạng Kaggle Dataset ZIP, xem tên file bằng:

```bash
!find /kaggle/input -maxdepth 3 -type f | sort | head -100
```

Sau đó giải nén, ví dụ:

```bash
!unzip -q /kaggle/input/waste-edge-project/waste-edge-quantization.zip -d /kaggle/working
%cd /kaggle/working/waste-edge-quantization
```

Nếu dùng GitHub:

```bash
%cd /kaggle/working
!git clone <URL_REPOSITORY_CUA_BAN> waste-edge-quantization
%cd /kaggle/working/waste-edge-quantization
```

## 3. Cài môi trường đã khóa

Chạy cell sau, rồi **Restart Session** nếu Kaggle yêu cầu do thay phiên bản PyTorch:

```bash
!python -m pip install --no-cache-dir -r requirements-kaggle.txt
```

Sau khi restart, `%cd` lại thư mục dự án và kiểm tra:

```bash
%cd /kaggle/working/waste-edge-quantization
!python -c "import torch,torchvision,torchao,executorch; print(torch.__version__, torchvision.__version__, torchao.__version__); print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

Không tiếp tục nếu `CUDA: False`.

## 4. Chạy toàn bộ pipeline

Lệnh duy nhất:

```bash
!python -u -m src.run_kaggle --config configs/kaggle.yaml 2>&1 | tee /kaggle/working/kaggle-training.log
```

Pipeline lần lượt:

1. Giải nén và đối chiếu SHA-256 của đủ 4.752 ảnh với audit đã khóa.
2. Fine-tune 3 baseline FP32 trên CUDA.
3. Export 3 FP32 sang ExecuTorch/XNNPACK.
4. PTQ W8A8 bằng 200 ảnh calibration train và export 3 model.
5. QAT trên CUDA từ đúng checkpoint FP32 tương ứng, đánh giá validation sau convert trên CPU, rồi export 3 model.
6. Đánh giá 9 file `.pte` trên cùng test set.
7. Tạo `/kaggle/working/waste-edge-complete.zip`.

Không chạy đồng thời hai model trong cùng GPU. Script chạy tuần tự để giảm nguy cơ hết VRAM và ghi log liên tục.

## 5. Chạy từng bước khi cần khôi phục

Nếu phiên Kaggle dừng giữa chừng, tải checkpoint đã có lên lại `/kaggle/working/waste-edge-quantization/checkpoints` rồi chạy riêng model còn thiếu. Ví dụ MobileNetV2:

```bash
!python -m src.verify_data --config configs/kaggle.yaml
!python -u -m src.train_fp32 --config configs/kaggle.yaml --model mobilenet_v2 --device cuda
!python -u -m src.export_edge --config configs/kaggle.yaml --model mobilenet_v2
!python -u -m src.ptq_int8 --config configs/kaggle.yaml --model mobilenet_v2
!python -u -m src.qat_int8 --config configs/kaggle.yaml --model mobilenet_v2 --device cuda
```

Thay `mobilenet_v2` bằng `mobilenet_v3_large` hoặc `shufflenet_v2_x1_0`.

Các script cố ý từ chối ghi đè checkpoint/artifact đã tồn tại. Điều này tránh trộn kết quả từ hai lần chạy. Muốn làm thí nghiệm mới, dùng một bản sao project/output mới thay vì xóa lẻ từng file.

Nếu QAT đã train xong và có `qat_summary.json` nhưng export bị ngắt:

```bash
!python -u -m src.qat_int8 --config configs/kaggle.yaml --model mobilenet_v2 --export-only
```

Sau khi đủ 9 file `.pte`:

```bash
!python -u -m src.evaluate --config configs/kaggle.yaml
```

## 6. Tải kết quả khỏi Kaggle

Trong panel Output, tải:

```text
/kaggle/working/waste-edge-complete.zip
/kaggle/working/kaggle-training.log
```

File ZIP gồm checkpoint, 9 model `.pte`, metadata operator/delegation, metric chất lượng, audit, split và config. Với một run dài, nên bấm **Save Version** sau mỗi kiến trúc để Kaggle lưu output.

## 7. Benchmark đúng trên MacBook Air M3

Không dùng latency Kaggle để trả lời hiệu suất trên M3. Sau khi tải ZIP về, chép các thư mục `checkpoints`, `exported`, `results`, `data`, `configs` vào đúng project này. Kích hoạt môi trường local và chạy:

```bash
source .venv/bin/activate
python -u -m src.benchmark --config configs/project.yaml --conditions "AC power; background apps closed; room temperature recorded"
python -u -m src.report --config configs/project.yaml --threads 1
```

Benchmark dùng batch 1, warm-up 30, 200 lượt đo, 3 đợt, threads 1 và 4; mỗi tổ hợp chạy trong process mới. Kết quả chính:

```text
results/quality_metrics.csv
results/mac_m3_benchmark.csv
results/summary.csv
docs/REPORT.md
figures/
```

## 8. Điều chỉnh khi GPU thiếu bộ nhớ

Chỉ giảm `train.batch_size` từ 64 xuống 32 hoặc 16 và `qat.batch_size` từ 32 xuống 16 trong `configs/kaggle.yaml`. Không đổi preprocessing, classes, seed hoặc manifests giữa các mô hình. Batch size thay đổi phải được ghi trong báo cáo.

## Cấu trúc chính

```text
configs/                 cấu hình local và Kaggle
data/data_audit.csv      audit ảnh đã khóa
data/splits/             train/val/test/calibration cố định
src/train_fp32.py        fine-tune baseline
src/ptq_int8.py          PTQ static W8A8
src/qat_int8.py          QAT CUDA, convert/export CPU
src/export_edge.py       ExecuTorch/XNNPACK `.pte`
src/evaluate.py          đánh giá 9 artifact triển khai
src/benchmark.py         benchmark CPU M3
src/report.py            bảng, hình và report scaffold
tests/                   kiểm tra leakage, preprocessing, runtime
```

## Kiểm tra code

```bash
python -m pytest -q
```

Các kiểm tra xác nhận split không giao nhau, calibration chỉ thuộc train, preprocessing validation xác định, số thread ExecuTorch đúng, và 9 artifact smoke có output hữu hạn đúng 9 lớp.
