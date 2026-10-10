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

Dataset của bạn đã nằm trong **Input** của notebook. Dùng trực tiếp cấu trúc này:

```text
/kaggle/input/watse_set/realwaste-main/RealWaste/
  Cardboard/
  Food Organics/
  Glass/
  Metal/
  Miscellaneous Trash/
  Paper/
  Plastic/
  Textile Trash/
  Vegetation/
```

`configs/kaggle.yaml` đã đặt `data.archive: null` và `data.root` trỏ thẳng tới `RealWaste`. Không cần nén hoặc chép ảnh sang `/kaggle/working`. Nếu slug thực tế khác `watse_set`, sửa `data.root` cho đúng thư mục chứa 9 lớp.

Tạo Kaggle Notebook, chọn:

- Accelerator: GPU (T4 x2, P100 hoặc GPU khả dụng).
- Internet: On, vì TorchVision cần tải pretrained ImageNet lần đầu.
- Persistence: nên bật để giảm rủi ro mất checkpoint.

Kiểm tra GPU trong một cell:

```bash
!nvidia-smi
```

## 2. Clone hoặc cập nhật code từ GitHub

Chạy cell sau khi thư mục source chưa tồn tại:

```bash
%cd /kaggle/working
!git clone https://github.com/xmongng/deeplearning.git waste-edge-quantization
%cd /kaggle/working/waste-edge-quantization
```

Nếu đã clone từ lần trước, dùng cell này thay cho clone:

```bash
%cd /kaggle/working/waste-edge-quantization
!git pull --ff-only origin main
```

`/kaggle/input` chứa ảnh đã có của bạn. `/kaggle/working/waste-edge-quantization` chứa code, checkpoint và kết quả được tạo khi chạy.

## 3. Cài và kiểm tra môi trường

Chạy cell sau, rồi **Restart Session** nếu Kaggle yêu cầu do thay phiên bản PyTorch:

```bash
!python -m pip install --no-cache-dir -r requirements-kaggle.txt
```

Sau khi restart, `%cd` lại thư mục dự án và kiểm tra:

```bash
%cd /kaggle/working/waste-edge-quantization
!python -c "import torch,torchvision,torchao,executorch; print(torch.__version__, torchvision.__version__, torchao.__version__); print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

Không tiếp tục nếu cài thư viện thất bại, import lỗi hoặc `CUDA: False`. Các phiên bản được pin trong `requirements-kaggle.txt`; phiên Kaggle thực tế chưa được kiểm chứng trong lần sửa này. Nếu pip báo không tìm thấy phiên bản hoặc xung đột, lưu nguyên lỗi để sửa môi trường trước khi train.

Kiểm tra dependency sau khi cài:

```bash
!python -m pip check
```

## 4. Kiểm tra dữ liệu trước khi train

Chạy cell Python để kiểm tra đường dẫn và tên lớp. Nếu tên hiển thị trong Input khác đường dẫn mount thực tế, cell sẽ tìm duy nhất thư mục `RealWaste` chứa đủ 9 lớp và cập nhật `data.root`:

```python
from pathlib import Path
import yaml

config_path = Path("configs/kaggle.yaml")
cfg = yaml.safe_load(config_path.read_text())
root = Path(cfg["data"]["root"])
if not all((root / name).is_dir() for name in cfg["classes"]):
    candidates = [p for p in Path("/kaggle/input").rglob("RealWaste")
                  if p.is_dir() and all((p / name).is_dir() for name in cfg["classes"])]
    assert len(candidates) == 1, f"Cần chọn đúng data.root; tìm thấy: {candidates}"
    root = candidates[0]
    cfg["data"]["root"] = str(root)
    cfg["data"]["archive"] = None
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
print("Dataset:", root)
for name in cfg["classes"]:
    assert (root / name).is_dir(), f"Thiếu thư mục lớp: {name}"
    print(name, len([p for p in (root / name).rglob("*")
                     if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]))
```

Sau đó đối chiếu tên file và SHA-256 với audit đã khóa:

```bash
!python -m src.stage_runner verify_data --config configs/kaggle.yaml
```

Chỉ tiếp tục khi thấy `Verified 4752 RealWaste images`. Nếu có `missing`, `extra` hoặc `changed`, dataset chưa khớp bản đã audit; kiểm tra đúng phiên bản dữ liệu trước khi train. File phụ như README không được tính là ảnh. Không chạy lại `audit_data` hoặc `prepare_data` trên giao thức đã khóa.

## 5. Chạy từng bước

Chạy lần lượt từng cell dưới đây cho MobileNetV2. Đợi mỗi lệnh thành công trước khi chuyển bước tiếp theo.

**5.1. Train FP32 trên GPU:**

```bash
!python -u -m src.stage_runner train_fp32 --config configs/kaggle.yaml --model mobilenet_v2 --device cuda
```

Sau train, script tự đánh giá checkpoint FP32 tốt nhất trên **train và validation**, cùng preprocessing xác định và eval mode. Kết quả nằm trong `results/baseline_validation/mobilenet_v2/`. Đây là đánh giá baseline trước quantization, chưa sử dụng test.

Nếu training đã hoàn tất nhưng bước đánh giá baseline bị ngắt, chạy riêng:

```bash
!python -u -m src.stage_runner evaluate_checkpoint --config configs/kaggle.yaml --model mobilenet_v2
```

**5.2. Export FP32 sang ExecuTorch:**

```bash
!python -u -m src.stage_runner export_edge --config configs/kaggle.yaml --model mobilenet_v2
```

**5.3. PTQ INT8 và export:**

```bash
!python -u -m src.stage_runner ptq_int8 --config configs/kaggle.yaml --model mobilenet_v2
```

**5.4. Train QAT trên GPU và export INT8:**

```bash
!python -u -m src.stage_runner qat_int8 --config configs/kaggle.yaml --model mobilenet_v2 --device cuda
```

Lặp lại cả bốn bước cho `mobilenet_v3_large`, rồi `shufflenet_v2_x1_0`, bằng cách thay giá trị `--model`. Tổng cộng cần 3 model × 3 phương pháp = 9 file `.pte`.

**5.5. Đánh giá cả 9 artifact:**

```bash
!python -u -m src.stage_runner evaluate --config configs/kaggle.yaml
```

**5.6. Đóng gói toàn bộ bằng chứng:**

```bash
!python -u -m src.package_results --output /kaggle/working/waste-edge-complete.zip
```

Script kiểm tra đủ 9 artifact và metric có cùng SHA-256 trước khi đóng gói. ZIP gồm source, requirements, cấu hình, audit/split, checkpoint, đồ thị export, lịch sử train, biểu đồ, dự đoán từng ảnh và log của các giai đoạn. `package_manifest.json` ghi hash từng file. Không chứa ảnh gốc trong Input. Nếu ZIP cùng tên đã tồn tại, dùng tên mới với `--output`.

## 6. Tùy chọn chạy tự động toàn bộ pipeline

Dùng tùy chọn này trên một project chưa có checkpoint/artifact. Nếu đã chạy mục 5, không chạy lại lệnh này.

Lệnh tự động:

```bash
!python -u -m src.stage_runner run_kaggle --config configs/kaggle.yaml 2>&1 | tee /kaggle/working/kaggle-training.log
```

Pipeline lần lượt:

1. Đọc trực tiếp thư mục dữ liệu và đối chiếu SHA-256 của đủ 4.752 ảnh với audit đã khóa.
2. Fine-tune 3 baseline FP32 trên CUDA, vẽ đường cong và đánh giá checkpoint tốt nhất trên train/validation.
3. Export 3 FP32 sang ExecuTorch/XNNPACK.
4. PTQ W8A8 bằng 200 ảnh calibration train và export 3 model.
5. QAT trên CUDA từ đúng checkpoint FP32 tương ứng, đánh giá validation sau convert trên CPU, rồi export 3 model.
6. Đánh giá 9 file `.pte` trên cùng test set.
7. Tạo `/kaggle/working/waste-edge-complete.zip`.

Không chạy đồng thời hai model trong cùng GPU. Script chạy tuần tự để giảm nguy cơ hết VRAM và ghi log liên tục.

## 7. Khôi phục khi phiên bị ngắt

Nếu phiên Kaggle dừng giữa chừng, giữ hoặc khôi phục đủ thư mục `checkpoints`, `exported`, `results` của cùng lần chạy. Cài môi trường, kiểm tra đường dẫn Input và chạy lại bước xác minh dữ liệu ở mục 4.

Chỉ chạy bước chưa hoàn tất trong mục 5:

- FP32 có `training_summary.json` và `fp32_best.pth`: bỏ qua train FP32, chuyển tới export nếu chưa có artifact FP32.
- FP32 có checkpoint nhưng thiếu `training_summary.json`: training đã bị ngắt. Code hiện chưa hỗ trợ resume optimizer; dùng một bản project/output mới để train lại kiến trúc đó.
- Artifact `.pte` và metadata `.json` tương ứng đã đủ: bỏ qua bước export/PTQ đó.
- Artifact chỉ ghi được một phần: lưu riêng output của lần lỗi trước khi chạy lại trong một bản project/output mới.

Không chạy lại cả chuỗi train từ đầu trên checkpoint đã tồn tại: script sẽ từ chối ghi đè. Không trộn checkpoint, config hoặc manifest của các lần thí nghiệm khác nhau.

Nếu QAT đã train xong và có `qat_summary.json` nhưng export bị ngắt:

```bash
!python -u -m src.stage_runner qat_int8 --config configs/kaggle.yaml --model mobilenet_v2 --export-only
```

Sau khi đủ 9 file `.pte`:

```bash
!python -u -m src.stage_runner evaluate --config configs/kaggle.yaml
```

## Kiểm tra parity khi export và xử lý lỗi PTQ cũ

Với FP32, giữ kiểm tra từng logit với `rtol=0.001`, `atol=0.001`. Smoke test cũng kiểm tra số học nghiêm ngặt.

Với PTQ/QAT INT8, mức `rtol=0.001`, `atol=0.15` được giữ để ghi nhận sai lệch số học, nhưng không còn dừng ngay khi một logit vượt mức này. Trước khi chấp nhận artifact, code so sánh **toàn bộ validation** giữa model đã convert trước export và `.pte` chạy XNNPACK. Mặc định **mọi dự đoán top-1 validation phải trùng nhau**, và output phải đúng shape, hữu hạn. Độ lệch Accuracy/Macro-F1 cùng mọi logit được ghi lại. Test chưa được dùng ở bước này.

Vì vậy sai lệch `0.159983 > 0.15` có thể được chấp nhận khi tất cả dự đoán validation vẫn trùng; nếu có dự đoán validation khác nhau, export vẫn thất bại và cần phân tích bằng chứng. Điều này không khẳng định một artifact cụ thể đã đạt khi chưa chạy kiểm tra thật.

Mỗi lần thử lưu vào:

```text
results/export_validation/<model>/<method>/<thời điểm UTC>/
  export_report.json       passed/failed, exception, cấu hình, operator/delegate và hash
  parity_samples.json     từng ảnh: logits trước/sau export, sai lệch, nhãn/dự đoán
  parity_summary.json     tổng hợp mức sai lệch, agreement, Accuracy/Macro-F1 validation
  <method>.pte            candidate giữ lại để chẩn đoán
  <method>.pt2
  <method>_edge_graph.txt
  legacy_incomplete/      artifact dở từ code cũ, nếu có
```

Chỉ candidate đạt kiểm tra mới được đưa vào `exported/<model>/`; metadata `.json` được ghi cuối cùng để đánh dấu hoàn tất. Nếu thất bại, artifact dở không chiếm tên kết quả chính, nên có thể chạy lại sau khi xử lý nguyên nhân.

Nếu code cũ đã để lại `exported/mobilenet_v2/ptq.pte` nhưng chưa có `ptq.json`, code mới tự chuyển bộ file dở vào `legacy_incomplete/` để bảo toàn bằng chứng rồi thử export lại. Artifact đã có đủ `.pte` và `.json` vẫn được bảo vệ khỏi ghi đè.

Cập nhật source rồi chạy lại PTQ mà không train FP32 lại:

```bash
%cd /kaggle/working/waste-edge-quantization
!git pull --ff-only origin main
!python -u -m src.stage_runner ptq_int8 --config configs/kaggle.yaml --model mobilenet_v2
```

Nếu lỗi ở QAT export nhưng training đã hoàn tất và có `qat_summary.json`:

```bash
!python -u -m src.stage_runner qat_int8 --config configs/kaggle.yaml --model mobilenet_v2 --export-only
```

Giữ nguyên các cảnh báo thư viện để phục vụ chẩn đoán. Không coi một warning là nguyên nhân làm chương trình dừng nếu traceback chỉ ra exception khác.

## 8. Tải kết quả khỏi Kaggle

Trong panel Output, tải file ZIP. File log bên dưới chỉ có khi dùng pipeline tự động:

```text
/kaggle/working/waste-edge-complete.zip
/kaggle/working/kaggle-training.log
```

File ZIP gồm toàn bộ bằng chứng ở mục dưới, checkpoint, 9 model `.pte` và source để tái lập. Với một run dài, nên bấm **Save Version** sau mỗi kiến trúc để Kaggle lưu output.

## Bằng chứng được lưu ở đâu?

Các lệnh trong README dùng `src.stage_runner` để lưu cả stdout/stderr, kể cả lỗi import hoặc thông báo native runtime. Các lệnh CLI trực tiếp của các giai đoạn cũng tự bật ghi log sau khi import thành công. Pipeline tự động gọi wrapper cho từng bước con.

```text
results/
  evidence/<model hoặc all>/<stage>/<thời điểm UTC>/
    console.log                 toàn bộ stdout/stderr của lần chạy
    stage.json                  lệnh, thời gian, exit code, trạng thái, phiên bản,
                                Git commit, hash source/input và output mới tạo
    config.yaml                 bản chụp config khi bắt đầu bước
  training/<model>/fp32/         tương tự cho qat/
    run_config.json             config, epoch thực tế, optimizer, scheduler,
                                class weights, thiết bị, số tham số, môi trường
    architecture.txt            cấu trúc FP32 (QAT dùng prepared_graph.txt)
    epochs.csv                  metric train/val từng epoch để plot lại
    epochs.json                 cùng dữ liệu ở định dạng JSON
    summary.json                best epoch, số epoch đã chạy, thời gian, hash checkpoint
    training_curves.png         loss, accuracy, Macro-F1 và learning rate
    training_curves.pdf          bản vector dùng trong báo cáo
  baseline_validation/<model>/
    metrics.json                train/val của best FP32, gap train–val
    train_predictions.csv       nhãn thật/dự đoán, logits, xác suất từng lớp
    val_predictions.csv
    {train,val}_per_class.json   precision/recall/F1/support từng lớp
    {train,val}_confusion.*      ma trận JSON và hình PNG/PDF
  export_validation/<model>/<method>/<thời điểm UTC>/
    export_report.json, parity_samples.json, parity_summary.json
  quantization/<model>/ptq/
    run_config.json              nguồn checkpoint và cấu hình calibration
    calibration.csv             tên/nhãn ảnh calibration, thời gian từng batch
    prepared_graph.txt          graph đã gắn observer
    observer_state.pth          thống kê observer sau calibration
    summary.json                 thời gian PTQ/export, operator, parity, hash artifact
  quality/<model>/
    <method>_metrics.json        metric test và thông tin artifact/config
    <method>_predictions.csv     từng ảnh: nhãn, logits, xác suất, confidence
    <method>_per_class.json      precision/recall/F1/support
    <method>_confusion.*         JSON, PNG và PDF
  quality_metrics.csv           bảng test tổng hợp của 9 artifact, gồm loss
  raw_logs/<thời điểm>/          log và số đo từng lượt benchmark
  packaging/<thời điểm>.json     đường dẫn, kích thước và SHA-256 của ZIP
checkpoints/<model>/
  fp32_best.pth, fp32_last.pth
  qat_best.pth, qat_last.pth
exported/<model>/
  {fp32,ptq,qat}.pte             artifact triển khai
  {fp32,ptq,qat}.pt2             graph trước lowering
  {fp32,ptq,qat}.json            cấu hình, môi trường, checkpoint/artifact hash,
                                thời gian export, operators/delegates và parity
  <method>_edge_graph.txt
```

Mỗi epoch hiện đầy đủ `train loss/acc/F1`, `val loss/acc/F1`, learning rate và thời gian. Với QAT, validation được đo trên model đã convert INT8 chạy CPU; train dùng fake quantization. Lịch sử QAT ghi cả trạng thái freeze observer/BatchNorm. PTQ chỉ calibration, nên không có đường cong training loss.

Loss là weighted cross-entropy, tổng hợp theo tổng trọng số nhãn; accuracy dùng %, Macro-F1 dùng thang 0–1. Train mỗi epoch có augmentation, dropout và cập nhật trọng số, còn validation ở eval mode, nên không diễn giải chênh lệch một epoch như bằng chứng chắc chắn overfit. Dùng xu hướng qua nhiều epoch và đánh giá train/val xác định của best FP32 để phân tích.

Lịch sử được ghi sau mỗi epoch hoàn tất; nếu lỗi giữa epoch, console log và `stage.json` giữ bằng chứng lỗi, nhưng epoch dở chưa có đủ metric. Nếu runtime bị tắt cưỡng bức, bản ghi có thể còn `status: running`, không được coi là hoàn tất. Checkpoint `last` lưu optimizer, scheduler (FP32), RNG và trạng thái generator của DataLoader; code hiện chưa có lệnh resume tự động.

Vẽ lại đường cong mà không train lại:

```bash
!python -m src.stage_runner training_history --model mobilenet_v2 --phase fp32
!python -m src.stage_runner training_history --model mobilenet_v2 --phase qat
```

Các log bổ sung chỉ có cho lần chạy bằng code mới. Những lần đã chạy trước đó vẫn có `checkpoints/<model>/history.csv` hoặc `qat_history.csv`, nhưng không thể khôi phục console log hay thông số chưa từng được lưu. Nếu đang train bằng bản cũ, giữ kết quả hiện tại trước khi cập nhật source.

## 9. Benchmark đúng trên MacBook Air M3

Không dùng latency Kaggle để trả lời hiệu suất trên M3. Sau khi tải ZIP về, chép các thư mục `checkpoints`, `exported`, `results`, `data`, `configs` vào đúng project này. Nếu chưa có môi trường local, tạo và cài trước:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip check
```

Sau đó chạy (benchmark dùng tensor mẫu, không cần chép ảnh Input về Mac):

```bash
source .venv/bin/activate
python -u -m src.stage_runner benchmark --config configs/project.yaml --conditions "AC power; background apps closed; room temperature recorded"
python -u -m src.stage_runner report --config configs/project.yaml --threads 1
```

Benchmark dùng batch 1, warm-up 30, 200 lượt đo, 3 đợt, threads 1 và 4; mỗi tổ hợp chạy trong process mới. Kết quả chính:

```text
results/quality_metrics.csv
results/mac_m3_benchmark.csv
results/summary.csv
docs/REPORT.md
figures/
```

## 10. Điều chỉnh khi GPU thiếu bộ nhớ

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
# Hoặc chỉ kiểm tra ghi log/lịch sử với unittest:
python -m unittest discover -s tests -p test_training_evidence.py -v
python -m unittest discover -s tests -p test_export_validation.py -v
```

Bộ test đầy đủ cần dữ liệu local tại `data/raw` cho kiểm tra preprocessing; các kiểm tra smoke được bỏ qua nếu chưa có artifact CP0. Chỉ kiểm tra phần xác minh và giao thức dữ liệu bằng `python -m pytest -q tests/test_verify_data.py tests/test_data_protocol.py`.

Các kiểm tra xác nhận split không giao nhau, calibration chỉ thuộc train, preprocessing validation xác định, số thread ExecuTorch đúng, và 9 artifact smoke có output hữu hạn đúng 9 lớp.
