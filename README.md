# Dự đoán giá chứng khoán VN30

Đồ án xây dựng pipeline dự báo giá đóng cửa cho 30 mã VN30 bằng RNN, LSTM và GRU. Dữ liệu đầu vào là `Dataset_VN30_2016_2026.csv`; kết quả chạy, metrics, model và biểu đồ được lưu trong thư mục `results/`.

## Nội dung chính

- `main.py`: pipeline huấn luyện và đánh giá mô hình.
- `config.py`: cấu hình danh sách feature, scenario, model và hyperparameter.
- `rebuild_metrics.py`: rebuild lại metrics từ các model/artifact đã huấn luyện.
- `visualize_model_report.py`: tạo biểu đồ trực quan hóa từ `results/merged_metrics_ALL.csv`.
- `CRAWVN30.py`: script thu thập dữ liệu ban đầu bằng `vnstock`.
- `DACS_2386400763_NguyenLePhuAn.pdf`: file báo cáo PDF.
- `results/`: kết quả chạy thực nghiệm, model, prediction, metrics và chart.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy pipeline

```bash
python main.py
```

## Rebuild metrics khi đã có artifact

```bash
python rebuild_metrics.py --results-dir results --write-predictions
```

## Vẽ biểu đồ báo cáo

```bash
python visualize_model_report.py --results-dir results --max-detail-plots 0
```

## Ghi chú dữ liệu

File dữ liệu thô có 68,069 dòng cho 30 mã VN30. Trong lần chạy cuối, pipeline dùng strict common dates nên vùng dữ liệu chung sau feature engineering là từ 2021-04-20 đến 2026-04-17.

